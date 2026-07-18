"""Proximal Policy Optimization (PPO) for discrete action spaces with masking.

Phase 1: PPO agent for 5G-TSN scheduling — discrete-native, mirrors the SAC
pipeline (same env, same obs/action, same reward) so the two are directly
comparable.

Design
------
- Actor:  MLP -> logits -> masked categorical policy
- Critic: MLP -> scalar V(s)
- Masking: invalid actions get logits = -1e9 before softmax (zero prob).
- Advantage: GAE(lambda) over on-policy rollouts.
- Loss: clipped surrogate + value MSE + entropy bonus.

PPO is on-policy: we collect a rollout (N full episodes), do K epochs of
minibatch updates, then discard the rollout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.optim import Adam

from .sac import MLP


class PPOAgent:
    """PPO agent (discrete, masked)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_dims: tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        n_steps: int = 5,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        value_coef: float = 0.5,
        grad_clip: float = 0.5,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        self.actor = MLP(obs_dim, hidden_dims, output_dim=n_actions,
                         use_layer_norm=True).to(self.device)
        self.critic = MLP(obs_dim, hidden_dims, output_dim=1,
                          use_layer_norm=True).to(self.device)

        self.opt_actor = Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = Adam(self.critic.parameters(), lr=lr)

        self._reset_rollout()

    # --- rollout storage ------------------------------------------------

    def _reset_rollout(self) -> None:
        self._obs: list[np.ndarray] = []
        self._actions: list[int] = []
        self._logps: list[float] = []
        self._rewards: list[float] = []
        self._values: list[float] = []
        self._next_values: list[float] = []
        self._dones: list[bool] = []
        self._masks: list[np.ndarray] = []

    # --- n-step target computation --------------------------------------

    def _nstep_targets(
        self,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """Compute n-step bootstrapped return-to-go for every step t.

        g_t = sum_{k=0}^{n-1} gamma^k * r_{t+k}  (truncated at episode end)
              + gamma^n * V(s_{t+n})             (0 if a done is hit first)

        next_values[t] holds V(s_{t+1}), so V(s_{t+n}) = next_values[t+n-1].
        Episode boundaries are respected via the `dones` flags (no cross-episode
        bootstrapping).
        """
        T = len(rewards)
        g = torch.zeros(T, device=self.device)
        for t in range(T):
            G = 0.0
            power = 1.0
            crossed = False
            for k in range(n):
                idx = t + k
                if idx >= T:
                    break
                G += power * float(rewards[idx])
                if bool(dones[idx]):
                    crossed = True
                    break
                power *= self.gamma
            boot = 0.0
            if not crossed and (t + n - 1) < T:
                boot = (self.gamma ** n) * float(next_values[t + n - 1])
            g[t] = G + boot
        return g

    # --- inference -------------------------------------------------------

    @staticmethod
    def _mask_logits(logits: torch.Tensor, mask: torch.Tensor,
                     invalid: float = -1e9) -> torch.Tensor:
        return torch.where(mask, logits, torch.full_like(logits, invalid))

    def value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return float(self.critic(s).item())

    def select_action(
        self, state: np.ndarray, mask: np.ndarray, deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Return (action, log_prob, value)."""
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
            logits = self.actor(s)
            masked = self._mask_logits(logits, m)
            value = self.critic(s).item()
            if deterministic:
                action = masked.argmax(dim=-1)
                logp = F.log_softmax(masked, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
            else:
                dist = torch.distributions.Categorical(logits=masked)
                action = dist.sample()
                logp = dist.log_prob(action)
            return int(action.item()), float(logp.item()), float(value)

    def store_step(
        self, state, action, logp, reward, value, next_value, done, mask,
    ) -> None:
        self._obs.append(state.astype(np.float32))
        self._actions.append(int(action))
        self._logps.append(float(logp))
        self._rewards.append(float(reward))
        self._values.append(float(value))
        self._next_values.append(float(next_value))
        self._dones.append(bool(done))
        self._masks.append(mask.copy())

    # --- update ----------------------------------------------------------

    def update(self, epochs: int = 4, minibatch: int = 256) -> dict[str, float]:
        if len(self._obs) == 0:
            return {}

        obs = torch.from_numpy(np.stack(self._obs)).float().to(self.device)
        actions = torch.tensor(self._actions, dtype=torch.long, device=self.device)
        old_logps = torch.tensor(self._logps, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self._rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self._values, dtype=torch.float32, device=self.device)
        next_values = torch.tensor(self._next_values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self._dones, dtype=torch.bool, device=self.device)
        masks = torch.from_numpy(np.stack(self._masks)).bool().to(self.device)

        # --- n-step bootstrapped targets + GAE(lambda) advantages ---
        # Credit-assignment fix (Phase 1.5 / approach A):
        #   * gamma raised so head-of-episode decisions receive gradient from the
        #     final peak (episode-level MAX statistic).
        #   * n-step target accumulates n future rewards + gamma^n * V(s_{t+n}),
        #     so the "reserve this cell for a later packet" signal propagates
        #     across multiple steps instead of dying at gamma^316 ~ 0.
        T = len(rewards)
        n = self.n_steps
        if n and n > 1:
            g = self._nstep_targets(rewards, next_values, dones, n)
        else:
            # 1-step TD target: g_t = r_t + gamma * V(s_{t+1}) (0 at terminal)
            g = torch.where(dones, rewards, rewards + self.gamma * next_values)

        advantages = torch.zeros(T, device=self.device)
        last_adv = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                last_adv = 0.0
            delta = g[t] - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * last_adv
            advantages[t] = last_adv
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO epochs over minibatches ---
        idx = np.arange(T)
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, T, minibatch):
                mb = idx[start:start + minibatch]
                if len(mb) == 0:
                    continue
                logits = self.actor(obs[mb])
                masked = self._mask_logits(logits, masks[mb])
                dist = torch.distributions.Categorical(logits=masked)
                new_logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logps[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * advantages[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss = policy_loss - self.ent_coef * entropy

                value_pred = self.critic(obs[mb]).squeeze(-1)
                value_loss = F.mse_loss(value_pred, returns[mb])

                loss = actor_loss + self.value_coef * value_loss

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn_utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                nn_utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.opt_actor.step()
                self.opt_critic.step()

                actor_losses.append(policy_loss.item())
                critic_losses.append(value_loss.item())
                entropies.append(entropy.item())

        self._reset_rollout()
        return {
            "policy_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    # --- save / load -----------------------------------------------------

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
