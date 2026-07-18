"""Double Deep Q-Network (Double-DQN) for discrete action spaces with masking.

Phase 3: Value-based RL agent for 5G-TSN scheduling.

Architecture
------------
- Online net:  MLP → Q-values for ALL actions (shape: (B, n_actions))
- Target net:  Soft-updated copy of online net (Polyak averaging)
- Double-DQN trick: online net selects action (argmax), target net evaluates Q

Key formulas
------------
Q_target:
    a* = argmax_a Q_online(s', a)           # action selection by online net
    Q_tgt(s, a) = r + gamma * (1 - done) * Q_target(s', a*)   # evaluation by target net

This decouples selection and evaluation, reducing overestimation bias
(compared to vanilla DQN where both use the same network).

Action masking
--------------
Invalid actions get Q-values = -1e9 before argmax, ensuring they are
never selected and their Q-targets are never used for updates.

Epsilon-greedy + masking
------------------------
During training, with probability epsilon the agent picks a RANDOM VALID
action (from the mask), not just any random action. This ensures
exploration stays within feasible actions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .sac import MLP, ReplayBuffer


# =====================================================================
#  Double-DQN Agent
# =====================================================================

class DDQNAgent:
    """Double-DQN agent for discrete action spaces with action masking.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    n_actions : int
        Number of discrete actions.
    hidden_dims : tuple
        Hidden layer sizes for Q-networks.
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    tau : float
        Soft target update rate (Polyak averaging). 0 = hard update every
        `target_update_freq` steps; >0 = gradual soft update every step.
    epsilon_start : float
        Initial exploration rate.
    epsilon_end : float
        Final exploration rate.
    epsilon_decay_steps : int
        Number of steps for epsilon linear decay from start to end.
    target_update_freq : int
        Hard target update frequency (in training steps). Only used when
        tau=0. When tau>0, soft update is done every step instead.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_dims: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        target_update_freq: int = 500,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # --- Networks ---
        self.q_net = MLP(obs_dim, hidden_dims, output_dim=n_actions,
                         use_layer_norm=True).to(self.device)
        self.q_target = MLP(obs_dim, hidden_dims, output_dim=n_actions,
                            use_layer_norm=True).to(self.device)
        # Hard copy initially
        self.q_target.load_state_dict(self.q_net.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.opt = Adam(self.q_net.parameters(), lr=lr)

        self._train_steps = 0  # total gradient updates
        self._total_steps = 0  # total env steps (for epsilon decay)

    # --- epsilon schedule ------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Linear decay from epsilon_start to epsilon_end."""
        if self._total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        progress = self._total_steps / self.epsilon_decay_steps
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    # --- action selection ------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Select an action given state and action mask.

        When deterministic=True: always pick argmax of masked Q-values.
        When deterministic=False: epsilon-greedy with random VALID actions.
        """
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)

            q_vals = self.q_net(s)  # (1, n_actions)
            masked_q = self._mask_q(q_vals, m)  # invalid → -1e9

            if deterministic:
                action = int(masked_q.argmax(dim=-1).item())
            else:
                eps = self.epsilon
                valid_indices = np.where(mask)[0]
                if len(valid_indices) == 0:
                    # No valid actions — shouldn't happen in our env
                    action = 0
                elif np.random.random() < eps:
                    action = int(np.random.choice(valid_indices))
                else:
                    action = int(masked_q.argmax(dim=-1).item())

            return action

    # --- masking ---------------------------------------------------------

    @staticmethod
    def _mask_q(q_vals: torch.Tensor, mask: torch.Tensor,
                invalid: float = -1e9) -> torch.Tensor:
        """Mask invalid actions by setting their Q-values to a large negative."""
        return torch.where(mask, q_vals, torch.full_like(q_vals, invalid))

    # --- training step ---------------------------------------------------

    def update(self, buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        """One Double-DQN gradient update from the replay buffer.

        Returns a dict of loss/metric values for logging.
        """
        if len(buffer) < batch_size:
            return {}

        batch = buffer.sample(batch_size)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device)
        next_mask = batch["next_mask"].to(self.device)

        # ---- Double-DQN target ----
        with torch.no_grad():
            # Step 1: online net selects best action in next state
            next_q_online = self.q_net(next_state)  # (B, n_actions)
            next_q_online_masked = self._mask_q(next_q_online, next_mask)
            best_next_action = next_q_online_masked.argmax(dim=-1)  # (B,)

            # Step 2: target net evaluates Q-value of that action
            next_q_target = self.q_target(next_state)  # (B, n_actions)
            next_q_target_masked = self._mask_q(next_q_target, next_mask)
            # Gather Q_target(s', a*) where a* is chosen by online net
            next_q_best = next_q_target_masked.gather(
                1, best_next_action.unsqueeze(-1)).squeeze(-1)  # (B,)

            # Q_target = r + gamma * (1 - done) * Q_target(s', a*)
            q_target = reward + self.gamma * (1.0 - done) * next_q_best

        # ---- Update online net ----
        q_pred = self.q_net(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(q_pred, q_target)

        self.opt.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.opt.step()

        # ---- Target network update ----
        self._train_steps += 1
        if self.tau > 0:
            # Soft (Polyak) update every training step
            with torch.no_grad():
                for p, p_t in zip(self.q_net.parameters(),
                                  self.q_target.parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
        elif self._train_steps % self.target_update_freq == 0:
            # Hard update every N steps
            self.q_target.load_state_dict(self.q_net.state_dict())

        self._total_steps += 1  # approximate: 1 env step ≈ 1 train step when update_every=1

        return {
            "q_loss": float(loss.item()),
            "q_value": float(q_pred.mean().item()),
            "epsilon": self.epsilon,
        }

    def increment_total_steps(self, n: int = 1) -> None:
        """Manually increment total steps counter (for epsilon decay tracking).

        Called from the training loop after each env step, even when no
        gradient update happens (e.g., during warmup).
        """
        self._total_steps += n

    # --- save / load -----------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "train_steps": self._train_steps,
            "total_steps": self._total_steps,
        }, path)

    def load(self, path: str) -> None:
        """Load model weights from a file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self._train_steps = ckpt.get("train_steps", 0)
        self._total_steps = ckpt.get("total_steps", 0)


# =====================================================================
#  BC warm-start for DQN
# =====================================================================

def bc_train_dqn(
    agent: "DDQNAgent",
    demo: dict,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = True,
) -> dict:
    """Supervised pre-training of the Q-network on (obs, mask, action) demos.

    The Q-network outputs raw Q-values for ALL actions; we treat them as
    logits and apply a masked categorical cross-entropy so that the teacher
    action gets the highest Q. The target network is synced after training
    so evaluation starts from the warm-started Q-values.

    Returns a small stats dict.
    """
    device = agent.device
    obs = torch.from_numpy(demo["obs"]).float().to(device)
    masks = torch.from_numpy(demo["mask"]).bool().to(device)
    actions = torch.from_numpy(demo["action"]).long().to(device)
    N = len(actions)

    opt = Adam(agent.q_net.parameters(), lr=lr)
    idx = np.arange(N)
    last_acc = 0.0
    for ep in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, N, batch_size):
            mb = idx[s:s + batch_size]
            if len(mb) == 0:
                continue
            q_vals = agent.q_net(obs[mb])
            masked = DDQNAgent._mask_q(q_vals, masks[mb])
            logp = F.log_softmax(masked, dim=-1)
            loss = -logp.gather(-1, actions[mb].unsqueeze(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            q_vals = agent.q_net(obs)
            masked = DDQNAgent._mask_q(q_vals, masks)
            pred = masked.argmax(dim=-1)
            last_acc = float((pred == actions).float().mean().item())

        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"  [BC-DQN] epoch {ep+1:>3d}/{epochs}  "
                  f"greedy_match={last_acc:.3f}")

    # Sync target network with the warm-started online net
    agent.q_target.load_state_dict(agent.q_net.state_dict())
    return {"bc_epochs": epochs, "bc_samples": N, "greedy_match": last_acc}
