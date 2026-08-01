"""Soft Actor-Critic (SAC) for discrete action spaces with action masking.

Phase 2: SAC agent implementation for 5G-TSN scheduling.

Architecture
------------
- Actor:  MLP → logits → masked categorical policy
- Critic: Twin Q-networks (Q1, Q2), each outputs Q-values for ALL actions
- Target: Soft-updated copies of Q1, Q2 (Polyak averaging)
- Alpha:  Automatic temperature tuning via entropy constraint

Key formulas (discrete SAC)
---------------------------
Policy loss:
    J_pi = E_s [ Σ_a pi(a|s) * (alpha * log_pi(a|s) - min(Q1(s,a), Q2(s,a))) ]

Q target:
    V(s') = Σ_a pi(a|s') * (min(Q1_tgt(s',a), Q2_tgt(s',a)) - alpha * log_pi(a|s'))
    Q_tgt(s,a) = r + gamma * (1 - done) * V(s')

Alpha loss:
    J_alpha = -alpha * (log_pi(a|s) + H_target).detach()

Action masking
--------------
Invalid actions get logits = -1e9 before softmax, ensuring zero probability.
Both actor sampling and critic V-computation respect the mask.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# =====================================================================
#  Replay Buffer
# =====================================================================

class Transition(NamedTuple):
    """A single transition stored in the replay buffer."""
    state: np.ndarray         # (obs_dim,)
    action: int               # scalar
    reward: float             # scalar
    next_state: np.ndarray    # (obs_dim,)
    done: bool                # scalar
    action_mask: np.ndarray   # (n_actions,) bool — mask when action was taken
    next_mask: np.ndarray     # (n_actions,) bool — mask at next state


class ReplayBuffer:
    """Fixed-size circular replay buffer with batch sampling."""

    def __init__(self, capacity: int = 50_000):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Random sample a batch of transitions.

        Returns dict with keys: state, action, reward, next_state,
        done, action_mask, next_mask — all as tensors.
        """
        indices = random.sample(range(len(self._buf)), min(batch_size, len(self._buf)))
        batch = [self._buf[i] for i in indices]

        return {
            "state":      torch.from_numpy(np.stack([t.state for t in batch])).float(),
            "action":     self._stack_actions([t.action for t in batch]),
            "reward":     torch.tensor([t.reward for t in batch], dtype=torch.float32),
            "next_state": torch.from_numpy(np.stack([t.next_state for t in batch])).float(),
            "done":       torch.tensor([t.done for t in batch], dtype=torch.float32),
            "action_mask":  torch.from_numpy(
                np.stack([t.action_mask for t in batch])).bool(),
            "next_mask":    torch.from_numpy(
                np.stack([t.next_mask for t in batch])).bool(),
        }

    def __len__(self) -> int:
        return len(self._buf)

    @staticmethod
    def _stack_actions(actions):
        """Stack actions into a tensor. Native: (B,) long. Dimred: (B, n_bits)."""
        if isinstance(actions[0], np.ndarray):
            return torch.from_numpy(np.stack(actions)).float()
        return torch.tensor(actions, dtype=torch.long)


# =====================================================================
#  Neural Networks
# =====================================================================

class MLP(nn.Module):
    """Simple multi-layer perceptron with optional layer norm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        output_dim: int = 0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        if output_dim > 0:
            layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Actor: state → logits over actions (pre-mask)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.n_actions = n_actions
        self.mlp = MLP(obs_dim, hidden_dims, output_dim=n_actions, use_layer_norm=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw logits (pre-mask). Shape: (B, n_actions)."""
        return self.mlp(state)

    @staticmethod
    def masked_logits(
        logits: torch.Tensor, mask: torch.Tensor, invalid_value: float = -1e9,
    ) -> torch.Tensor:
        """Apply action mask to logits."""
        return torch.where(mask, logits, torch.full_like(logits, invalid_value))

    def sample(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log-probabilities.

        Returns (action, log_prob) each of shape (B,).
        """
        logits = self.forward(state)
        masked = self.masked_logits(logits, mask)

        if deterministic:
            # Greedy: pick argmax of masked logits
            action = masked.argmax(dim=-1)
            log_prob = F.log_softmax(masked, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(logits=masked)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob

    def log_prob_all(
        self, state: torch.Tensor, mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log_pi(a|s) and pi(a|s) for ALL actions.

        Returns:
            log_probs: (B, n_actions) — log π(a|s) for each action
            probs:     (B, n_actions) — π(a|s) for each action
        """
        logits = self.forward(state)
        masked = self.masked_logits(logits, mask)
        log_probs = F.log_softmax(masked, dim=-1)
        probs = log_probs.exp()
        return log_probs, probs


class QNetwork(nn.Module):
    """Critic: state → Q-values for ALL actions. Shape: (B, n_actions)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.mlp = MLP(obs_dim, hidden_dims, output_dim=n_actions, use_layer_norm=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)


# =====================================================================
#  SAC Agent
# =====================================================================

class SACAgent:
    """Soft Actor-Critic for discrete action spaces with action masking.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    n_actions : int
        Number of discrete actions.
    hidden_dims : tuple
        Hidden layer sizes for all networks.
    lr : float
        Learning rate for all optimizers.
    gamma : float
        Discount factor.
    tau : float
        Soft target update rate (Polyak averaging).
    alpha_init : float
        Initial temperature parameter.
    auto_alpha : bool
        If True, automatically tune alpha.
    target_entropy_ratio : float
        Target entropy as a fraction of -log(n_actions).
        1.0 → -log(n_actions); 0.5 → half of that.
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
        alpha_init: float = 0.1,
        auto_alpha: bool = False,
        target_entropy_ratio: float = 0.5,
        max_alpha: float = 10.0,
        device: str = "cpu",
        action_mode: str = "native",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self._action_mode = action_mode
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # --- Networks ---
        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q1 = QNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q2 = QNetwork(obs_dim, n_actions, hidden_dims).to(self.device)

        # Target networks (hard copy initially)
        self.q1_target = QNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q2_target = QNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # --- Optimizers ---
        self.opt_policy = Adam(self.policy.parameters(), lr=lr)
        self.opt_q1 = Adam(self.q1.parameters(), lr=lr)
        self.opt_q2 = Adam(self.q2.parameters(), lr=lr)

        # --- Temperature (alpha) ---
        self.auto_alpha = auto_alpha
        self.max_alpha = max_alpha
        target_entropy = -target_entropy_ratio * np.log(n_actions)
        self.target_entropy = float(target_entropy)

        if auto_alpha:
            # log_alpha is the learnable parameter; alpha = exp(log_alpha)
            self.log_alpha = torch.tensor(
                np.log(alpha_init), dtype=torch.float32,
                requires_grad=True, device=self.device,
            )
            self.opt_alpha = Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor(
                np.log(alpha_init), dtype=torch.float32, device=self.device,
            )

        self._step = 0

    # --- properties ---

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    # --- action selection ---

    def select_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
    ):
        """Select an action. Native: returns int. Dimred: returns (bits, logp)."""
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            if self._action_mode == "dimred":
                logits = self.policy(s)                 # (1, n_bits)
                probs = torch.sigmoid(logits).squeeze(0)   # (n_bits,)
                if deterministic:
                    bits = (probs > 0.5).int().cpu().numpy()
                    logp = self._bits_logprob(probs, torch.from_numpy(bits).float())
                else:
                    dist = torch.distributions.Bernoulli(probs)
                    b = dist.sample()
                    bits = b.int().cpu().numpy()
                    logp = dist.log_prob(b).sum().item()
                return bits, logp
            m = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
            action, _ = self.policy.sample(s, m, deterministic=deterministic)
            return int(action.item())

    # --- training step ---

    def update(self, buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        """One gradient update step from the replay buffer.

        Returns a dict of loss/metric values for logging.
        """
        if len(buffer) < batch_size:
            return {}

        if self._action_mode == "dimred":
            return self._update_dimred(buffer, batch_size)

        batch = buffer.sample(batch_size)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device)
        mask = batch["action_mask"].to(self.device)
        next_mask = batch["next_mask"].to(self.device)

        # ---- Compute V(s') using target networks + current policy ----
        with torch.no_grad():
            # π(a'|s') and log π(a'|s') from *current* policy (not target)
            next_log_probs, next_probs = self.policy.log_prob_all(next_state, next_mask)

            # Q_target values for all actions
            q1_next = self.q1_target(next_state)   # (B, n_actions)
            q2_next = self.q2_target(next_state)
            q_next_min = torch.min(q1_next, q2_next)

            # V(s') = Σ_a π(a|s') * (Q(s',a) - α * log π(a|s'))
            alpha_val = self.log_alpha.exp()
            v_next = (next_probs * (q_next_min - alpha_val * next_log_probs)).sum(dim=-1)

            # Q target for the taken action
            q_target = reward + self.gamma * (1.0 - done) * v_next

        # ---- Update Q1 ----
        q1_pred = self.q1(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        q1_loss = F.mse_loss(q1_pred, q_target)
        self.opt_q1.zero_grad()
        q1_loss.backward()
        self.opt_q1.step()

        # ---- Update Q2 ----
        q2_pred = self.q2(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        q2_loss = F.mse_loss(q2_pred, q_target)
        self.opt_q2.zero_grad()
        q2_loss.backward()
        self.opt_q2.step()

        # ---- Update Policy ----
        log_probs, probs = self.policy.log_prob_all(state, mask)
        with torch.no_grad():
            q1_val = self.q1(state)
            q2_val = self.q2(state)
            q_min = torch.min(q1_val, q2_val)

        # J_pi = Σ_a π(a|s) * (α * log π(a|s) - Q(s,a))
        # Equivalently: E_a~π [α * log π(a|s) - Q(s,a)]
        policy_loss = (probs * (alpha_val * log_probs - q_min)).sum(dim=-1).mean()

        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        # ---- Update Alpha (temperature) ----
        # Always compute entropy for diagnostics (even with fixed alpha)
        with torch.no_grad():
            entropy = -(probs * log_probs).sum(dim=-1).mean()

        if self.auto_alpha:
            # Correct discrete SAC alpha loss:
            #   alpha_loss = log_alpha * (H - H_target).detach()
            # When H < H_target (too deterministic): gradient < 0 -> alpha increases
            # When H > H_target (too random):       gradient > 0 -> alpha decreases
            alpha_loss = (self.log_alpha * (entropy + self.target_entropy).detach()).mean()

            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()

            # Clip alpha to prevent explosion when target entropy is unreachable
            # (e.g., edge users with very few valid actions)
            with torch.no_grad():
                self.log_alpha.data.clamp_(min=np.log(0.01), max=np.log(self.max_alpha))

        # ---- Soft target update ----
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
            for p, p_t in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        self._step += 1

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "q1_value": float(q1_pred.mean().item()),
            "q2_value": float(q2_pred.mean().item()),
            "alpha": self.alpha,
            "entropy": float(entropy.item()),
        }

    # --- dimred helpers ------------------------------------------------

    @staticmethod
    def _bits_logprob(probs: torch.Tensor, bits: torch.Tensor) -> float:
        """Sum of independent Bernoulli log-probs for a 7-bit action."""
        return float(((bits) * torch.log(probs + 1e-8) +
                      (1 - bits) * torch.log(1 - probs + 1e-8)).sum().item())

    def _all_combos(self) -> torch.Tensor:
        """All 2^n_bits binary combinations, cached on device."""
        if getattr(self, "_combos", None) is None:
            n = self.n_actions
            idx = torch.arange(2 ** n, device=self.device)
            combos = torch.zeros(2 ** n, n, device=self.device)
            for i in range(n):
                combos[:, i] = ((idx >> (n - 1 - i)) & 1).float()
            self._combos = combos
        return self._combos

    def _update_dimred(self, buffer, batch_size):
        """Factored SAC update for the 7-bit action space.

        Actor outputs 7 per-bit Bernoulli logits; Q-networks output 7 per-bit
        Q-values with joint Q = Σ_i q_bit[i] * bit[i]. V(s') and the policy
        objective are computed by enumerating the 2^7 = 128 joint actions.
        """
        batch = buffer.sample(batch_size)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)        # (B, n_bits) 0/1
        reward = batch["reward"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device)
        alpha_val = self.log_alpha.exp()
        combos = self._all_combos()                      # (K, n_bits)

        # ---- V(s') over all joint actions ----
        with torch.no_grad():
            next_probs = torch.sigmoid(self.policy(next_state))          # (B, n)
            logp_c = (combos.unsqueeze(0) * torch.log(next_probs.unsqueeze(1) + 1e-8) +
                      (1 - combos).unsqueeze(0) *
                      torch.log(1 - next_probs.unsqueeze(1) + 1e-8)).sum(-1)
            logpi = logp_c                                      # (B, K)
            pi = torch.exp(logpi)
            qmin_bit = torch.min(self.q1_target(next_state),
                                 self.q2_target(next_state))     # (B, n)
            q_comb = (qmin_bit.unsqueeze(1) * combos.unsqueeze(0)).sum(-1)  # (B, K)
            v_next = (pi * (q_comb - alpha_val * logpi)).sum(-1)
            q_target = reward + self.gamma * (1.0 - done) * v_next

        # ---- Q updates (joint Q = Σ q_bit * bit) ----
        q1_pred = (self.q1(state) * action).sum(-1)
        q1_loss = F.mse_loss(q1_pred, q_target)
        self.opt_q1.zero_grad(); q1_loss.backward(); self.opt_q1.step()
        q2_pred = (self.q2(state) * action).sum(-1)
        q2_loss = F.mse_loss(q2_pred, q_target)
        self.opt_q2.zero_grad(); q2_loss.backward(); self.opt_q2.step()

        # ---- Policy update ----
        probs = torch.sigmoid(self.policy(state))               # (B, n)
        logp_cs = (combos.unsqueeze(0) * torch.log(probs.unsqueeze(1) + 1e-8) +
                   (1 - combos).unsqueeze(0) *
                   torch.log(1 - probs.unsqueeze(1) + 1e-8)).sum(-1)
        logpi_s = logp_cs
        pi_s = torch.exp(logpi_s)
        qmin_s = torch.min(self.q1(state), self.q2(state))
        q_comb_s = (qmin_s.unsqueeze(1) * combos.unsqueeze(0)).sum(-1)   # (B, K)
        policy_loss = (pi_s * (alpha_val * logpi_s - q_comb_s)).sum(-1).mean()

        self.opt_policy.zero_grad(); policy_loss.backward(); self.opt_policy.step()

        # ---- Alpha ----
        with torch.no_grad():
            entropy = -(pi_s * logpi_s).sum(-1).mean()
        if self.auto_alpha:
            alpha_loss = (self.log_alpha * (entropy + self.target_entropy).detach()).mean()
            self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()
            with torch.no_grad():
                self.log_alpha.data.clamp_(min=np.log(0.01), max=np.log(self.max_alpha))

        # ---- soft target ----
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
            for p, p_t in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
        self._step += 1
        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "q1_value": float(q1_pred.mean().item()),
            "q2_value": float(q2_pred.mean().item()),
            "alpha": self.alpha,
            "entropy": float(entropy.item()),
        }

    # --- save / load ---

    def save(self, path: str) -> None:
        """Save model weights to a file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.data,
            "step": self._step,
        }, path)

    def load(self, path: str) -> None:
        """Load model weights from a file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self._step = ckpt["step"]
