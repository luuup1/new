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
        n_cells: Optional[int] = None,
        n_links: int = 3,
        n_slots: int = 32,
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

        # n_cells: critic output dim (the FULL cell space). For native it equals
        # n_actions (96). For dimred the actor outputs n_actions=7 bits, but the
        # critic still evaluates the FULL 96-cell space so that the value
        # function loses no representational power (this is the fix for the
        # previously-failing factored-Q dimred).
        if n_cells is not None:
            self.n_cells = n_cells
        elif action_mode == "dimred":
            self.n_cells = n_links * n_slots
        else:
            self.n_cells = n_actions
        self.n_links = n_links
        self.n_slots = n_slots
        self.n_link_bits = int(np.ceil(np.log2(self.n_links)))
        if action_mode == "dimred":
            # dimred: n_actions IS the bit count (7); slot bits = total - link bits.
            self.n_slot_bits = n_actions - self.n_link_bits
        else:
            self.n_slot_bits = int(np.ceil(np.log2(self.n_slots)))

        # Precompute the canonical cell -> bits encoding matrix (n_cells x n_bits).
        # Matches env._cell_to_bits: n_link_bits link bits (MSB) + n_slot_bits slot
        # bits (MSB first). Used only in dimred mode: the n_bits Bernoulli policy
        # induces a distribution over the FULL 96-cell space, which the
        # (full-dimensional) critic then evaluates. This is the
        # "strategy-space reduction + full-value critic" fix that replaces the
        # previously-failing factored-Q dimred.
        n_bits = self.n_link_bits + self.n_slot_bits
        cell_bits = torch.zeros(self.n_cells, n_bits, dtype=torch.float32)
        for c in range(self.n_cells):
            link = c // self.n_slots + 1
            slot = c % self.n_slots
            link_code = link - 1
            cell_bits[c, 0] = (link_code >> 1) & 1
            cell_bits[c, 1] = link_code & 1
            for i in range(self.n_slot_bits):
                cell_bits[c, self.n_link_bits + i] = (
                    slot >> (self.n_slot_bits - 1 - i)) & 1
        self._cell_bits = cell_bits.to(self.device)

        # --- Networks ---
        # Actor outputs n_actions (96 native / 7 dimred); critics output n_cells.
        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q1 = QNetwork(obs_dim, self.n_cells, hidden_dims).to(self.device)
        self.q2 = QNetwork(obs_dim, self.n_cells, hidden_dims).to(self.device)

        # Target networks (hard copy initially)
        self.q1_target = QNetwork(obs_dim, self.n_cells, hidden_dims).to(self.device)
        self.q2_target = QNetwork(obs_dim, self.n_cells, hidden_dims).to(self.device)
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
        """Select an action. Native: returns int (cell). Dimred: also returns int
        (cell index), but the cell is sampled from a distribution induced by the
        7-bit Bernoulli policy (strategy-space reduction). Both return a plain
        int so the training loop / buffer treat them identically."""
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            if self._action_mode == "dimred":
                m = torch.from_numpy(mask).bool().to(self.device)   # (n_cells,)
                logits = self.policy(s)                             # (1, n_bits)
                bit_probs = torch.sigmoid(logits).squeeze(0)        # (n_bits,)

                # Induced cell distribution: pi(c|s) ∝ prod_i p_i^{b_i(c)} (1-p_i)^{1-b_i(c)}
                logp_cell = (
                    self._cell_bits * torch.log(bit_probs + 1e-8)
                    + (1 - self._cell_bits) * torch.log(1 - bit_probs + 1e-8)
                ).sum(-1)                                           # (n_cells,)

                # Mask invalid cells -> -1e9 (finite, avoids NaN when no cell is
                # valid), then normalize to a valid-cell distribution.
                masked_logp = torch.where(
                    m, logp_cell,
                    torch.full_like(logp_cell, -1e9),
                )
                if not bool(m.any()):
                    # No valid cell at all (terminal): return 0.
                    return 0
                cell_probs = torch.softmax(masked_logp, dim=-1)     # (n_cells,)

                if deterministic:
                    cell = int(torch.argmax(cell_probs).item())
                else:
                    dist = torch.distributions.Categorical(probs=cell_probs)
                    cell = int(dist.sample().item())
                return cell
            m = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
            action, _ = self.policy.sample(s, m, deterministic=deterministic)
            return int(action.item())

    def cell_to_bits(self, cell: int) -> np.ndarray:
        """Map a cell index (0..n_cells-1) to its canonical 7-bit action.

        Used by the training loop: `select_action` returns a cell index even in
        dimred mode (so the buffer/critic treat it like native), and this method
        converts it back to the 7-bit form the dimred env expects in `env.step`.
        """
        return self._cell_bits[int(cell)].cpu().numpy().astype(int)

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

    def _cell_distribution(
        self, state: torch.Tensor, mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Induced distribution over the FULL cell space from the 7-bit policy.

        The actor outputs 7 independent Bernoulli logits. Each valid cell c has a
        canonical 7-bit code b(c) (see `self._cell_bits`); the induced cell
        probability is
            pi(c|s) ∝ prod_i p_i(s)^{b_i(c)} (1 - p_i(s))^{1 - b_i(c)}  *  1[c valid]
        i.e. log pi(c|s) = sum_i [ b_i(c) log p_i + (1-b_i(c)) log(1-p_i) ].
        Invalid cells are masked to zero probability and the valid cells
        renormalize (softmax).

        Returns:
            log_pi : (B, n_cells)  log-probability over cells (renormalized)
            pi     : (B, n_cells)  probability over cells
        """
        bit_probs = torch.sigmoid(self.policy(state))               # (B, n_bits)
        # (B, n_cells) unnormalized log-prob per cell
        logp_cell = (
            self._cell_bits.unsqueeze(0) * torch.log(bit_probs.unsqueeze(1) + 1e-8)
            + (1 - self._cell_bits.unsqueeze(0))
            * torch.log(1 - bit_probs.unsqueeze(1) + 1e-8)
        ).sum(-1)

        # Mask invalid cells -> -1e9 (finite, avoids NaN when the whole mask is
        # False at terminal states), then renormalize via softmax.
        invalid = torch.full_like(logp_cell, -1e9)
        masked_logp = torch.where(mask.bool(), logp_cell, invalid)
        log_pi = torch.log_softmax(masked_logp, dim=-1)
        pi = torch.exp(log_pi)
        return log_pi, pi

    def _update_dimred(self, buffer, batch_size):
        """Discrete-SAC update for the reduced (7-bit) action space.

        Key fix vs. the old factored-Q dimred: the **critic stays full 96-dim**,
        so the value function loses no representational power. The 7-bit policy
        induces a distribution pi(c|s) over the 96 valid cells, and the standard
        discrete-SAC objectives are computed exactly as in the native branch —
        the only difference is how pi(c|s) is parameterized (masked-softmax of 96
        logits vs. product of 7 Bernoulli bits).
        """
        batch = buffer.sample(batch_size)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)        # (B,) cell indices (long)
        reward = batch["reward"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device)
        mask = batch["action_mask"].to(self.device)     # (B, n_cells)
        next_mask = batch["next_mask"].to(self.device)  # (B, n_cells)
        alpha_val = self.log_alpha.exp()

        # ---- V(s') over the full cell space (target nets + current policy) ----
        with torch.no_grad():
            next_log_pi, next_pi = self._cell_distribution(next_state, next_mask)
            q1_next = self.q1_target(next_state)        # (B, n_cells)
            q2_next = self.q2_target(next_state)
            q_next_min = torch.min(q1_next, q2_next)
            v_next = (next_pi * (q_next_min - alpha_val * next_log_pi)).sum(dim=-1)
            q_target = reward + self.gamma * (1.0 - done) * v_next

        # ---- Update Q1 / Q2 (full 96-dim critics, gather taken cell) ----
        q1_pred = self.q1(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        q1_loss = F.mse_loss(q1_pred, q_target)
        self.opt_q1.zero_grad(); q1_loss.backward(); self.opt_q1.step()
        q2_pred = self.q2(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        q2_loss = F.mse_loss(q2_pred, q_target)
        self.opt_q2.zero_grad(); q2_loss.backward(); self.opt_q2.step()

        # ---- Update Policy ----
        log_pi, pi = self._cell_distribution(state, mask)
        with torch.no_grad():
            q1_val = self.q1(state)
            q2_val = self.q2(state)
            q_min = torch.min(q1_val, q2_val)
        policy_loss = (pi * (alpha_val * log_pi - q_min)).sum(dim=-1).mean()

        self.opt_policy.zero_grad(); policy_loss.backward(); self.opt_policy.step()

        # ---- Alpha ----
        with torch.no_grad():
            entropy = -(pi * log_pi).sum(dim=-1).mean()
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
