# -*- coding: utf-8 -*-
"""TD3 (Twin Delayed DDPG) for 5G-TSN scheduling.

TD3 is a continuous-action, off-policy actor-critic algorithm adapted to the
discrete scheduling MDP via a *continuous relaxation* bridge. The key design
(2026-09-06 fix) is **strategy-space reduction + full-dimensional critic**:

- The critic ALWAYS consumes a 96-dim (L*S) cell distribution vector — the
  value function loses no representational power in either mode.
- native mode: actor outputs 96 logits; masked softmax -> 96-dim probability
  vector. Execution argmaxes / samples a discrete cell index.
- dimred mode: actor outputs 7 logits (strategy reduction); sigmoid -> 7
  Bernoulli probs which induce a 96-dim cell distribution via
  pi(c|s) ∝ prod_i p_i^{b_i(c)} (1-p_i)^{1-b_i(c)} (masked + renormalized).
  Execution samples a cell from that distribution and encodes it back to 7 bits.

Because the critic input is 96-dim in BOTH modes, the only difference between
native and dimred is how that 96-dim distribution is parameterized (96 free
logits vs. a rank-1-ish product of 7 bit probabilities). This isolates the
effect of the action-space reduction, which is the paper's contribution.

Core TD3 components
-------------------
- Twin critics Q1, Q2 with min() to suppress overestimation bias.
- Delayed policy updates (policy updated every `policy_delay` critic updates).
- Target policy smoothing (Gaussian noise clipped, added to the target action).
- Soft Polyak target updates.

Reuses `MLP` and `ReplayBuffer` from `tsn_sim.sac`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .sac import MLP, ReplayBuffer


# =====================================================================
#  Networks
# =====================================================================

class Actor(nn.Module):
    """Deterministic policy: state -> n_actions logits.

    The raw logits are converted to a 96-dim cell distribution by the agent
    (masked softmax in native mode, product-induced distribution in dimred).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.mlp = MLP(obs_dim, hidden_dims, output_dim=n_actions,
                       use_layer_norm=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw logits. Shape: (B, n_actions)."""
        return self.mlp(state)


class Critic(nn.Module):
    """Twin-able Q network: (state, 96-dim cell distribution) -> scalar Q.

    `action_dim` is the FULL cell-space dimension (n_cells = L*S = 96) in BOTH
    modes, so the value function is never reduced.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.mlp = MLP(obs_dim + action_dim, hidden_dims, output_dim=1,
                       use_layer_norm=True)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q(state, action). Shape: (B,)."""
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x).squeeze(-1)


# =====================================================================
#  TD3 Agent
# =====================================================================

class TD3Agent:
    """TD3 agent adapted to the discrete scheduling action space.

    Parameters
    ----------
    obs_dim : int
        Observation dimension.
    n_actions : int
        Action dimension (96 native / 7 dimred).
    hidden_dims : tuple
        Hidden layer sizes for all networks.
    lr : float
        Learning rate for all optimizers.
    gamma : float
        Discount factor.
    tau : float
        Soft target update rate (Polyak averaging).
    policy_delay : int
        Policy (and target) update frequency relative to critic updates.
    target_noise : float
        Std of Gaussian noise added to target action for smoothing.
    noise_clip : float
        Clip bound for the target smoothing noise.
    action_mode : str
        "native" (96 logits -> masked softmax) or "dimred" (7 logits -> sigmoid).
    device : str
        "cpu" or "cuda".
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
        gamma: float = 0.9,
        tau: float = 0.005,
        policy_delay: int = 2,
        target_noise: float = 0.1,
        noise_clip: float = 0.2,
        action_mode: str = "native",
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self._action_mode = action_mode
        self.device = torch.device(device)

        # n_cells: critic input dim (the FULL cell space = L*S = 96). The critic
        # always consumes a 96-dim cell distribution so the value function keeps
        # full representational power in BOTH modes.
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
            self.n_slot_bits = n_actions - self.n_link_bits
        else:
            self.n_slot_bits = int(np.ceil(np.log2(self.n_slots)))

        # Precompute canonical cell -> bits encoding matrix (n_cells x n_bits),
        # matching env._cell_to_bits. Used only in dimred mode to induce the
        # 96-dim cell distribution from the 7-bit policy.
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
        # Actor outputs n_actions (96 native / 7 dimred); critics consume n_cells.
        self.actor = Actor(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q1 = Critic(obs_dim, self.n_cells, hidden_dims).to(self.device)
        self.q2 = Critic(obs_dim, self.n_cells, hidden_dims).to(self.device)

        self.actor_target = Actor(obs_dim, n_actions, hidden_dims).to(self.device)
        self.q1_target = Critic(obs_dim, self.n_cells, hidden_dims).to(self.device)
        self.q2_target = Critic(obs_dim, self.n_cells, hidden_dims).to(self.device)

        # Hard-copy targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in list(self.actor_target.parameters()) + \
                 list(self.q1_target.parameters()) + \
                 list(self.q2_target.parameters()):
            p.requires_grad = False

        # --- Optimizers ---
        self.opt_actor = Adam(self.actor.parameters(), lr=lr)
        self.opt_q1 = Adam(self.q1.parameters(), lr=lr)
        self.opt_q2 = Adam(self.q2.parameters(), lr=lr)

        self._step = 0

    # --- continuous action bridge ---------------------------------------

    def _cont_action(self, logits: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to the 96-dim cell distribution fed to the critic.

        native: masked softmax over 96 logits -> (B, n_cells).
        dimred: sigmoid over 7 logits -> (B, n_bits); product-induced distribution
                over the 96 cells, masked + renormalized -> (B, n_cells).
        """
        if self._action_mode == "dimred":
            bit_probs = torch.sigmoid(logits)                       # (B, n_bits)
            logp_cell = (
                self._cell_bits.unsqueeze(0) * torch.log(bit_probs.unsqueeze(1) + 1e-8)
                + (1 - self._cell_bits.unsqueeze(0))
                * torch.log(1 - bit_probs.unsqueeze(1) + 1e-8)
            ).sum(-1)                                               # (B, n_cells)
            invalid = torch.full_like(logp_cell, -1e9)
            masked_logp = torch.where(mask.bool(), logp_cell, invalid)
            return torch.softmax(masked_logp, dim=-1)               # (B, n_cells)
        m = mask.bool()
        masked = torch.where(m, logits, torch.full_like(logits, -1e9))
        return torch.softmax(masked, dim=-1)

    # --- action selection ------------------------------------------------

    def select_action(self, state: np.ndarray, mask: np.ndarray,
                      deterministic: bool = False):
        """Select an action.

        Returns a tuple (a_exec, a_cont):
        - a_exec: the action fed to env.step (int for native, (n_bits,) 0/1 array
          for dimred).
        - a_cont: the 96-dim cell distribution vector stored in the replay
          buffer (float32 array of shape (n_cells,)) — identical dim in BOTH
          modes so the critic never sees a reduced input.
        """
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            logits = self.actor(s)  # (1, n_actions)

            m = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
            a_cont = self._cont_action(logits, m).squeeze(0)  # (n_cells,)

            if self._action_mode == "dimred":
                cell_probs = a_cont  # already a valid-cell distribution
                if deterministic:
                    cell = int(torch.argmax(cell_probs).item())
                else:
                    dist = torch.distributions.Categorical(probs=cell_probs)
                    cell = int(dist.sample().item())
                bits = self.cell_to_bits(cell)
                return bits, a_cont.cpu().numpy().astype(np.float32)

            # native
            if deterministic:
                a_exec = int(torch.argmax(a_cont).item())
            else:
                dist = torch.distributions.Categorical(probs=a_cont)
                a_exec = int(dist.sample().item())
            return a_exec, a_cont.cpu().numpy().astype(np.float32)

    def cell_to_bits(self, cell: int) -> np.ndarray:
        """Map a cell index (0..n_cells-1) to its canonical 7-bit action."""
        return self._cell_bits[int(cell)].cpu().numpy().astype(int)

    # --- training step ---------------------------------------------------

    def update(self, buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        """One TD3 gradient update from the replay buffer.

        Returns a dict of loss/metric values for logging.
        """
        if len(buffer) < batch_size:
            return {}

        batch = buffer.sample(batch_size)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)          # (B, n_actions) continuous
        reward = batch["reward"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device)
        next_mask = batch["next_mask"].to(self.device)
        mask = batch["action_mask"].to(self.device)

        # ---- Critic targets (twin + target policy smoothing) ----
        with torch.no_grad():
            next_logits = self.actor_target(next_state)
            next_a_cont = self._cont_action(next_logits, next_mask)
            noise = (torch.randn_like(next_a_cont) * self.target_noise) \
                .clamp(-self.noise_clip, self.noise_clip)
            a_smooth = (next_a_cont + noise).clamp(0.0, 1.0)
            q1_t = self.q1_target(next_state, a_smooth)
            q2_t = self.q2_target(next_state, a_smooth)
            q_t = torch.min(q1_t, q2_t)
            q_target = reward + self.gamma * (1.0 - done) * q_t

        # ---- Update Q1 ----
        q1_pred = self.q1(state, action)
        q1_loss = F.mse_loss(q1_pred, q_target)
        self.opt_q1.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
        self.opt_q1.step()

        # ---- Update Q2 ----
        q2_pred = self.q2(state, action)
        q2_loss = F.mse_loss(q2_pred, q_target)
        self.opt_q2.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
        self.opt_q2.step()

        # ---- Delayed policy update ----
        policy_loss = 0.0
        if self._step % self.policy_delay == 0:
            logits = self.actor(state)
            a_cont = self._cont_action(logits, mask)
            policy_loss_t = -self.q1(state, a_cont).mean()

            self.opt_actor.zero_grad()
            policy_loss_t.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.opt_actor.step()

            # Soft-update all target networks
            with torch.no_grad():
                for p, p_t in zip(self.actor.parameters(),
                                  self.actor_target.parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
                for p, p_t in zip(self.q1.parameters(),
                                  self.q1_target.parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
                for p, p_t in zip(self.q2.parameters(),
                                  self.q2_target.parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

            policy_loss = float(policy_loss_t.item())

        self._step += 1

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": policy_loss,
            "q1_value": float(q1_pred.mean().item()),
            "q2_value": float(q2_pred.mean().item()),
        }

    # --- save / load -----------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "step": self._step,
        }, path)

    def load(self, path: str) -> None:
        """Load model weights from a file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self._step = ckpt.get("step", 0)
