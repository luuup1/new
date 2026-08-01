"""5G-TSN Multi-Link Scheduling Gym Environment.

Phase 1: Gym environment encapsulation for DRL-based scheduling.

MDP formulation
---------------
- Episode: schedule all packets in one hyperperiod (~124 packets)
- State:   resource grid (L×S) + current packet features + channel quality
- Action:  Discrete(L×S) — choose (link, slot) for the current packet
- Reward:  -effective_peak_load (terminal), small shaping reward per step

Design decisions
----------------
- Packet ordering: fixed EDF (earliest deadline first), proven best baseline.
  The agent learns *where* to place, not *which packet to pick*.
- Action masking: pre-computes validity for each (link, slot) based on
  available links, time window, remaining RB, and deadline constraints.
- Sparse reward: only terminal objective matters. Step-level local shaping
  is optional (controlled by `use_step_shaping`).
"""

from __future__ import annotations

import math
from dataclasses import replace as _replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Env
from gymnasium import spaces as _spaces
from gymnasium import utils as _gym_utils

from .candidate import instantiate_packets
from .config import SimulationConfig
from .metrics import compute_metrics
from .models import (
    CellKey,
    Flow,
    Packet,
    PacketKey,
    Scenario,
    ScheduleEntry,
)
from .scenario import build_scenario


class TSNSchedulingEnv(Env):
    """Gymnasium environment for 5G-TSN packet scheduling.

    Observation
    -----------
    Flat vector of shape (obs_dim,):
        [0:L*S)     grid_load:    used_rb / capacity per cell, normalized [0, 1]
        [L*S:2*L*S) channel_eff:  bits_per_rb(best_link) / bits_per_rb(this_cell)
                                  for the *current* packet only
        [-N:]       packet_features:
            [0]  link_1_available   {0, 1}
            [1]  link_2_available   {0, 1}
            [2]  link_3_available   {0, 1}
            [3]  norm_earliest      earliest_slot / S
            [4]  norm_latest        latest_slot / S
            [5]  window_urgency     1 / (window_width + 1)  scaled
            [6]  progress           packets_done / total_packets

    Action
    ------
    Discrete(L × S): action = (link-1) * S + slot
    Link = action // S + 1, Slot = action % S

    Masking
    -------
    action_masks() returns boolean mask over [0, L×S).
    Only cells that satisfy all of the following are True:
        - link ∈ flow.available_links
        - slot ∈ [earliest_slot, latest_slot]
        - required_rb ≤ remaining_rb
        - delay ≤ deadline
    """

    metadata = {"render_modes": ["human"]}

    # --- construction -------------------------------------------------

    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        use_step_shaping: bool = False,
        shaping_weight: float = 0.01,
        infeasible_penalty: float = 0.5,
        reward_mode: str = "terminal",
        obs_mode: str = "full",
        reward_delta: float = 0.1,
        reward_zeta: float = 3.0,
        multi_scenario: bool = False,
        order_mode: str = "edf",
        action_mode: str = "native",
    ):
        """Initialize the environment.

        Parameters
        ----------
        reward_mode : str
            - "terminal": only terminal -effective_peak_load (original behavior)
            - "shaping": terminal + weak step shaping (original)
            - "exponential": delta*exp(-zeta*peak) per step + terminal
        obs_mode : str
            - "full": 199-dim (grid_load + channel_eff + pkt_feat)
            - "compact": 111-dim (grid_load + channel_stats + global_stats + pkt_feat)
        reward_delta : float
            Scale of exponential per-step reward.
        reward_zeta : float
            Sensitivity of exponential reward to peak load.
        multi_scenario : bool
            If True, each reset() rebuilds scenario with seed+episode_count,
            giving diverse traffic patterns for training.
        """
        super().__init__()

        self._config = config or SimulationConfig()
        self._use_step_shaping = use_step_shaping
        self._shaping_weight = shaping_weight
        self._infeasible_penalty = infeasible_penalty
        self._reward_mode = reward_mode
        self._obs_mode = obs_mode
        self._reward_delta = reward_delta
        self._reward_zeta = reward_zeta
        self._multi_scenario = multi_scenario
        self._order_mode = order_mode
        self._episode_count: int = 0

        # --- Fixed dimensions (from config, not scenario) ---
        self.L: int = self._config.link_count              # link count (3)
        self.S: int = self._get_hyperperiod()              # slot count (32)
        self._n_cells: int = self.L * self.S              # 96

        # --- Action-mode dimensions ---
        # native: Discrete(L*S) picks one of 96 (link, slot) cells.
        # dimred: factored binary action — 2 link bits + ceil(log2 S) slot bits
        #         (32 slots -> 5 bits) = 7 (0/1) outputs, decoded + projected
        #         to a valid cell by the env. This is the action-space
        #         dimensionality reduction (the paper's contribution).
        self._action_mode: str = action_mode
        self._n_link_bits: int = 2                      # 3 links -> 2 bits
        self._n_slot_bits: int = int(math.ceil(math.log2(self.S)))  # 32 -> 5
        self._n_bits: int = self._n_link_bits + self._n_slot_bits      # 7

        # Build initial episode data (scenario, packets, flows, order)
        self._build_episode_data(self._config.seed)

        # Observation space
        self._grid_dim: int = self._n_cells       # L×S  load channel
        self._ch_dim: int = self._n_cells         # L×S  channel-efficiency (full mode)
        self._channel_stats_dim: int = 5          # compact: per_link_best(3) + best_link + var
        self._global_stats_dim: int = 3           # compact: peak + mean + imbalance
        self._pkt_feat_dim: int = 7               # packet-level scalars

        if self._obs_mode == "compact":
            self._obs_dim: int = (
                self._grid_dim + self._channel_stats_dim
                + self._global_stats_dim + self._pkt_feat_dim
            )
        else:
            self._obs_dim: int = self._grid_dim + self._ch_dim + self._pkt_feat_dim

        self.observation_space = _spaces.Box(
            low=-1.0, high=2.0,
            shape=(self._obs_dim,), dtype=np.float32,
        )

        # Action space: native = Discrete(96); dimred = 7 (0/1) bits
        if self._action_mode == "dimred":
            self.action_space = _spaces.MultiBinary(self._n_bits)
        else:
            self.action_space = _spaces.Discrete(self._n_cells)

        # Runtime state (populated in reset)
        self._used_rb: Dict[CellKey, int] = {}
        self._remaining_rb: Dict[CellKey, int] = {}
        self._schedule: List[ScheduleEntry] = []
        self._infeasible_keys: List[PacketKey] = []
        self._current_idx: int = 0
        self._done: bool = False

    # --- gym interface -------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an empty grid, start a new episode."""
        super().reset(seed=seed)

        # --- Multi-scenario: rebuild scenario each episode ---
        if self._multi_scenario:
            self._episode_count += 1
            ep_seed = self._config.seed + self._episode_count
            self._build_episode_data(ep_seed)

        # Reset resource grid
        self._remaining_rb = dict(self._scenario.rb_capacity)
        self._used_rb = {cell: 0 for cell in self._scenario.rb_capacity}
        self._schedule = []
        self._infeasible_keys = []
        self._current_idx = 0
        self._done = False

        return self._get_obs(), self._get_info()

    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one scheduling decision.

        Parameters
        ----------
        action : int  (native mode)
            Index into the flattened (link, slot) grid:
            link = action // S + 1, slot = action % S
        action : np.ndarray of shape (n_bits,)  (dimred mode)
            7 (0/1) bits: [link_bit0, link_bit1, slot_bit0..slot_bit4].
            Decoded to (link, slot) and deterministically projected to the
            nearest *valid* cell (min Hamming distance) if infeasible.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self._done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        packet_key = self._packet_order[self._current_idx]
        packet = self._packets[packet_key]
        flow = self._flows[packet.flow_id]

        if self._action_mode == "dimred":
            action = np.asarray(action, dtype=int).reshape(-1)
            projected = self._project_to_valid(action)
            if projected is None:
                link = slot = -1            # no valid cell at all -> infeasible
            else:
                link, slot = projected
        else:
            link = int(action) // self.S + 1
            slot = int(action) % self.S

        valid, required_rb, bits_per_rb, delay_ms = self._validate_action(
            link, slot, packet, flow,
        )

        reward = 0.0

        if valid:
            cell = (link, slot)
            self._remaining_rb[cell] -= required_rb
            self._used_rb[cell] += required_rb
            self._schedule.append(
                ScheduleEntry(
                    flow_id=packet.flow_id,
                    packet_index=packet.packet_index,
                    arrival_ms=packet.arrival_ms,
                    deadline_ms=packet.deadline_ms,
                    link=link,
                    slot=slot,
                    required_rb=required_rb,
                    bits_per_rb=bits_per_rb,
                    delay_ms=delay_ms,
                )
            )
        else:
            # Action was invalid → packet is infeasible → record as dropped.
            # (The environment already applies action masking, so a truly
            #  infeasible raw action here means no valid cell was available.)
            self._infeasible_keys.append(packet_key)

        # --- Single unified dense reward: negative effective peak load ---
        #   r_t = -hat_P_t = -(max_{l,s} L_{l,s} + N_inf,t / N_total)
        # The per-step signal and the terminal objective live in this one
        # expression; the final step's reward already equals the terminal
        # reward, so no separate terminal addition is needed.
        reward = -self._effective_peak_load()

        self._current_idx += 1

        if self._current_idx >= self._total_packets:
            self._done = True

        return self._get_obs(), float(reward), self._done, False, self._get_info()

    # --- action masking -------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return boolean mask: True = valid action."""
        mask = np.zeros(self._n_cells, dtype=bool)

        if self._done or self._current_idx >= self._total_packets:
            return mask

        packet_key = self._packet_order[self._current_idx]
        packet = self._packets[packet_key]
        flow = self._flows[packet.flow_id]

        for link in flow.available_links:
            base_idx = (link - 1) * self.S
            for slot in range(self.S):
                # Time window constraint
                if slot < packet.earliest_slot or slot > packet.latest_slot:
                    continue

                # Compute required RBs
                bits_per_rb = self._scenario.rb_bits[(flow.flow_id, link, slot)]
                required_rb = math.ceil(flow.packet_size_bits / bits_per_rb)

                # Resource capacity check
                if required_rb > self._remaining_rb[(link, slot)]:
                    continue

                # Deadline check
                finish_time = (slot + 1) * self._scenario.slot_ms
                delay = finish_time - packet.arrival_ms
                if delay > flow.deadline_ms:
                    continue

                mask[base_idx + slot] = True

        return mask

    # --- dimred action helpers ------------------------------------------

    def _decode_bits(self, bits: np.ndarray):
        """Decode 7 (0/1) bits -> (link, slot). link=-1 if link bits invalid."""
        bits = np.asarray(bits, dtype=int).reshape(-1)
        link_code = int(bits[0]) * 2 + int(bits[1])
        link = link_code + 1 if link_code < self.L else -1
        slot = 0
        for i in range(self._n_slot_bits):
            slot = (slot << 1) | int(bits[2 + i])
        return link, slot

    def _cell_to_bits(self, link: int, slot: int) -> np.ndarray:
        """Encode a valid (link, slot) cell back to its canonical 7-bit form."""
        link_code = link - 1  # 0..L-1
        bits = np.zeros(self._n_bits, dtype=int)
        bits[0] = (link_code >> 1) & 1
        bits[1] = link_code & 1
        for i in range(self._n_slot_bits):
            bits[2 + i] = (slot >> (self._n_slot_bits - 1 - i)) & 1  # MSB first
        return bits

    def _project_to_valid(self, bits: np.ndarray):
        """Decode bits; if the resulting cell is invalid, return the valid cell
        whose canonical bit-pattern has the smallest Hamming distance (ties:
        lowest cell index). Pure constraint projection — no scheduling
        preference. Returns None only if no valid cell exists at all."""
        link, slot = self._decode_bits(bits)
        mask = self.action_masks()
        if link > 0 and 0 <= slot < self.S and mask[(link - 1) * self.S + slot]:
            return link, slot
        best, best_h = None, 1_000_000
        for c in np.where(mask)[0]:
            l, s = c // self.S + 1, c % self.S
            cb = self._cell_to_bits(l, s)
            h = int(np.sum(cb != np.asarray(bits, dtype=int)))
            if h < best_h or (h == best_h and best is None):
                best, best_h = (l, s), h
        return best

    def expert_min_load_action(self) -> int:
        """Return the feasible action that minimizes post-placement cell load
        for the *current* packet. This is the per-step placement rule used by
        the `edf_min_load` heuristic (placement part only). It is used as the
        BC / DAGGER **teacher on the random-order env** so the demonstration
        distribution matches the DRL MDP (random order + learned placement).

        Ties (equal post-load) are broken by the lowest action index.
        Returns 0 if no feasible action exists (should not happen before done).
        """
        if self._done or self._current_idx >= self._total_packets:
            return 0
        packet_key = self._packet_order[self._current_idx]
        packet = self._packets[packet_key]
        flow = self._flows[packet.flow_id]
        mask = self.action_masks()
        feasible = np.where(mask)[0]
        if len(feasible) == 0:
            return 0
        best_action = int(feasible[0])
        best_post = float("inf")
        for a in feasible:
            link = a // self.S + 1
            slot = a % self.S
            cell = (link, slot)
            bits_per_rb = self._scenario.rb_bits.get((flow.flow_id, link, slot), 0)
            if bits_per_rb <= 0:
                continue
            required_rb = math.ceil(flow.packet_size_bits / bits_per_rb)
            post = (self._used_rb[cell] + required_rb) / self._scenario.rb_capacity[cell]
            if post < best_post:
                best_post = post
                best_action = int(a)
        return best_action

    # --- rendering ------------------------------------------------------

    def render(self):
        """Print current grid state to console."""
        link_names = [f"Link{l}" for l in range(1, self.L + 1)]
        header = "Slot    " + " ".join(f"{n:>8}" for n in link_names) + "   | Action"
        print(header)
        print("-" * len(header))

        for slot in range(self.S):
            loads = []
            for link in range(1, self.L + 1):
                cell = (link, slot)
                cap = self._scenario.rb_capacity[cell]
                lvl = self._used_rb[cell] / cap
                bar = "█" * min(8, int(lvl * 8))
                loads.append(f"{bar:<8}")
            print(f"Slot{slot:>2}   {' '.join(loads)}")

        print(f"\nProgress: {self._current_idx}/{self._total_packets}  "
              f"Dropped: {len(self._infeasible_keys)}")

    # --- internal helpers -----------------------------------------------

    def _get_hyperperiod(self) -> int:
        """Compute hyperperiod from config (deterministic, no RNG needed)."""
        from functools import reduce
        from math import gcd
        return reduce(lambda a, b: a * b // gcd(a, b), self._config.periods_ms, 1)

    def _build_episode_data(self, seed: int) -> None:
        """Build scenario, packets, flows, and packet order for one episode.

        Called in __init__ and in reset() when multi_scenario is enabled.
        """
        from dataclasses import replace as _replace
        ep_config = _replace(self._config, seed=seed)
        self._scenario: Scenario = build_scenario(ep_config)
        self._packets: Dict[PacketKey, Packet] = instantiate_packets(self._scenario)
        self._flows: Dict[int, Flow] = {f.flow_id: f for f in self._scenario.flows}

        # Packet ordering:
        #   - "edf"   : earliest-deadline-first (historical baseline复现, NOT the
        #              naive baseline — DRL on edf order gets the ordering gain
        #              for free, which is NOT learned by RL).
        #   - "random": replicate random_feasible's packet order so DRL is
        #              evaluated "on the baseline" (random order + learned
        #              placement) and its gain is attributable to RL, not to a
        #              hand-picked ordering. Seed matches the scenario seed so
        #              the order is identical to what random_feasible used.
        if self._order_mode == "random":
            import random as _random
            keys = list(self._packets.keys())
            _random.Random(ep_config.seed).shuffle(keys)
            self._packet_order = tuple(keys)
        else:
            self._packet_order = tuple(
                sorted(
                    self._packets,
                    key=lambda k: (
                        self._packets[k].arrival_ms + self._packets[k].deadline_ms,
                        self._packets[k].period_ms,
                        k,
                    ),
                )
            )
        self._total_packets: int = len(self._packet_order)

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        # --- grid load channel ---
        grid = np.zeros((self.L, self.S), dtype=np.float32)
        for link in range(1, self.L + 1):
            for slot in range(self.S):
                cap = self._scenario.rb_capacity[(link, slot)]
                grid[link - 1, slot] = self._used_rb[(link, slot)] / cap if cap else 0.0

        # --- packet features ---
        if self._current_idx < self._total_packets:
            packet_key = self._packet_order[self._current_idx]
            packet = self._packets[packet_key]
            flow = self._flows[packet.flow_id]

            pkt_feats = np.array([
                float(1 in flow.available_links),                     # link1 avail
                float(2 in flow.available_links),                     # link2 avail
                float(3 in flow.available_links),                     # link3 avail
                packet.earliest_slot / self.S,                        # norm earliest
                packet.latest_slot / self.S,                          # norm latest
                1.0 / max(1, packet.latest_slot - packet.earliest_slot + 1),
                                                                      # urgency
                self._current_idx / max(1, self._total_packets),      # progress
            ], dtype=np.float32)
        else:
            pkt_feats = np.zeros(self._pkt_feat_dim, dtype=np.float32)

        if self._obs_mode == "compact":
            # --- channel stats (5-dim, replaces 96-dim channel_eff) ---
            channel_stats = np.zeros(self._channel_stats_dim, dtype=np.float32)
            if self._current_idx < self._total_packets:
                packet_key = self._packet_order[self._current_idx]
                flow = self._flows[packet_key[0]]

                # Per-link best channel efficiency (relative to global best)
                best_bprb = max(
                    self._scenario.rb_bits.get((flow.flow_id, l, s), 1)
                    for l in range(1, self.L + 1)
                    for s in range(self.S)
                )
                best_bprb = max(best_bprb, 1)

                per_link_best = np.zeros(self.L, dtype=np.float32)
                for link in range(1, self.L + 1):
                    link_max = max(
                        self._scenario.rb_bits.get((flow.flow_id, link, s), 1)
                        for s in range(self.S)
                    )
                    per_link_best[link - 1] = link_max / best_bprb

                best_link = int(np.argmax(per_link_best))
                channel_stats[:self.L] = per_link_best
                channel_stats[self.L] = best_link / max(1, self.L - 1)  # best link idx
                channel_stats[self.L + 1] = float(np.var(per_link_best))  # channel variance

            # --- global stats (3-dim) ---
            grid_flat = grid.flatten()
            current_peak = float(np.max(grid_flat))
            mean_load = float(np.mean(grid_flat))
            global_stats = np.array([
                current_peak,
                mean_load,
                current_peak - mean_load,  # imbalance
            ], dtype=np.float32)

            return np.concatenate([grid_flat, channel_stats, global_stats, pkt_feats])

        else:
            # --- full mode: channel efficiency channel (current packet only) ---
            channel = np.zeros((self.L, self.S), dtype=np.float32)
            if self._current_idx < self._total_packets:
                packet_key = self._packet_order[self._current_idx]
                flow = self._flows[packet_key[0]]
                # Best bits_per_rb for this flow across all (link, slot)
                best_bprb = max(
                    self._scenario.rb_bits.get((flow.flow_id, l, s), 1)
                    for l in range(1, self.L + 1)
                    for s in range(self.S)
                )
                best_bprb = max(best_bprb, 1)
                for link in range(1, self.L + 1):
                    for slot in range(self.S):
                        bprb = self._scenario.rb_bits.get((flow.flow_id, link, slot), 1)
                        channel[link - 1, slot] = bprb / best_bprb

            return np.concatenate([grid.flatten(), channel.flatten(), pkt_feats])

    def _get_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "step": self._current_idx,
            "total_packets": self._total_packets,
            "dropped": len(self._infeasible_keys),
        }
        if self._done:
            metrics = compute_metrics(
                self._scenario,
                tuple(self._schedule),
                self._packets,
                tuple(self._infeasible_keys),
            )
            info.update(metrics)
            info["schedule"] = tuple(self._schedule)
        return info

    def _validate_action(
        self, link: int, slot: int, packet: Packet, flow: Flow,
    ) -> Tuple[bool, int, int, float]:
        """Check whether (link, slot) is feasible for this packet.

        Returns (valid, required_rb, bits_per_rb, delay_ms).
        """
        # Link availability
        if link not in flow.available_links:
            return (False, 0, 0, 0.0)

        # Time window
        if slot < packet.earliest_slot or slot > packet.latest_slot:
            return (False, 0, 0, 0.0)

        # Compute resources
        bits_per_rb = self._scenario.rb_bits.get((flow.flow_id, link, slot), 0)
        if bits_per_rb <= 0:
            return (False, 0, 0, 0.0)

        required_rb = math.ceil(flow.packet_size_bits / bits_per_rb)
        if required_rb > self._remaining_rb.get((link, slot), 0):
            return (False, 0, 0, 0.0)

        # Deadline
        finish_time = (slot + 1) * self._scenario.slot_ms
        delay_ms = finish_time - packet.arrival_ms
        if delay_ms > flow.deadline_ms:
            return (False, 0, 0, 0.0)

        return (True, required_rb, bits_per_rb, delay_ms)

    def _effective_peak_load(self) -> float:
        """Effective peak load = max cell load + current drop ratio.

            hat_P = max_{l,s} (used_rb / capacity) + N_inf / N_total

        This is the single unified reward quantity: r_t = -hat_P.
        """
        peak = 0.0
        for cell, cap in self._scenario.rb_capacity.items():
            if cap > 0:
                load = self._used_rb[cell] / cap
                if load > peak:
                    peak = load
        drop_ratio = len(self._infeasible_keys) / max(1, self._total_packets)
        return peak + drop_ratio

    def _compute_terminal_reward(self) -> float:
        """Terminal reward: negative effective peak load (== last step reward)."""
        return -self._effective_peak_load()

    def _compute_current_peak(self) -> float:
        """Compute current peak load across all cells (max used_rb/capacity)."""
        peak = 0.0
        for cell, cap in self._scenario.rb_capacity.items():
            if cap > 0:
                load = self._used_rb[cell] / cap
                if load > peak:
                    peak = load
        return peak

    # --- properties for external use ------------------------------------

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def num_links(self) -> int:
        return self.L

    @property
    def num_slots(self) -> int:
        return self.S

    @property
    def num_packets(self) -> int:
        return self._total_packets

    @property
    def n_actions(self) -> int:
        """Action dimension for the agent: 96 (native) or 7 (dimred)."""
        return self._n_bits if self._action_mode == "dimred" else self._n_cells

    @property
    def action_mode(self) -> str:
        return self._action_mode

    @property
    def schedule(self) -> Tuple[ScheduleEntry, ...]:
        return tuple(self._schedule)

    @property
    def current_packet(self) -> PacketKey | None:
        if self._done or self._current_idx >= self._total_packets:
            return None
        return self._packet_order[self._current_idx]
