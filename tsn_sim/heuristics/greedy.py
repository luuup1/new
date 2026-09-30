"""Greedy scheduling heuristic.

A deterministic, myopic placement rule:

* **Order** — earliest-deadline-first (EDF), i.e. packets with the tightest
  remaining slack are placed first; ties broken by period and key.
* **Placement** — pick the feasible cell with the *lowest load after
  placement*, so load is spread as evenly as possible across cells.

This is the classic "balance the load greedily" baseline and the natural
non-RL upper reference for the DRL agents (DRL should at least beat a simple
min-load greedy under random order).
"""

from __future__ import annotations

from typing import List

from ..models import Candidate, PacketKey, Scenario, SimulationResult
from .base import SequentialScheduler, State, slot_load_after


class GreedyScheduler(SequentialScheduler):
    """EDF ordering + min-load placement."""

    def order_packets(self) -> List[PacketKey]:
        return sorted(
            self.packets,
            key=lambda key: (
                self.packets[key].arrival_ms + self.packets[key].deadline_ms,
                self.packets[key].period_ms,
                key,
            ),
        )

    def select_candidate(
        self, packet_key: PacketKey, candidates: List[Candidate], state: State
    ) -> Candidate | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                slot_load_after(candidate, self.scenario, state),
                candidate.delay_ms,
                -candidate.bits_per_rb,
                candidate.link,
                candidate.slot,
            ),
        )


def schedule(scenario: Scenario, seed: int = 0, **kwargs) -> SimulationResult:
    """Run the greedy (EDF + min-load) heuristic on ``scenario``."""
    return GreedyScheduler(scenario, seed=seed).run()
