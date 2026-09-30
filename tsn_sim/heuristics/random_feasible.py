"""Random-feasible heuristic (the only legitimate baseline).

Packets are placed in a uniformly shuffled order; each packet is assigned to a
uniformly random cell among those that currently have enough free RBs.

This is the "do nothing smart" reference: random order + random placement.  It
marks the absolute performance floor that any informed scheduler should beat.
"""

from __future__ import annotations

from typing import List

from ..models import Candidate, PacketKey, Scenario, SimulationResult
from .base import SequentialScheduler, State


class RandomFeasibleScheduler(SequentialScheduler):
    """Random order + random feasible placement."""

    def order_packets(self) -> List[PacketKey]:
        keys = list(self.packets)
        self.rng.shuffle(keys)
        return keys

    def select_candidate(
        self, packet_key: PacketKey, candidates: List[Candidate], state: State
    ) -> Candidate | None:
        if not candidates:
            return None
        return self.rng.choice(candidates)


def schedule(scenario: Scenario, seed: int = 0, **kwargs) -> SimulationResult:
    """Run the random-feasible heuristic on ``scenario``."""
    return RandomFeasibleScheduler(scenario, seed=seed).run()
