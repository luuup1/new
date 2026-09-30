"""Proportional-fair (PF) scheduling heuristic.

Proportional fairness aims to balance throughput across flows: no flow should
be starved by a few heavy flows.  The canonical objective is to maximize
``sum(log(R_f))`` where ``R_f`` is the cumulative allocation of flow ``f``.

Because the full problem is NP-hard, we use the standard greedy marginal
approximation: when placing a packet of flow ``f`` that consumes ``d`` RBs, the
marginal log-utility gain is ``log(R_f + d) - log(R_f) ≈ d / R_f``.  We always
place the packet on the cell that maximises this gain, i.e. favour flows with
the *smallest* accumulated allocation so far — while keeping deadline order
(EDF) so fairness does not come at the cost of missed deadlines.

Compared to :mod:`greedy` (which only balances cell load), PF additionally
balances *flow* allocation, trading a little raw peak-load for fairness.
"""

from __future__ import annotations

from typing import Dict, List

from ..models import Candidate, PacketKey, Scenario, SimulationResult
from .base import SequentialScheduler, State, slot_load_after

#: Small prior so the first packet of a flow never divides by zero and the
#: marginal gain stays bounded.
_EPS = 1.0


class ProportionalFairScheduler(SequentialScheduler):
    """EDF ordering + proportional-fair placement."""

    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        super().__init__(scenario, seed=seed)
        self.allocated: Dict[int, float] = {
            flow.flow_id: _EPS for flow in scenario.flows
        }

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

        flow_id = packet_key[0]
        current = self.allocated[flow_id]

        def score(candidate: Candidate) -> tuple:
            # Larger = more fair: flows with little allocation are preferred.
            fairness = candidate.required_rb / current
            # Spread load as a secondary objective.
            load = slot_load_after(candidate, self.scenario, state)
            return (fairness, -load, candidate.delay_ms)

        best = max(candidates, key=score)
        # Commit the allocation immediately: the packet *will* be placed.
        self.allocated[flow_id] += best.required_rb
        return best


def schedule(scenario: Scenario, seed: int = 0, **kwargs) -> SimulationResult:
    """Run the proportional-fair heuristic on ``scenario``."""
    return ProportionalFairScheduler(scenario, seed=seed).run()
