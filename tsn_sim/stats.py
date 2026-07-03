"""Problem-size diagnostics."""

from __future__ import annotations

from typing import Mapping

from .candidate import build_candidates, instantiate_packets
from .models import Scenario


def problem_stats(scenario: Scenario) -> Mapping[str, float]:
    packets = instantiate_packets(scenario)
    candidates_by_packet, infeasible = build_candidates(scenario, packets)
    candidate_count = sum(len(values) for values in candidates_by_packet.values())
    cell_count = len(scenario.links) * scenario.hyperperiod_ms

    return {
        "flow_count": float(len(scenario.flows)),
        "packet_count": float(len(packets)),
        "candidate_count": float(candidate_count),
        "infeasible_candidate_packet_count": float(len(infeasible)),
        "link_count": float(len(scenario.links)),
        "hyperperiod_ms": float(scenario.hyperperiod_ms),
        "cell_count": float(cell_count),
        "milp_binary_variables": float(candidate_count),
        "milp_continuous_variables": 1.0,
        "milp_constraints_approx": float(len(packets) + 2 * cell_count),
        "avg_candidates_per_packet": (
            float(candidate_count / len(packets)) if packets else 0.0
        ),
    }
