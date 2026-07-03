"""Small-scale MILP optimum solver for heuristic quality checks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack

from .candidate import build_candidates, instantiate_packets
from .metrics import compute_metrics
from .models import CellKey, PacketKey, Scenario, ScheduleEntry, SimulationResult


def solve_optimal_milp(scenario: Scenario, time_limit_s: float = 30.0) -> SimulationResult:
    packets = instantiate_packets(scenario)
    candidates_by_packet, infeasible_packets = build_candidates(scenario, packets)
    if infeasible_packets:
        return SimulationResult(
            status="infeasible",
            objective=float("inf"),
            schedule=(),
            metrics={"infeasible_packet_count": float(len(infeasible_packets))},
            infeasible_packets=infeasible_packets,
            message="At least one packet has no feasible link-slot candidate.",
        )

    candidates = [candidate for group in candidates_by_packet.values() for candidate in group]
    packet_keys = list(packets)
    cells = [(link, slot) for link in scenario.links for slot in range(scenario.hyperperiod_ms)]
    y_count = len(candidates)
    rho_index = y_count
    variable_count = y_count + 1

    packet_to_indices: dict[PacketKey, list[int]] = {key: [] for key in packet_keys}
    cell_to_indices: dict[CellKey, list[int]] = {cell: [] for cell in cells}
    for index, candidate in enumerate(candidates):
        packet_to_indices[candidate.packet].append(index)
        cell_to_indices[(candidate.link, candidate.slot)].append(index)

    rows = []
    lower_bounds = []
    upper_bounds = []

    for packet_key in packet_keys:
        row = lil_matrix((1, variable_count), dtype=float)
        for index in packet_to_indices[packet_key]:
            row[0, index] = 1.0
        rows.append(row.tocsr())
        lower_bounds.append(1.0)
        upper_bounds.append(1.0)

    for cell in cells:
        capacity = scenario.rb_capacity[cell]

        resource_row = lil_matrix((1, variable_count), dtype=float)
        for index in cell_to_indices[cell]:
            resource_row[0, index] = candidates[index].required_rb
        rows.append(resource_row.tocsr())
        lower_bounds.append(-np.inf)
        upper_bounds.append(float(capacity))

        peak_row = lil_matrix((1, variable_count), dtype=float)
        for index in cell_to_indices[cell]:
            peak_row[0, index] = candidates[index].required_rb
        peak_row[0, rho_index] = -float(capacity)
        rows.append(peak_row.tocsr())
        lower_bounds.append(-np.inf)
        upper_bounds.append(0.0)

    objective = np.zeros(variable_count)
    objective[rho_index] = 1.0
    bounds = Bounds(lb=np.zeros(variable_count), ub=np.r_[np.ones(y_count), [1.0]])
    integrality = np.r_[np.ones(y_count), [0]]

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(vstack(rows, format="csr"), lower_bounds, upper_bounds),
        options={"time_limit": time_limit_s, "mip_rel_gap": 0.0},
    )

    if not result.success or result.x is None:
        return SimulationResult(
            status="infeasible",
            objective=float("inf"),
            schedule=(),
            metrics={"infeasible_packet_count": 0.0},
            infeasible_packets=(),
            message=result.message,
        )

    chosen = [index for index in range(y_count) if result.x[index] >= 0.5]
    schedule = []
    for index in chosen:
        candidate = candidates[index]
        packet = packets[candidate.packet]
        schedule.append(
            ScheduleEntry(
                flow_id=packet.flow_id,
                packet_index=packet.packet_index,
                arrival_ms=packet.arrival_ms,
                deadline_ms=packet.deadline_ms,
                link=candidate.link,
                slot=candidate.slot,
                required_rb=candidate.required_rb,
                bits_per_rb=candidate.bits_per_rb,
                delay_ms=candidate.delay_ms,
            )
        )

    schedule_tuple = tuple(sorted(schedule, key=lambda item: (item.slot, item.link, item.flow_id)))
    metrics = compute_metrics(scenario, schedule_tuple, packets, ())
    return SimulationResult(
        status="optimal" if getattr(result, "mip_gap", 0.0) == 0 else "feasible",
        objective=metrics["effective_peak_load"],
        schedule=schedule_tuple,
        metrics=metrics,
        infeasible_packets=(),
        message=result.message,
    )
