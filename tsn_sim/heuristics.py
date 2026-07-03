"""Heuristic schedulers for first-pass simulation."""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Tuple

from .candidate import build_candidates, instantiate_packets
from .config import HeuristicConfig
from .metrics import compute_metrics
from .models import (
    Candidate,
    CellKey,
    Packet,
    PacketKey,
    Scenario,
    ScheduleEntry,
    SimulationResult,
)


def schedule_with_heuristic(
    scenario: Scenario, heuristic: HeuristicConfig | None = None, seed: int = 0
) -> SimulationResult:
    heuristic = heuristic or HeuristicConfig()
    rng = random.Random(seed)
    packets = instantiate_packets(scenario)
    candidates_by_packet, initially_infeasible = build_candidates(scenario, packets)

    remaining_rb = dict(scenario.rb_capacity)
    used_rb: Dict[CellKey, int] = {cell: 0 for cell in scenario.rb_capacity}
    schedule: List[ScheduleEntry] = []
    infeasible = list(initially_infeasible)

    for packet_key in _ordered_packets(packets, candidates_by_packet, heuristic.strategy, rng):
        if packet_key in initially_infeasible:
            continue

        selected = _select_candidate(
            packet_key=packet_key,
            candidates=candidates_by_packet[packet_key],
            scenario=scenario,
            remaining_rb=remaining_rb,
            used_rb=used_rb,
            heuristic=heuristic,
            rng=rng,
        )

        if selected is None:
            infeasible.append(packet_key)
            continue

        remaining_rb[(selected.link, selected.slot)] -= selected.required_rb
        used_rb[(selected.link, selected.slot)] += selected.required_rb
        packet = packets[packet_key]
        schedule.append(
            ScheduleEntry(
                flow_id=packet.flow_id,
                packet_index=packet.packet_index,
                arrival_ms=packet.arrival_ms,
                deadline_ms=packet.deadline_ms,
                link=selected.link,
                slot=selected.slot,
                required_rb=selected.required_rb,
                bits_per_rb=selected.bits_per_rb,
                delay_ms=selected.delay_ms,
            )
        )

    schedule_tuple = tuple(sorted(schedule, key=lambda item: (item.slot, item.link, item.flow_id)))
    infeasible_tuple = tuple(infeasible)
    metrics = compute_metrics(scenario, schedule_tuple, packets, infeasible_tuple)
    status = "feasible" if not infeasible_tuple else "partial"

    return SimulationResult(
        status=status,
        objective=metrics["effective_peak_load"],
        schedule=schedule_tuple,
        metrics=metrics,
        infeasible_packets=infeasible_tuple,
        message=(
            "All packets scheduled by heuristic."
            if status == "feasible"
            else "Heuristic left some packets unscheduled."
        ),
    )


def _ordered_packets(
    packets: Mapping[PacketKey, Packet],
    candidates_by_packet: Mapping[PacketKey, List[Candidate]],
    strategy: str,
    rng: random.Random,
) -> List[PacketKey]:
    if strategy == "random_feasible":
        keys = list(packets)
        rng.shuffle(keys)
        return keys

    # EDF-based ordering: used by edf_min_load and edf_min_peak
    if strategy in ("edf_min_load", "edf_min_peak"):
        return sorted(
            packets,
            key=lambda key: (
                packets[key].arrival_ms + packets[key].deadline_ms,
                packets[key].period_ms,
                key,
            ),
        )

    # urgency ordering: narrowest window first
    return sorted(
        packets,
        key=lambda key: (
            packets[key].latest_slot - packets[key].earliest_slot,
            len(candidates_by_packet[key]),
            packets[key].arrival_ms + packets[key].deadline_ms,
            packets[key].period_ms,
            key,
        ),
    )


def _select_candidate(
    packet_key: PacketKey,
    candidates: List[Candidate],
    scenario: Scenario,
    remaining_rb: Mapping[CellKey, int],
    used_rb: Mapping[CellKey, int],
    heuristic: HeuristicConfig,
    rng: random.Random,
) -> Candidate | None:
    feasible = [
        candidate
        for candidate in candidates
        if remaining_rb[(candidate.link, candidate.slot)] >= candidate.required_rb
    ]
    if not feasible:
        return None

    if heuristic.strategy == "random_feasible":
        return rng.choice(feasible)

    # edf_min_load: choose slot with lowest load after placement
    if heuristic.strategy == "edf_min_load":
        return min(
            feasible,
            key=lambda candidate: (
                _slot_load_after(candidate, scenario, used_rb),
                candidate.delay_ms,
                -candidate.bits_per_rb,
                candidate.link,
                candidate.slot,
            ),
        )

    # urgency_lexicographic and edf_min_peak: lexicographic scoring (Plan B)
    # Primary = peak_after, Secondary = slot_load_after (captures channel efficiency)
    # Then link_avg, delay ratio, efficiency as further tiebreakers
    return min(
        feasible,
        key=lambda candidate: (
            _score_candidate(candidate, scenario, used_rb),
            candidate.delay_ms,
            candidate.link,
            candidate.slot,
            packet_key,
        ),
    )


def _score_candidate(
    candidate: Candidate,
    scenario: Scenario,
    used_rb: Mapping[CellKey, int],
) -> tuple:
    """Lexicographic scoring: (peak_after, slot_load_after, link_avg_after, delay_ratio, -efficiency).
    
    Each dimension is compared in order — peak dominates, slot_load is secondary,
    then link balance, delay urgency, and channel efficiency as tiebreakers.
    This replaces the previous weighted-sum approach (Plan B).
    """
    slot_load_after = _slot_load_after(candidate, scenario, used_rb)
    peak_after = max(_current_peak(scenario, used_rb), slot_load_after)
    link_avg_after = _link_average_after(candidate, scenario, used_rb)
    delay_ratio = candidate.delay_ms / max(1e-9, _deadline_for_packet(candidate, scenario))
    efficiency_bonus = candidate.bits_per_rb / _max_bits_for_flow(candidate, scenario)

    return (
        peak_after,
        slot_load_after,
        link_avg_after,
        delay_ratio,
        -efficiency_bonus,
    )


def _peak_after_candidate(
    candidate: Candidate, scenario: Scenario, used_rb: Mapping[CellKey, int]
) -> float:
    """Peak normalized load if this candidate is placed."""
    slot_load = _slot_load_after(candidate, scenario, used_rb)
    current_peak = _current_peak(scenario, used_rb)
    return max(current_peak, slot_load)


def _slot_load_after(
    candidate: Candidate, scenario: Scenario, used_rb: Mapping[CellKey, int]
) -> float:
    cell = (candidate.link, candidate.slot)
    return (used_rb[cell] + candidate.required_rb) / scenario.rb_capacity[cell]


def _current_peak(scenario: Scenario, used_rb: Mapping[CellKey, int]) -> float:
    return max(
        used_rb[cell] / scenario.rb_capacity[cell]
        for cell in scenario.rb_capacity
    )


def _link_average_after(
    candidate: Candidate, scenario: Scenario, used_rb: Mapping[CellKey, int]
) -> float:
    total = 0.0
    for slot in range(scenario.hyperperiod_ms):
        cell = (candidate.link, slot)
        extra = candidate.required_rb if slot == candidate.slot else 0
        total += (used_rb[cell] + extra) / scenario.rb_capacity[cell]
    return total / scenario.hyperperiod_ms


def _deadline_for_packet(candidate: Candidate, scenario: Scenario) -> float:
    flow_id = candidate.packet[0]
    for flow in scenario.flows:
        if flow.flow_id == flow_id:
            return flow.deadline_ms
    return 1.0


def _max_bits_for_flow(candidate: Candidate, scenario: Scenario) -> int:
    flow_id = candidate.packet[0]
    return max(
        scenario.rb_bits[(flow_id, link, slot)]
        for link in scenario.links
        for slot in range(scenario.hyperperiod_ms)
        if (flow_id, link, slot) in scenario.rb_bits
    )
