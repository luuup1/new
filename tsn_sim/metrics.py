"""Metrics and output helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .models import CellKey, Packet, PacketKey, Scenario, ScheduleEntry


def compute_cell_load(
    scenario: Scenario, schedule: Sequence[ScheduleEntry]
) -> Dict[CellKey, int]:
    load = {
        (link, slot): 0
        for link in scenario.links
        for slot in range(scenario.hyperperiod_ms)
    }
    for item in schedule:
        load[(item.link, item.slot)] += item.required_rb
    return load


def compute_metrics(
    scenario: Scenario,
    schedule: Sequence[ScheduleEntry],
    packets: Mapping[PacketKey, Packet],
    infeasible_packets: Sequence[PacketKey],
) -> Mapping[str, float]:
    load = compute_cell_load(scenario, schedule)
    normalized = {
        cell: used / scenario.rb_capacity[cell]
        for cell, used in load.items()
    }
    link_averages = []
    link_peaks = []
    for link in scenario.links:
        values = [normalized[(link, slot)] for slot in range(scenario.hyperperiod_ms)]
        link_averages.append(sum(values) / len(values))
        link_peaks.append(max(values))

    total_used = sum(load.values())
    total_capacity = sum(scenario.rb_capacity.values())
    total_packets = len(packets)
    scheduled_count = len(schedule)
    infeasible_count = len(infeasible_packets)

    # Deadline checks — only meaningful for *scheduled* packets
    deadline_ok_count = sum(1 for item in schedule if item.delay_ms <= item.deadline_ms)

    # Scheduling success rate: fraction of total packets that got a slot
    scheduling_success_rate = scheduled_count / total_packets if total_packets else 1.0

    # True deadline satisfaction rate: fraction of *scheduled* packets that
    # met their deadline.  This is distinct from scheduling_success_rate because
    # candidate generation already guarantees deadline feasibility — but the
    # separation makes the metric semantically honest for future DRL work
    # where the agent might schedule a packet past its deadline.
    true_deadline_satisfaction_rate = (
        deadline_ok_count / scheduled_count if scheduled_count else 1.0
    )

    # Drop-penalized peak load: peak_load adjusted for unscheduled packets.
    # Without this, a scheduler that drops heavy packets can appear to have a
    # *lower* peak, which is misleading.  The α=1.0 penalty means each dropped
    # packet adds a flat 1.0 unit to the effective peak — chosen so that a
    # single dropped packet always dominates over any load reduction.
    peak_load_raw = max(normalized.values()) if normalized else 0.0
    drop_ratio = infeasible_count / total_packets if total_packets else 0.0
    effective_peak_load = peak_load_raw + drop_ratio  # α = 1.0 per dropped packet

    return {
        "peak_load": peak_load_raw,
        "effective_peak_load": effective_peak_load,
        "average_load": sum(normalized.values()) / len(normalized) if normalized else 0.0,
        "resource_utilization": total_used / total_capacity if total_capacity else 0.0,
        "scheduling_success_rate": scheduling_success_rate,
        "true_deadline_satisfaction_rate": true_deadline_satisfaction_rate,
        "deadline_satisfaction_ratio": scheduling_success_rate,  # backward compat alias
        "scheduled_packet_count": float(scheduled_count),
        "infeasible_packet_count": float(infeasible_count),
        "drop_ratio": drop_ratio,
        "load_variance_across_links": _variance(link_averages),
        "max_load_per_link": max(link_peaks) if link_peaks else 0.0,
        "average_delay_ms": (
            sum(item.delay_ms for item in schedule) / len(schedule) if schedule else 0.0
        ),
    }


def write_schedule_csv(schedule: Sequence[ScheduleEntry], path: str | Path) -> None:
    rows = [asdict(item) for item in schedule]
    fieldnames = list(rows[0].keys()) if rows else list(ScheduleEntry.__dataclass_fields__)
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_json(metrics: Mapping[str, float], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)
