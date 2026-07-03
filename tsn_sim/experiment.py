"""Experiment runners for parameter comparisons."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Mapping

from .config import HeuristicConfig, SimulationConfig
from .heuristics import schedule_with_heuristic
from .milp import solve_optimal_milp
from .scenario import build_scenario


def run_single(config: SimulationConfig):
    scenario = build_scenario(config)
    return schedule_with_heuristic(scenario, config.heuristic, seed=config.seed)


def compare_link_counts(config: SimulationConfig, counts: Iterable[int] = (1, 2, 3)):
    rows: List[Mapping[str, float | int | str]] = []
    for count in counts:
        run_config = replace(config, link_count=count)
        result = run_single(run_config)
        rows.append(
            {
                "links": count,
                "status": result.status,
                "peak_load": result.metrics["peak_load"],
                "effective_peak_load": result.metrics["effective_peak_load"],
                "resource_utilization": result.metrics["resource_utilization"],
                "drop_ratio": result.metrics["drop_ratio"],
                "average_delay_ms": result.metrics["average_delay_ms"],
            }
        )
    return rows


def sweep_load_scale(config: SimulationConfig, scales: Iterable[float]):
    rows: List[Mapping[str, float | str]] = []
    for scale in scales:
        run_config = replace(config, load_scale=scale)
        result = run_single(run_config)
        rows.append(
            {
                "load_scale": scale,
                "status": result.status,
                "peak_load": result.metrics["peak_load"],
                "effective_peak_load": result.metrics["effective_peak_load"],
                "resource_utilization": result.metrics["resource_utilization"],
                "drop_ratio": result.metrics["drop_ratio"],
                "average_delay_ms": result.metrics["average_delay_ms"],
            }
        )
    return rows


def compare_heuristics(
    config: SimulationConfig,
    strategies: Iterable[str] = ("random_feasible", "edf_min_load", "urgency_lexicographic", "edf_min_peak"),
):
    rows: List[Mapping[str, float | str]] = []
    scenario = build_scenario(config)
    for strategy in strategies:
        heuristic = _heuristic_with_strategy(config.heuristic, strategy)
        result = schedule_with_heuristic(scenario, heuristic, seed=config.seed)
        rows.append(
            {
                "strategy": strategy,
                "status": result.status,
                "peak_load": result.metrics["peak_load"],
                "effective_peak_load": result.metrics["effective_peak_load"],
                "scheduling_success_rate": result.metrics["scheduling_success_rate"],
                "true_deadline_satisfaction_rate": result.metrics["true_deadline_satisfaction_rate"],
                "resource_utilization": result.metrics["resource_utilization"],
                "drop_ratio": result.metrics["drop_ratio"],
                "average_delay_ms": result.metrics["average_delay_ms"],
            }
        )
    return rows


def compare_with_milp(
    config: SimulationConfig,
    strategies: Iterable[str] = ("random_feasible", "edf_min_load", "urgency_lexicographic", "edf_min_peak"),
    time_limit_s: float = 30.0,
):
    scenario = build_scenario(config)
    optimal = solve_optimal_milp(scenario, time_limit_s=time_limit_s)
    opt_effective = optimal.metrics.get("effective_peak_load", float("inf"))
    optimal_is_comparable = optimal.status in {"optimal", "feasible"} and opt_effective not in (
        0.0,
        float("inf"),
    )
    rows: List[Mapping[str, float | str]] = [
        {
            "solver": "milp_optimal",
            "status": optimal.status,
            "peak_load": optimal.metrics.get("peak_load", float("inf")),
            "effective_peak_load": opt_effective,
            "drop_ratio": "n/a" if not optimal_is_comparable else optimal.metrics.get("drop_ratio", 0.0),
            "gap_to_milp": "n/a" if not optimal_is_comparable else 0.0,
        }
    ]

    for strategy in strategies:
        heuristic = _heuristic_with_strategy(config.heuristic, strategy)
        result = schedule_with_heuristic(scenario, heuristic, seed=config.seed)
        eff_peak = result.metrics["effective_peak_load"]
        gap = (eff_peak - opt_effective) / opt_effective if optimal_is_comparable else "n/a"
        rows.append(
            {
                "solver": strategy,
                "status": result.status,
                "peak_load": result.metrics["peak_load"],
                "effective_peak_load": eff_peak,
                "drop_ratio": result.metrics["drop_ratio"],
                "gap_to_milp": gap,
            }
        )
    return rows


def _heuristic_with_strategy(base: HeuristicConfig, strategy: str) -> HeuristicConfig:
    return HeuristicConfig(
        strategy=strategy,
        peak_weight=base.peak_weight,
        slot_load_weight=base.slot_load_weight,
        link_avg_weight=base.link_avg_weight,
        delay_weight=base.delay_weight,
        efficiency_weight=base.efficiency_weight,
    )
