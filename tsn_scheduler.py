"""Command-line entry point for the 5G-TSN heuristic simulator."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tsn_sim.config import default_config, load_config
from tsn_sim.config import HeuristicConfig
from tsn_sim.experiment import (
    compare_heuristics,
    compare_link_counts,
    compare_with_milp,
    run_single,
    sweep_load_scale,
)
from tsn_sim.metrics import write_metrics_json, write_schedule_csv
from tsn_sim.scenario import build_scenario
from tsn_sim.stats import problem_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--flows", type=int, help="Flow count override.")
    parser.add_argument("--links", type=int, choices=(1, 2, 3), help="Link count override.")
    parser.add_argument("--load-scale", type=float, help="Packet-size load multiplier.")
    parser.add_argument(
        "--strategy",
        choices=("random_feasible", "greedy", "proportional_fair", "genetic_algorithm"),
        help="Heuristic strategy override.",
    )
    parser.add_argument("--preview", type=int, default=30, help="Rows of schedule preview.")
    parser.add_argument("--compare-links", action="store_true", help="Compare 1, 2, and 3 links.")
    parser.add_argument("--compare-heuristics", action="store_true", help="Compare heuristic baselines.")
    parser.add_argument(
        "--compare-milp",
        action="store_true",
        help="Compare heuristics against small-scale MILP optimum.",
    )
    parser.add_argument("--stats", action="store_true", help="Print generated problem size diagnostics.")
    parser.add_argument("--milp-time-limit", type=float, default=30.0, help="MILP time limit in seconds.")
    parser.add_argument(
        "--sweep-load",
        nargs="+",
        type=float,
        help="Run several load scales, for example: --sweep-load 0.8 1.0 1.2",
    )
    parser.add_argument("--schedule-csv", help="Write schedule rows to CSV.")
    parser.add_argument("--metrics-json", help="Write metrics to JSON.")
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    config = load_config(args.config) if args.config else default_config()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.flows is not None:
        overrides["flow_count"] = args.flows
    if args.links is not None:
        overrides["link_count"] = args.links
    if args.load_scale is not None:
        overrides["load_scale"] = args.load_scale
    config = replace(config, **overrides)
    if args.strategy is not None:
        config = replace(config, heuristic=HeuristicConfig(strategy=args.strategy))
    return config


def print_result(result, preview: int) -> None:
    print(f"status: {result.status}")
    print(f"message: {result.message}")
    print(f"objective peak_load: {result.objective:.4f}")
    print("metrics:")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.4f}")

    if result.infeasible_packets:
        print(f"unscheduled packets: {len(result.infeasible_packets)}")

    limit = min(preview, len(result.schedule))
    print(f"schedule preview, first {limit} rows:")
    print("  flow packet arrival deadline link slot rb q delay")
    for item in result.schedule[:limit]:
        print(
            "  "
            f"{item.flow_id:>4} {item.packet_index:>6} "
            f"{item.arrival_ms:>7.2f} {item.deadline_ms:>8.2f} "
            f"{item.link:>4} {item.slot:>4} {item.required_rb:>2} "
            f"{item.bits_per_rb:>4} {item.delay_ms:>5.2f}"
        )


def print_rows(rows) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    print(" ".join(f"{header:>18}" for header in headers))
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:>18.4f}")
            else:
                values.append(f"{str(value):>18}")
        print(" ".join(values))


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.compare_links:
        print_rows(compare_link_counts(config))
        return

    if args.stats:
        stats = problem_stats(build_scenario(config))
        for name, value in stats.items():
            print(f"{name}: {value:.4f}")
        return

    if args.compare_heuristics:
        print_rows(compare_heuristics(config))
        return

    if args.compare_milp:
        print_rows(compare_with_milp(config, time_limit_s=args.milp_time_limit))
        return

    if args.sweep_load:
        print_rows(sweep_load_scale(config, args.sweep_load))
        return

    result = run_single(config)
    print_result(result, args.preview)

    if args.schedule_csv:
        write_schedule_csv(result.schedule, Path(args.schedule_csv))
    if args.metrics_json:
        write_metrics_json(result.metrics, Path(args.metrics_json))


if __name__ == "__main__":
    main()
