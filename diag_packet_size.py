# -*- coding: utf-8 -*-
"""诊断：不同「包大小 × 流数」下的负载 / 丢包情况（只诊断，不改任何配置/数据）。

目的：
  实验四发现 60 流起 peak_load 已打满（=1.0），之后 eff_peak 超过 1 全靠丢包惩罚。
  本脚本扫描「包大小缩放 × 流数」网格，用确定性 greedy 启发式调度，输出每个组合的
  物理峰值负载(peak_load)、丢包率(drop_ratio)、有效峰值(eff_peak)、平均时延，
  帮你判断「包该设多大」才能在目标流数下保持零丢包（或可接受丢包）。

为什么用 greedy 而非 RL：
  - greedy 是确定性启发式，代表「该场景下的物理可行性上限」，且秒级跑完。
  - RL 的调度能力通常略弱于或接近 greedy，所以 greedy 零丢包是 RL 可达到的必要条件。

扫描维度：
  - 流数 flow_count ∈ {50, 60, 80, 100, 150, 200}
  - 包大小缩放 scale（等价于 packet_size_bits * load_scale，1.0 = 当前默认）
    当前默认 packet_size_bits = (800, 1200, 1800, 2400)。
  - 场景 seed ∈ {42, 123, 2024}，取 mean。

用法：
  RL/bin/python diag_packet_size.py
  RL/bin/python diag_packet_size.py --flows 50,60,80,100,200 --scales 0.25,0.5,0.75,1.0,1.25,1.5
  RL/bin/python diag_packet_size.py --csv   # 额外导出 CSV
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tsn_sim import SimulationConfig, build_scenario, schedule_with_heuristic

HERE = os.path.dirname(os.path.abspath(__file__))

SEEDS = (42, 123, 2024)
STRATEGY = "greedy"


def diagnose(flow_count: int, scale: float) -> dict:
    """给定流数与包大小缩放，返回跨 seed 的 mean 指标。"""
    peaks, drops, effs, delays = [], [], [], []
    for seed in SEEDS:
        scenario = build_scenario(
            SimulationConfig(
                seed=seed,
                period_mode="simple",
                flow_count=flow_count,
                load_scale=scale,
            )
        )
        result = schedule_with_heuristic(scenario, STRATEGY, seed=seed)
        m = result.metrics
        peaks.append(float(m["peak_load"]))
        drops.append(float(m["drop_ratio"]))
        effs.append(float(m["effective_peak_load"]))
        delays.append(float(m["average_delay_ms"]))
    return {
        "flow_count": flow_count,
        "scale": scale,
        "peak_load": sum(peaks) / len(peaks),
        "drop_ratio": sum(drops) / len(drops),
        "eff_peak": sum(effs) / len(effs),
        "avg_delay_ms": sum(delays) / len(delays),
    }


def main():
    parser = argparse.ArgumentParser(description="诊断：包大小 × 流数 的负载/丢包")
    parser.add_argument("--flows", type=str, default="50,60,80,100,150,200")
    parser.add_argument("--scales", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--csv", action="store_true", help="额外导出 CSV")
    args = parser.parse_args()

    flows = [int(x) for x in args.flows.split(",") if x.strip()]
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    print("=" * 90)
    print(f"诊断：包大小缩放 × 流数  （调度策略={STRATEGY}，seed={SEEDS} 取 mean）")
    print("=" * 90)
    header = f"{'flows':>6} {'scale':>6} {'peak_load':>10} {'drop_ratio':>10} {'eff_peak':>9} {'avg_delay':>9}"
    print(header)
    print("-" * 90)

    rows = []
    for fc in flows:
        for sc in scales:
            r = diagnose(fc, sc)
            rows.append(r)
            print(f"{fc:>6} {sc:>6.2f} {r['peak_load']:>10.3f} "
                  f"{r['drop_ratio']:>10.3f} {r['eff_peak']:>9.3f} "
                  f"{r['avg_delay_ms']:>9.2f}")

    # 汇总：每个流数下「零丢包的最大 scale」
    print("\n" + "=" * 90)
    print("关键结论：每个流数下，能保持『零丢包(drop_ratio≈0)』的最大包大小缩放 scale")
    print("=" * 90)
    for fc in flows:
        zero_ok = [r["scale"] for r in rows if r["flow_count"] == fc and r["drop_ratio"] < 0.005]
        if zero_ok:
            print(f"  flows={fc:>3d}: 零丢包的最大 scale ≈ {max(zero_ok):.2f}")
        else:
            print(f"  flows={fc:>3d}: 所有 scale 都有丢包（最小 scale {min(scales):.2f} 也 >0.5% 丢包）")

    if args.csv:
        out = os.path.join(HERE, "diag_packet_size.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["flow_count", "scale", "peak_load",
                                              "drop_ratio", "eff_peak", "avg_delay_ms"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n已导出 CSV -> {out}")


if __name__ == "__main__":
    main()
