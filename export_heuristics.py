# -*- coding: utf-8 -*-
"""导出 4 种启发式算法的负载与时延数据（供实验五横向对比使用）。

为什么需要这个脚本：
  data/ 下现有的 *_max_load.json、*_delay.json 只覆盖 RL 三算法
  （PPO/SAC/TD3 的 native/dimred），启发式（random_feasible / greedy /
  proportional_fair / genetic_algorithm）从未导出过数据文件。
  实验五要把「4 启发式 + 3 RL」放在同一张图上对比，因此需要先补启发式数据。

口径（与实验二/三严格对齐，保证同一组数据一致性）：
  - 测试场景：period_mode="simple"、flow_count=50、超周期 32（96 cell），
    场景 seed ∈ {42, 123, 2024}，与 RL 训练/评估完全一致。
  - 启发式为确定性调度：给定 seed → scenario 确定 → 调度结果确定，
    每个 seed 只跑一次，不存在「训练全程」概念。
  - 指标（与 RL 的 data/ 文件同源，compute_metrics 统一计算）：
      负载 effective_peak_load = max(cell 归一化负载) + drop_ratio（lower=better）
      时延 average_delay_ms（lower=better）

与 RL 口径的差异说明（论文需注明）：
  - RL 负载用 04_native_max_load.json 的 eval_mean_per_seed（训练全程评估均值，
    典型水平，与实验二一致）。
  - 启发式负载为确定性单次结果，跨 3 seed 求 mean±std（跨 seed 标准差）。

输出：
  - data/20_heuristics.json
      algorithms.<strategy>.eff_peak_per_seed / avg_delay_per_seed
      由 horizontal_compare.py（纯画图脚本）读取绘制实验五两幅图。

只写数据文件，不绘图、不修改任何 checkpoint。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tsn_sim import SimulationConfig, build_scenario, schedule_with_heuristic

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

SEEDS = (42, 123, 2024)
FLOW_COUNT = 50
PERIOD_MODE = "simple"

HEURISTICS = ["random_feasible", "greedy", "proportional_fair", "genetic_algorithm"]


def eval_heuristic(strategy: str, seed: int):
    """确定性单次调度，返回 (eff_peak, avg_delay)。"""
    scenario = build_scenario(
        SimulationConfig(seed=seed, period_mode=PERIOD_MODE, flow_count=FLOW_COUNT)
    )
    result = schedule_with_heuristic(scenario, strategy, seed=seed)
    eff_peak = float(result.metrics["effective_peak_load"])
    avg_delay = float(result.metrics["average_delay_ms"])
    return eff_peak, avg_delay


def main():
    data = {
        "meta": {
            "period_mode": PERIOD_MODE,
            "flow_count": FLOW_COUNT,
            "seeds": list(SEEDS),
            "metric_load": "effective_peak_load = max(cell 归一化负载) + drop_ratio (lower=better)",
            "metric_delay": "average_delay_ms (lower=better)",
            "note": (
                "启发式为确定性单次调度，每 seed 一个值，跨 3 seed 求 mean±std(跨 seed)。"
                "RL 负载用 04_native_max_load.json 的 eval_mean_per_seed（训练全程均值）；"
                "RL 时延用 05/06/16_*_delay.json 的 delay_mean_ms。"
            ),
        },
        "algorithms": {},
    }

    for strategy in HEURISTICS:
        eff, avg = {}, {}
        for seed in SEEDS:
            e, a = eval_heuristic(strategy, seed)
            eff[str(seed)] = round(e, 6)
            avg[str(seed)] = round(a, 6)
            print(f"[{strategy:20s}] seed={seed}  eff_peak={e:.4f}  avg_delay={a:.4f}ms")
        data["algorithms"][strategy] = {
            "eff_peak_per_seed": eff,
            "avg_delay_per_seed": avg,
            "eff_peak_mean": round(float(np.mean(list(eff.values()))), 6),
            "eff_peak_std": round(float(np.std(list(eff.values()), ddof=1)), 6),
            "avg_delay_mean": round(float(np.mean(list(avg.values()))), 6),
            "avg_delay_std": round(float(np.std(list(avg.values()), ddof=1)), 6),
        }

    out_path = os.path.join(OUT, "20_heuristics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n已导出 -> {out_path}")


if __name__ == "__main__":
    main()
