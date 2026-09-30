# -*- coding: utf-8 -*-
"""零样本多流负载对比：三算法 × native/dimred 在指定流数上评估（只导出 JSON）。

背景：
  plot_load_comparison.py 读的 04/11_max_load.json 是「50 流场景训练 + 50 流评估」的
  训练全程 eval 均值。本脚本补一种「零样本泛化」口径：用 50 流训练好的 best checkpoint
  直接在其它流数（默认 20）场景上贪婪评估，得到 native vs dimred 三算法负载对比。

  观测/动作维度由 L=3 × S=32 决定，与 flow_count 无关，因此可零样本直接评估。

评估口径（与 export_scale_stress.py 完全一致）：
  - period_mode="simple"、order_mode="random"、obs_mode="full"
  - 每个 seed 用「对应 seed 的 best checkpoint」在「对应 seed 的场景」上贪婪评估
  - 指标：effective_peak_load / average_delay_ms / scheduling_success_rate / drop_ratio

输出：
  data/31_flow_load_<flows>.json
      results.<native|dimred>.<PPO|SAC|TD3>.<metric>.mean / .std / .per_seed

只写数据文件，不绘图、不修改任何 checkpoint。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_scale_stress import (
    AGENT_CLASSES,
    AGENT_EXTRA_KWARGS,
    BEST_FILES,
    CHECKPOINT_DIRS,
    eval_one,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

ALGORITHMS = ["PPO", "SAC", "TD3"]
MODES = ["native", "dimred"]

METRICS = {
    "eff_peak": "effective_peak_load (lower=better)",
    "avg_delay_ms": "average_delay_ms (lower=better)",
    "success_rate": "scheduling_success_rate (higher=better)",
    "drop_ratio": "drop_ratio (lower=better)",
}


def summarize(records):
    out = {}
    for metric in METRICS:
        vals = {str(r["seed"]): float(r[metric]) for r in records}
        arr = np.array(list(vals.values()), dtype=float)
        out[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "per_seed": vals,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="零样本三算法负载对比（指定流数）")
    ap.add_argument("--flows", type=str, default="20", help="逗号分隔的流数序列（默认 20）")
    ap.add_argument("--seeds", type=str, default="42,123,2024", help="种子序列")
    args = ap.parse_args()

    flows = [int(x) for x in args.flows.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    print(f"=== 零样本负载对比 ===\n  流数: {flows}  种子: {seeds}  算法: {ALGORITHMS}\n")

    results = {mode: {} for mode in MODES}
    for mode in MODES:
        for algo in ALGORITHMS:
            per_flow = {}
            for fc in flows:
                records = []
                for sd in seeds:
                    try:
                        r = eval_one(algo, mode, fc, sd)
                        records.append(r)
                        print(f"  [{algo}/{mode}] flows={fc:3d} seed={sd:4d}  "
                              f"eff_peak={r['eff_peak']:.4f}  "
                              f"avg_delay={r['avg_delay_ms']:.3f}ms  "
                              f"accept={r['success_rate']*100:.1f}%  "
                              f"drop={r['drop_ratio']*100:.1f}%")
                    except FileNotFoundError as e:
                        print(f"  [SKIP] {e}")
                if records:
                    per_flow[fc] = summarize(records)
            results[mode][algo] = per_flow

    tag = "_".join(str(f) for f in flows)
    data = {
        "meta": {
            "flows": flows,
            "seeds": seeds,
            "algorithms": ALGORITHMS,
            "modes": MODES,
            "period_mode": "simple",
            "order_mode": "random",
            "obs_mode": "full",
            "metrics": METRICS,
            "note": (
                "零样本评估：用 50 流训练的 best checkpoint 在指定流数场景上贪婪评估；"
                "跨 seed 求 mean±std（ddof=1）。"
            ),
        },
        "results": results,
    }

    out_path = os.path.join(OUT, f"31_flow_load_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n已导出 -> {out_path}")

    # 控制台汇总：eff_peak mean ± std
    print("\n=== 汇总：eff_peak (mean ± std, lower=better) ===")
    for mode in MODES:
        for algo in ALGORITHMS:
            for fc in flows:
                e = results[mode][algo].get(fc, {}).get("eff_peak", {})
                if not e:
                    continue
                print(f"  {mode:7s} {algo:4s} flows={fc:3d}  "
                      f"eff_peak={e['mean']:.4f}±{e['std']:.4f}")


if __name__ == "__main__":
    main()
