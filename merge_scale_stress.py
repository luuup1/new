# -*- coding: utf-8 -*-
"""合并三个 21_scale_stress_*.json 为统一的 21_scale_stress.json。

数据源：
  - data/21_scale_stress_td3_perflow.json   （TD3 per-flow，20/40/60 流，seed42 单点）
  - data/21_scale_stress_td3.json           （TD3 zero-shot，80 流，跨 3 seed）
  - data/21_scale_stress_heuristics.json    （greedy + genetic_algorithm，跨 3 seed）

合并后结构（新，供 plot_scale_stress.py 读取）：
  results.<flow_count>.<algo>.<metric>.{mean, std, per_seed}
  其中 algo ∈ {td3, greedy, ga}，flow ∈ {20, 40, 60, 80}。

TD3 口径（已与用户确认）：
  - 20/40/60 流：per-flow 训练（每档流数各自训练，seed42 单点）
  - 80 流：zero-shot（50 流模型直接评估，跨 3 seed 均值）
  （80 流场景物理饱和，per-flow 训练收益退化，故保留 zero-shot。）

用法：
  RL/bin/python merge_scale_stress.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

METRICS = {
    "eff_peak": "effective_peak_load = max(cell 归一化负载) + drop_ratio (lower=better)",
    "avg_delay_ms": "average_delay_ms (lower=better)",
    "success_rate": "scheduling_success_rate (higher=better)",
    "drop_ratio": "drop_ratio (lower=better)",
    "max_delay_ms": "max_delay_ms (lower=better)",
}

FLOWS = [20, 40, 60, 80]
ALGOS = ["td3", "greedy", "ga"]


def load(name: str) -> dict:
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    td3_pf = load("21_scale_stress_td3_perflow.json")["results"]["dimred"]
    td3_zs = load("21_scale_stress_td3.json")["results"]["dimred"]
    heur = load("21_scale_stress_heuristics.json")["results"]

    results = {}
    td3_sources = {}
    for fc in FLOWS:
        key = str(fc)
        # TD3：优先 per-flow（20/40/60），缺失时回退 zero-shot（80）
        if key in td3_pf:
            td3 = td3_pf[key]
            td3_sources[key] = "per-flow (seed42 单点)"
        else:
            td3 = td3_zs[key]
            td3_sources[key] = "zero-shot (跨 3 seed)"

        results[key] = {
            "td3": td3,
            "greedy": heur["greedy"][key],
            "ga": heur["genetic_algorithm"][key],
        }

    out = {
        "meta": {
            "period_mode": "simple",
            "order_mode": "random",
            "obs_mode": "full",
            "flow_counts": FLOWS,
            "algorithms": ALGOS,
            "metrics": METRICS,
            "td3_source": td3_sources,
            "note": (
                "TD3：20/40/60 流为 per-flow 训练（seed42 单点，同流数评估）；"
                "80 流为 zero-shot（跨 3 seed）。"
                "greedy / ga 为确定性启发式当场重算，跨 3 seed 求 mean±std（ddof=1）。"
                "eff_peak / avg_delay_ms / max_delay_ms 越低越好；"
                "success_rate 越高越好；drop_ratio 越低越好。"
            ),
        },
        "results": results,
    }

    out_path = os.path.join(DATA, "21_scale_stress.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已导出 -> {out_path}")

    # 控制台汇总 eff_peak
    print("\n=== 合并结果汇总（eff_peak，mean±std）===")
    for fc in FLOWS:
        parts = []
        for algo in ALGOS:
            e = results[str(fc)][algo]["eff_peak"]
            parts.append(f"{algo}={e['mean']:.4f}")
        print(f"  flows={fc:3d}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
