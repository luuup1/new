# -*- coding: utf-8 -*-
"""实验四压力测试：TD3 vs 传统启发式对比图（纯画图，只读 JSON）。

数据来源：data/21_scale_stress.json（由 merge_scale_stress.py 合并生成）。

结构：
  results.<flow_count>.<algo>.<metric>.{mean, std, per_seed}
  其中 algo ∈ {td3, greedy, ga}，flow ∈ {20, 40, 60, 80}。

对比对象：TD3（DIM）+ greedy + GA（两个最强启发式）。
回答 RQ1 可扩展性：随流数增加，各算法的负载/时延/接受率如何退化。

输出三幅图：
  - figure_scale_compare_load.png    有效峰值负载 vs 流数
  - figure_scale_compare_delay.png   平均时延 vs 流数
  - figure_scale_compare_accept.png  接受率 vs 流数

用法：
  RL/bin/python plot_scale_stress_compare.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
JSON_FILE = os.path.join(DATA, "21_scale_stress.json")

# 对比对象：TD3（RL）在前，两个启发式在后
ALGOS = [
    ("td3", "#d62728"),                # 红（提出方法）
    ("greedy", "#2ca02c"),             # 绿
    ("ga", "#9467bd"),                 # 紫
]

DISPLAY_NAMES = {
    "td3": "TD3 (DIM)",
    "greedy": "Greedy",
    "ga": "GA",
}

# (metric, 图名, y 轴标签, 是否百分比)
PLOTS = [
    ("eff_peak", "exam4_load.png",
     "Effective peak load", False),
    ("avg_delay_ms", "exam4_delay.png",
     "Average delay", False),
    ("success_rate", "figure_scale_compare_accept.png",
     "Scheduling success rate (higher = better)", True),
]


def setup_style():
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_data():
    with open(JSON_FILE, encoding="utf-8") as f:
        return json.load(f)


def plot_metric(results, metric, ylabel, out_name, pct):
    """x 轴流数，每个流数下各算法并排柱。

    results: {flow_count(str): {algo: {metric: {mean, std, per_seed}}}}。
    """
    fcs = sorted(int(k) for k in results.keys())
    x = np.arange(len(fcs))
    n = len(ALGOS)
    width = 0.7 / n

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (algo, color) in enumerate(ALGOS):
        means = [results[str(fc)][algo][metric]["mean"] for fc in fcs]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, means, width, color=color,
               edgecolor="black", linewidth=1.0,
               label=DISPLAY_NAMES.get(algo, algo))
        for xi, mv in enumerate(means):
            ax.text(xi + offset, mv + 0.01, _fmt(mv, pct),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Flow count")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(fc) for fc in fcs])
    all_means = [results[str(fc)][algo][metric]["mean"]
                 for algo, _ in ALGOS for fc in fcs]
    if pct:
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    else:
        ax.set_ylim(0, max(all_means) * 1.15)
    ax.legend(framealpha=0.9)
    ax.set_title("TD3 vs heuristics — scalability stress test")
    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out)
    plt.close(fig)
    print(f"  已生成 {out}")


def _fmt(v, pct):
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def main():
    data = load_data()
    results = data["results"]

    setup_style()
    for metric, out_name, ylabel, pct in PLOTS:
        plot_metric(results, metric, ylabel, out_name, pct)

    # 控制台汇总对比
    print("\n=== eff_peak 对比（mean±std）===")
    for fc in sorted(results, key=int):
        parts = []
        for algo, _ in ALGOS:
            e = results[fc][algo]["eff_peak"]
            parts.append(f"{DISPLAY_NAMES[algo]}={e['mean']:.4f}±{e['std']:.4f}")
        print(f"  flows={int(fc):3d}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
