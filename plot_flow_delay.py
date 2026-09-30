# -*- coding: utf-8 -*-
"""20 流（零样本）三算法 native vs dimred 时延对比图。

数据来源：data/31_flow_load_<flows>.json（export_flow_load.py 生成）。
只读 JSON，不评估、不训练。

输出一幅图：
  - figure_flow<flows>_delay_vs.png        native vs dimred 并排对比（grouped bar）

口径：
  - average_delay_ms：每个 episode 调度完成包的平均时延，越低越好。
  - 零样本评估：50 流训练的 best checkpoint 直接在指定流数场景上贪婪评估。
  - 跨 3 seed 求 mean（ddof=1）。时延差异量级小，y 轴取非零下界以便观察差异。

用法：
  RL/bin/python plot_flow_delay.py                 # 默认画 20 流
  RL/bin/python plot_flow_delay.py --flows 20,50   # 画多个流数（各自出一组图）
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

ALGORITHMS = [
    ("PPO", "#1f77b4"),
    ("SAC", "#d62728"),
    ("TD3", "#2ca02c"),
]

NATIVE_COLOR = "#1f77b4"
DIMRED_COLOR = "#ff7f0e"

METRIC = "avg_delay_ms"


def load_data(flow):
    p = os.path.join(DATA, f"31_flow_load_{flow}.json")
    if not os.path.exists(p):
        print(f"[MISSING] {p}")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get_stats(d, mode, algo, flow):
    """返回 (mean, std, n_seed)。"""
    e = d["results"][mode][algo].get(str(flow), {}).get(METRIC, {})
    if not e:
        return None
    per_seed = e.get("per_seed", {})
    vals = [v for v in per_seed.values() if v is not None]
    mean = e.get("mean")
    if mean is None:
        mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(mean), std, len(vals)


def _ylim(means):
    """时延差异量级小，取非零下界放大差异。"""
    lo = min(means)
    hi = max(means)
    span = hi - lo
    pad = span * 0.6 + 0.05   # 给柱顶文字留空间
    return max(0.0, lo - pad), hi + pad


def plot_vs(d, flow, out_name, title):
    names, n_means, d_means = [], [], []
    for algo, _ in ALGORITHMS:
        ns = get_stats(d, "native", algo, flow)
        ds = get_stats(d, "dimred", algo, flow)
        if ns is None or ds is None:
            print(f"  [skip vs] {algo}")
            continue
        names.append(algo)
        n_means.append(ns[0])
        d_means.append(ds[0])
        print(f"  [vs] {algo}: native={ns[0]:.4f}  dimred={ds[0]:.4f}")

    if not names:
        print("  [vs] 无有效数据")
        return

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(x - width/2, n_means, width, color=NATIVE_COLOR, edgecolor="black",
           linewidth=1.2, label="native")
    ax.bar(x + width/2, d_means, width, color=DIMRED_COLOR, edgecolor="black",
           linewidth=1.2, label="dimred")
    for xi, nm in enumerate(n_means):
        ax.text(xi - width/2, nm + 0.01, f"{nm:.3f}", ha="center",
                va="bottom", fontsize=9, fontweight="bold", color=NATIVE_COLOR)
    for xi, dm in enumerate(d_means):
        ax.text(xi + width/2, dm + 0.01, f"{dm:.3f}", ha="center",
                va="bottom", fontsize=9, fontweight="bold", color=DIMRED_COLOR)
    ax.set_ylabel("Average delay (ms)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(*_ylim(n_means + d_means))
    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", type=str, default="20", help="逗号分隔的流数")
    args = ap.parse_args()
    flows = [int(x) for x in args.flows.split(",") if x.strip()]

    for flow in flows:
        d = load_data(flow)
        if d is None:
            continue
        print(f"\n=== flows={flow} ===")
       
        plot_vs(d, flow,
                f"exam2_delay.png",
                f"Average delay - native vs dimred | {flow} flows")


if __name__ == "__main__":
    main()
