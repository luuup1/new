# -*- coding: utf-8 -*-
"""4 启发式 + 3 强化学习算法横向对比（实验五：负载 + 时延，两幅图）。

本脚本为「纯画图脚本」，只读 data/ 下已导出的 JSON 文件，不做任何在线评估
（与实验二 plot_load_comparison.py、实验三 plot_delay_comparison.py 一致）。

【数据来源 —— 与实验二/三严格复用同一组数据，保证一致性】
  统一口径：period_mode="simple"、flow_count=50、超周期 32（96 cell），
           场景 seed ∈ {42, 123, 2024}。

  ┌─────────────┬───────────────────────────────────────────────┬──────────────────────────┐
  │ 算法        │ 数据文件                                      │ 读取字段                 │
  ├─────────────┼───────────────────────────────────────────────┼──────────────────────────┤
  │ RL 负载     │ data/04_native_max_load.json                  │ algorithms.<algo>.eval_mean_per_seed │
  │ RL 时延     │ data/05_ppo_native_delay.json                │ seeds.<seed>.delay_mean_ms│
  │             │ data/06_sac_native_delay.json                │ 同上                    │
  │             │ data/16_td3_native_delay.json                │ 同上                    │
  │ 启发式      │ data/20_heuristics.json                      │ algorithms.<strategy>.eff_peak_per_seed / avg_delay_per_seed │
  └─────────────┴───────────────────────────────────────────────┴──────────────────────────┘

  口径说明：
  - 负载（lower=better）：effective_peak_load = max(cell 归一化负载) + drop_ratio。
    RL 用 eval_mean_per_seed（训练全程评估均值，典型水平，与实验二一致）；
    启发式为确定性单次调度结果（每 seed 一个值，跨 3 seed 求 mean±std）。
  - 时延（lower=better）：average_delay_ms（平均时延，与实验三一致）。
    PPO/SAC 为 best checkpoint 贪婪评估补录值，TD3 为训练全程平均
    （data 文件内 delay_mean_ms 字段直接使用，不再重新评估）。

输出（文件名不含时间戳，与实验二/三一致）：
  - figure_load_7algo.png        负载柱状图（7 算法，含 std 误差棒 + 数值标注）
  - figure_delay_7algo.png       时延柱状图（7 算法平均时延，含 std 误差棒）
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

# ---------------------------------------------------------------------------
# 统一实验配置
# ---------------------------------------------------------------------------
SEEDS = ("42", "123", "2024")

# 图例配色（7 算法，稳定配色）
ALGO_COLORS = {
    "random_feasible":      "#7f7f7f",  # 灰（基线）
    "greedy":               "#2ca02c",  # 绿
    "proportional_fair":    "#ff7f0e",  # 橙
    "genetic_algorithm":    "#9467bd",  # 紫
    "PPO":                  "#1f77b4",  # 蓝
    "SAC":                  "#d62728",  # 红
    "TD3":                  "#8c564b",  # 棕
}

# 展示名（图例/横轴用）
DISPLAY_NAMES = {
    "random_feasible":      "Random-Feasible",
    "greedy":               "Greedy",
    "proportional_fair":    "Proportional-Fair",
    "genetic_algorithm":    "Genetic-Algorithm",
    "PPO":                  "PPO",
    "SAC":                  "SAC",
    "TD3":                  "TD3",
}

# 算法顺序：先启发式，后 RL
ALGO_ORDER = [
    "random_feasible", "greedy", "proportional_fair", "genetic_algorithm",
    "PPO", "SAC", "TD3",
]

# RL 数据文件映射（负载 + 时延）
RL_LOAD_FILE = "04_native_max_load.json"
RL_DELAY_FILES = {
    "PPO": "05_ppo_native_delay.json",
    "SAC": "06_sac_native_delay.json",
    "TD3": "16_td3_native_delay.json",
}
HEURISTIC_FILE = "20_heuristics.json"


def load_json(name: str):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        print(f"  [MISSING] {name}")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def collect_load(name: str) -> dict[str, float]:
    """读取某算法的 eff_peak per seed（负载）。"""
    if name in ("PPO", "SAC", "TD3"):
        d = load_json(RL_LOAD_FILE)
        if d is None:
            return {}
        entry = d.get("algorithms", {}).get(name, {})
        return {k: float(v) for k, v in entry.get("eval_mean_per_seed", {}).items()}
    d = load_json(HEURISTIC_FILE)
    if d is None:
        return {}
    entry = d.get("algorithms", {}).get(name, {})
    return {k: float(v) for k, v in entry.get("eff_peak_per_seed", {}).items()}


def collect_delay(name: str) -> dict[str, float]:
    """读取某算法的 average_delay per seed（时延）。"""
    if name in RL_DELAY_FILES:
        d = load_json(RL_DELAY_FILES[name])
        if d is None:
            return {}
        out = {}
        for seed, v in d.get("seeds", {}).items():
            m = v.get("delay_mean_ms")
            if m is not None:
                out[seed] = float(m)
        return out
    d = load_json(HEURISTIC_FILE)
    if d is None:
        return {}
    entry = d.get("algorithms", {}).get(name, {})
    return {k: float(v) for k, v in entry.get("avg_delay_per_seed", {}).items()}


def stats(vals: dict[str, float]):
    arr = np.array(list(vals.values()), dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })


def plot_load(load_data: dict[str, dict[str, float]]):
    names, means, stds, colors = [], [], [], []
    for n in ALGO_ORDER:
        if not load_data.get(n):
            print(f"  [SKIP load] {n} 无数据")
            continue
        m, s = stats(load_data[n])
        names.append(n); means.append(m); stds.append(s); colors.append(ALGO_COLORS[n])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.2,
                  yerr=stds, capsize=6,
                  error_kw=dict(elinewidth=1.5, ecolor="black"))

    for xi, (mv, sd) in enumerate(zip(means, stds)):
        ax.text(xi, mv + sd + 0.015, f"{mv:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        ax.text(xi, mv + sd + 0.075, f"±{sd:.3f}", ha="center", va="bottom",
                fontsize=8, color="dimgray")

    ax.set_ylabel("Effective peak load (lower = better)")
    ax.set_xlabel("Algorithm")
    ax.set_title("Heuristic vs DRL — Effective peak load (50 flows, simple set, 3 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[n] for n in names], rotation=20, ha="right")
    top = max(m + s for m, s in zip(means, stds)) + 0.12
    ax.set_ylim(0, top)
    ax.legend(bars, [DISPLAY_NAMES[n] for n in names],
              loc="upper left", ncol=2, framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(HERE, "figure_load_7algo.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_delay(delay_data: dict[str, dict[str, float]]):
    names, means, stds, colors = [], [], [], []
    for n in ALGO_ORDER:
        if not delay_data.get(n):
            print(f"  [SKIP delay] {n} 无数据")
            continue
        m, s = stats(delay_data[n])
        names.append(n); means.append(m); stds.append(s); colors.append(ALGO_COLORS[n])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.2,
           yerr=stds, capsize=6,
           error_kw=dict(elinewidth=1.5, ecolor="black"))

    for xi, (mv, sd) in enumerate(zip(means, stds)):
        ax.text(xi, mv + sd + 0.02, f"{mv:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        ax.text(xi, mv + sd + 0.08, f"±{sd:.3f}", ha="center", va="bottom",
                fontsize=8, color="dimgray")

    ax.set_ylabel("Average delay (ms, lower = better)")
    ax.set_xlabel("Algorithm")
    ax.set_title("Heuristic vs DRL — Average delay (50 flows, simple set, 3 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[n] for n in names], rotation=20, ha="right")
    top = max(m + s for m, s in zip(means, stds)) + 0.5
    ax.set_ylim(0, top)
    ax.legend([DISPLAY_NAMES[n] for n in names], framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(HERE, "figure_delay_7algo.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    setup_style()
    print("=== 横向对比实验（4 启发式 + 3 RL）—— 纯画图（读 JSON）===")
    print(f"统一口径: 50 flows / simple set / seeds={list(SEEDS)}")

    # 1) 收集负载数据
    load_data: dict[str, dict[str, float]] = {}
    for n in ALGO_ORDER:
        load_data[n] = collect_load(n)
        print(f"  [load] {DISPLAY_NAMES[n]:20s} per_seed={load_data[n]}")

    # 2) 收集时延数据
    delay_data: dict[str, dict[str, float]] = {}
    for n in ALGO_ORDER:
        delay_data[n] = collect_delay(n)
        print(f"  [delay] {DISPLAY_NAMES[n]:20s} per_seed={delay_data[n]}")

    # 3) 绘图
    load_png = plot_load(load_data)
    delay_png = plot_delay(delay_data)

    print("\n=== 输出文件 ===")
    print(f"  负载图: {load_png}")
    print(f"  时延图: {delay_png}")

    print("\n=== 汇总（mean ± std）===")
    for n in ALGO_ORDER:
        lm, ls = stats(load_data[n]) if load_data[n] else (float("nan"), float("nan"))
        dm, ds = stats(delay_data[n]) if delay_data[n] else (float("nan"), float("nan"))
        print(f"  {DISPLAY_NAMES[n]:20s} eff_peak={lm:.3f}±{ls:.3f}  "
              f"avg_delay={dm:.3f}±{ds:.3f}")


if __name__ == "__main__":
    main()
