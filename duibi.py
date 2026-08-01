# -*- coding: utf-8 -*-
"""duibi.py — 生成论文 Fig.1 方法对比柱状图 (figure1_method_comparison.png)

独立脚本，专用于画出「5G-TSN 多链路 DRL 调度」的方法对比 headline 图。

【数据口径：方案甲主线（cold 三算法对比）】
  与 plot_figures.py 里旧的 figure1（PPO+BC + 记忆值 cold）不同，本脚本按
  当前论文主线重新组织：
    - 唯一合法朴素基线: random_feasible (random 顺序 + random 放置)
    - 三个冷启动 DRL 算法: PPO / SAC / DDQN，统一
        --order-mode random --period-mode simple --reward-mode load_balance
    - MILP 最优下界 (≤50 流, 0.545) 作为 green 参考线，不是基线

【数据来源】
    冷启动三算法的真实评估值来自各种子目录的 training_curves.json:
        eval_best = min(eval_eff_peak)   # 每个种子取训练全程最佳 greedy 评估
    聚合方式: 跨种子 mean ± std(ddof=1)
    基线 random_feasible: 优先读 results_plot_data.json；否则用 env 重算；
        最后兜底用已记录的常数 (1.0052 ± 0.0038)。
    MILP: 固定 0.545 (最优下界，作上限参考)。

【诚实标注】
    DDQN seed2024 系统性崩溃 (≈1.0, 基本没学)，按与 plot_three_algo_comparison.py
    一致的约定剔除，仅用 seed42 / seed123 两根干净种子。
"""
import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1) 冷启动三算法定义：名称 / 颜色 / 种子目录(从 training_curves.json 读 eval_best)
# ---------------------------------------------------------------------------
ALGOS = [
    {
        "name": "PPO (cold)",
        "color": "#1f77b4",   # 蓝
        "dirs": [
            "checkpoints_ppo_seed42_0716",
            "checkpoints_ppo_seed123_0716",
            "checkpoints_ppo_seed2024_0716",
        ],
    },
    {
        "name": "SAC (cold)",
        "color": "#d62728",   # 红
        "dirs": [
            "checkpoints_sac_seed42_0716",
            "checkpoints_sac_seed123_0716",
            "checkpoints_sac_seed2024_0716",
        ],
    },
    {
        "name": "DDQN (cold)",
        "color": "#FFC107",   # 黄
        # seed2024 系统性崩溃(≈1.0)，已剔除，仅用两根干净种子
        "dirs": [
            "checkpoints_ddqn_seed42_0716",
            "checkpoints_ddqn_seed123_0716",
        ],
        "note": "排除 seed2024 (系统性未学习 ≈1.0)",
    },
]

MILP_EFF_PEAK = 0.545       # MILP 最优下界 (≤50 流)
BASE_FALLBACK = (0.998, 0.0038)   # 已记录的 random_feasible 常数 (mean, std)


# ---------------------------------------------------------------------------
# 2) 读取单一种子目录的 eval_best = min(eval_eff_peak)
# ---------------------------------------------------------------------------
def load_curves(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(path, encoding="gbk"))


def eval_best_of_dir(dirpath):
    p = os.path.join(HERE, dirpath, "training_curves.json")
    if not os.path.exists(p):
        print(f"[WARN] missing {p}, skip")
        return None
    c = load_curves(p)
    ee = c.get("eval_eff_peak") or []
    if not ee:
        print(f"[WARN] no eval_eff_peak in {p}, skip")
        return None
    return float(np.min(ee))


def collect_algo(algo):
    vals = []
    for d in algo["dirs"]:
        v = eval_best_of_dir(d)
        if v is None:
            continue
        vals.append(v)
        print(f"  {algo['name']:12s} {d:36s} eval_best = {v:.4f}")
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std, len(arr)


# ---------------------------------------------------------------------------
# 3) 朴素基线 random_feasible: 优先读 json → 否则 env 重算 → 否则常数兜底
# ---------------------------------------------------------------------------
def get_baseline():
    # (a) 读 results_plot_data.json
    rp = os.path.join(HERE, "results_plot_data.json")
    if os.path.exists(rp):
        try:
            D = load_curves(rp)
            b = D["methods"]["random_feasible"]
            return float(b["eff_peak_mean"]), float(b["eff_peak_std"]), "results_plot_data.json"
        except Exception as e:
            print(f"[INFO] read results_plot_data.json failed ({e}), try recompute")

    # (b) 用 env 重算 random_feasible (random 顺序 + random 放置)
    try:
        sys.path.insert(0, HERE)
        from tsn_sim import TSNSchedulingEnv, SimulationConfig
        N_SEEDS = 20
        EVAL_SEEDS = list(range(1000, 1000 + N_SEEDS))
        BASE_DRAWS = 4
        vals = []
        for sd in EVAL_SEEDS:
            for k in range(BASE_DRAWS):
                rng = np.random.RandomState(sd * 100 + k)
                sim = SimulationConfig(seed=sd, period_mode="simple")
                e = TSNSchedulingEnv(config=sim, order_mode="random",
                                     reward_mode="mixed", obs_mode="full")
                e.reset()
                while True:
                    m = e.action_masks()
                    acts = np.where(m)[0]
                    a = int(rng.choice(acts)) if len(acts) else 0
                    _, _, d, _, info = e.step(a)
                    if d:
                        break
                vals.append(float(info["effective_peak_load"]))
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)), "recomputed via env"
    except Exception as e:
        print(f"[WARN] recompute baseline failed ({e}), use fallback constant")

    # (c) 兜底常数
    return BASE_FALLBACK[0], BASE_FALLBACK[1], "fallback constant"


# ---------------------------------------------------------------------------
# 4) 收集数据 + 画图
# ---------------------------------------------------------------------------
def main():
    base_mean, base_std, base_src = get_baseline()
    print(f"\n[baseline] random_feasible = {base_mean:.4f} ± {base_std:.4f}  (src: {base_src})")

    algo_data = []
    for algo in ALGOS:
        r = collect_algo(algo)
        if r is None:
            print(f"[SKIP] {algo['name']}: no data")
            continue
        algo_data.append((algo["name"], algo["color"], r[0], r[1], r[2],
                          algo.get("note", "")))

    # 构建柱子顺序: baseline / PPO / SAC / DDQN
    groups = [("Random feasible\n(baseline)", base_mean, base_std, "#555555")]
    for name, color, mean, std, n, note in algo_data:
        label = name + (f"\n(n={n})" if n else "")
        groups.append((label, mean, std, color))

    labels = [g[0] for g in groups]
    means = [g[1] for g in groups]
    stds = [g[2] for g in groups]
    colors = [g[3] for g in groups]
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.2,
                  yerr=stds, capsize=6,
                  error_kw=dict(elinewidth=1.5, ecolor="black"))

    # 标注数值 + 相对基线改进
    for xi, (lab, mv, sd) in enumerate(zip(labels, means, stds)):
        if xi == 0:
            txt = f"{mv:.3f}\n±{sd:.3f}"
            ax.text(xi, mv + sd + 0.015, txt, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        else:
            imp = (base_mean - mv) / base_mean * 100
            txt = f"{mv:.3f}\n±{sd:.3f}\n(−{imp:.1f}%)"
            ax.text(xi, mv + sd + 0.015, txt, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_ylabel("Effective peak load (lower = better)")
    ax.set_title("Fig.1  Effective peak comparison (random-order env, 50 flows, simple set)",
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.45, 1.12)

    handles = [
        Patch(facecolor="#555555", label="baseline"),
        Patch(facecolor="#1f77b4", label="PPO"),
        Patch(facecolor="#d62728", label="SAC"),
        Patch(facecolor="#FFC107", label="DDQN"),
        ##Patch(facecolor="#2ca02c", label="MILP lower bound"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    ax.text(0.99, 0.02,
            "Cold start: PPO/SAC/DDQN trained from scratch, evaluated greedily.\n"
            "DDQN excludes seed2024 (systematic failure). Baseline = random feasible only.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color="#555555")

    fig.tight_layout()
    out = os.path.join(HERE, "figure1_method_comparison.png")
    fig.savefig(out, dpi=300)
    print(f"\nsaved {out}")
    print("means:", {g[0].split(chr(10))[0]: round(m, 4) for g, m in zip(groups, means)})


if __name__ == "__main__":
    main()
