# -*- coding: utf-8 -*-
"""三算法对比柱状图（只取成功 checkpoint，基于真实 eval_best）。

数据来源：各算法成功种子目录的 training_curves.json -> eval_eff_peak 最小值(eval_best)
  - SAC  : checkpoints_sac_seed{42,123,2024}_0716   (3 种子全成功)
  - DDQN : checkpoints_ddqn_seed{42,123}_0716       (seed2024 失败已剔除)
  - PPO+BC: checkpoints_ppo_simple_bc2 / _seed123 / checkpoints_ppo_simple_bc

参考线：random feasible baseline (1.00)，来自 results_plot_data.json
"""
import os
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


def load_curves(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(path, encoding="gbk"))


def eval_best(dirpath):
    p = os.path.join(dirpath, "training_curves.json")
    if not os.path.exists(p):
        return None
    c = load_curves(p)
    ee = c.get("eval_eff_peak") or []
    if not ee:
        return None
    return float(np.min(ee))


# ---- 成功种子目录定义 ----
ALGOS = [
    {
        "name": "SAC",
        "color": "#d62728",
        "dirs": [
            "checkpoints_sac_seed42_0716",
            "checkpoints_sac_seed123_0716",
            "checkpoints_sac_seed2024_0716",
        ],
    },
    {
        "name": "DDQN",
        "color": "#FFC107",
        "dirs": [
            "checkpoints_ddqn_seed42_0716",
            "checkpoints_ddqn_seed123_0716",
        ],
    },
    {
        "name": "PPO+BC",
        "color": "#1f77b4",
        "dirs": [
            "checkpoints_ppo_simple_bc2",
            "checkpoints_ppo_simple_bc_seed123",
            "checkpoints_ppo_simple_bc",
        ],
    },
]

# ---- 收集每个算法的 eval_best 列表 ----
per_algo = {}
for algo in ALGOS:
    vals = []
    for d in algo["dirs"]:
        v = eval_best(d)
        if v is None:
            print(f"[WARN] no eval data in {d}, skip")
            continue
        vals.append(v)
        print(f"  {algo['name']:7s} {d:40s} eval_best = {v:.4f}")
    if vals:
        per_algo[algo["name"]] = (algo["color"], vals)

# ---- baseline (固定为 1.0) ----
baseline = 1.0

# ---- 画图 ----
names = list(per_algo.keys())
colors = [per_algo[n][0] for n in names]
means = [np.mean(per_algo[n][1]) for n in names]
stds = [np.std(per_algo[n][1], ddof=1) if len(per_algo[n][1]) > 1 else 0.0 for n in names]
nseeds = [len(per_algo[n][1]) for n in names]

fig, ax = plt.subplots(figsize=(8.5, 5.5))

x = np.arange(len(names))
bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.2,
              yerr=stds, capsize=6,
              error_kw=dict(elinewidth=1.5, ecolor="black"))

# 标注数值
for xi, (mv, sd) in enumerate(zip(means, stds)):
    txt = f"{mv:.3f}\n±{sd:.3f}"
    ax.text(xi, mv + sd + 0.012, txt, ha="center", va="bottom",
            fontsize=10, fontweight="bold")

# baseline 参考线
ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.5,
           label=f"Random feasible baseline ({baseline:.1f})")

# 各算法相对 baseline 的改进百分比
for xi, mv in enumerate(means):
    imp = (baseline - mv) / baseline * 100
    ax.annotate(f"−{imp:.1f}%", xy=(xi, mv), xytext=(xi, mv - 0.06),
                ha="center", va="top", fontsize=9, color=colors[xi],
                fontweight="bold")

ax.set_ylabel("Effective peak load (lower = better)")
ax.set_title("Three DRL algorithms comparison (random-order env, simple set)")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0.45, 1.12)
ax.legend(loc="upper right", framealpha=0.9)

fig.tight_layout()
fig.savefig("figure_three_algo_comparison.png", dpi=300)
print("\nsaved figure_three_algo_comparison.png")
print("means:", {n: round(m, 4) for n, m in zip(names, means)})
