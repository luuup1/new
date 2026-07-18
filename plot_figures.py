# -*- coding: utf-8 -*-
"""生成论文图 1（带误差棒方法对比柱状图）与图 3（PPO 冷启动训练奖励收敛曲线）。
数据来源：
  - 图1: results_plot_data.json（random 顺序 env, simple 集, 50 流, 3 链路 x 32 slot）
  - 图3: checkpoints_ppo_cold_seed{42,123,2024}/training_curves.json
         （方案 A：冷启动 PPO 多随机种子训练，rollout=10 对齐 eval 网格）
输出：figure1_method_comparison.png / figure3_training_curves.png
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

# 文件由 eval_plot_data.py 在 Windows 默认编码(gbk/cp936)下写出，故按 gbk 读
try:
    DATA = json.load(open("results_plot_data.json", encoding="utf-8"))
except UnicodeDecodeError:
    DATA = json.load(open("results_plot_data.json", encoding="gbk"))
M = DATA["methods"]
D = DATA["derived"]

# ----------------------------------------------------------------------------
# 公共数据
# ----------------------------------------------------------------------------
baseline_mean = M["random_feasible"]["eff_peak_mean"]
baseline_std = M["random_feasible"]["eff_peak_std"]
baseline_per = M["random_feasible"]["per_seed"]

ppo_mean = M["ppo_bc_best"]["eff_peak_mean"]
ppo_std = M["ppo_bc_best"]["eff_peak_std"]
ppo_per = M["ppo_bc_best"]["per_seed"]

teacher_mean = M["teacher_random_minload"]["eff_peak_mean"]
teacher_std = M["teacher_random_minload"]["eff_peak_std"]
teacher_per = M["teacher_random_minload"]["per_seed"]

milp = M["milp"]["eff_peak"]
ppo_cold = M["ppo_cold_random"]["eff_peak"]  # 记忆值，无 std

# ============================================================================
# 图 1：带误差棒方法对比柱状图（headline 图）
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(9, 5.5))

groups = [
    ("Random feasible\n(baseline)", baseline_mean, baseline_std, "#d62728", "baseline"),
    ("PPO (cold,\nno BC)*", ppo_cold, None, "#ff9896", "cold"),
    ("PPO + BC", ppo_mean, ppo_std, "#1f77b4", "ppo"),
    ("Teacher\n(rand. min-load)", teacher_mean, teacher_std, "#7f7f7f", "teacher"),
    ("MILP\nlower bound", milp, None, "#2ca02c", "milp"),
]
labels = [g[0] for g in groups]
means = [g[1] for g in groups]
stds = [g[2] if g[2] is not None else np.nan for g in groups]
colors = [g[3] for g in groups]

x = np.arange(len(groups))
bars = ax1.bar(x, means, color=colors, edgecolor="black", linewidth=1.0,
               yerr=stds, capsize=6, error_kw=dict(elinewidth=1.5, ecolor="black"))

# 标注数值
for xi, (lab, mv, sd) in enumerate(zip(labels, means, stds)):
    if sd is None:
        txt = f"{mv:.3f}"
    else:
        txt = f"{mv:.3f}\n±{sd:.3f}"
    ax1.text(xi, mv + 0.012, txt, ha="center", va="bottom", fontsize=10, fontweight="bold")

# 标注 PPO+BC 相对基线的改进（20 种子均值口径）
imp = D["framing_20seed_mean"]["improvement_vs_baseline_pct"]
gap = D["framing_20seed_mean"]["milp_gap_filled_pct"]
ax1.annotate(
    f"PPO+BC vs baseline:\n−{imp:.1f}% eff_peak\n(fills {gap:.1f}% of\nMILP gap)",
    xy=(2, ppo_mean), xytext=(2.35, 0.62),
    fontsize=10, color="#1f77b4", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", fc="#e8f0fe", ec="#1f77b4", alpha=0.9),
)

ax1.axhline(milp, color="#2ca02c", linestyle=":", linewidth=1.5)
ax1.set_ylabel("Effective peak load (lower = better)")
ax1.set_title("Fig.1  Eff-peak comparison on random-order env (50 flows, simple set)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0.45, 1.12)
ax1.legend(handles=[Patch(facecolor="#d62728", label="Naïve baseline (random feas.)"),
                    Patch(facecolor="#1f77b4", label="PPO + BC (ours)"),
                    Patch(facecolor="#7f7f7f", label="Non-RL teacher"),
                    Patch(facecolor="#2ca02c", label="MILP lower bound")],
           loc="upper right", framealpha=0.9)
ax1.text(0.99, 0.02, "* cold start = memory value, not robustly re-evaluated",
         transform=ax1.transAxes, ha="right", va="bottom", fontsize=8, style="italic", color="#555555")

fig1.savefig("figure1_method_comparison.png", dpi=300)
print("saved figure1_method_comparison.png")


# ============================================================================
# 图 3：PPO 冷启动训练奖励收敛曲线
#        方案 A：3 随机种子 mean ± std；A1 平移为正数（不改变语义）
# ============================================================================
def load_curves(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(path, encoding="gbk"))


def smooth(y, window=5):
    """简单滚动平均，仅用于显示平滑，不影响 std 带。"""
    if window <= 1 or len(y) < window:
        return y
    box = np.ones(window) / window
    return np.convolve(y, box, mode="same")


def plot_ppo_convergence(seed_dirs, save_name="figure3_training_curves.png",
                         smooth_window=5):
    curves = []
    for d in seed_dirs:
        p = os.path.join(d, "training_curves.json")
        if not os.path.exists(p):
            print(f"[WARN] missing {p}, skip this seed")
            continue
        c = load_curves(p)
        xs = np.array(c["episode"], dtype=float)
        ys = np.array(c["ep_reward"], dtype=float)
        curves.append((xs, ys))

    if len(curves) == 0:
        print("[SKIP] no training_curves.json found, skip figure3")
        return

    # rollout=10 → 各种子 x 网格天然一致，以第一个为基准
    x_common = curves[0][0]
    Y = np.stack([cv[1] for cv in curves], axis=0)   # (n_seeds, n_points)
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)

    # A1：平移到正数区（纯可视化；env 奖励本身仍为负，最优策略/数值结论不变）
    offset = -mean.min() + 10.0
    y_pos = mean + offset

    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    ax3.fill_between(x_common, y_pos - std, y_pos + std,
                     color="#1f77b4", alpha=0.25, label="±1 std (3 seeds)")
    ax3.plot(x_common, smooth(y_pos, smooth_window), color="#1f77b4", lw=2,
             label="PPO cold-start (mean, smoothed)")

    ax3.set_xlabel("Training episode")
    ax3.set_ylabel(f"Episode reward (shifted, +{offset:.0f})")
    ax3.set_title("Fig.3  PPO cold-start reward convergence (random-order env, 3 seeds)")
    ax3.legend(loc="lower right", framealpha=0.9)
    fig3.tight_layout()
    fig3.savefig(save_name, dpi=300)
    print(f"saved {save_name}  (n_seeds={len(curves)}, offset=+{offset:.0f})")


# 方案 A 训练产物目录（train_ppo.py --save-dir 指定的 3 个种子）
COLD_SEEDS = [
    "checkpoints_ppo_cold_seed42",
    "checkpoints_ppo_cold_seed123",
    "checkpoints_ppo_cold_seed2024",
]
plot_ppo_convergence(COLD_SEEDS)
print("DONE")
