# -*- coding: utf-8 -*-
"""生成论文图 1（方法对比柱状图）与图 3（三算法训练收敛曲线）。

数据来源：
  - 图1: results_plot_data.json（random 顺序 env, simple 集, 50 流）
  - 图3: 各算法种子目录的 training_curves.json
        PPO:   checkpoints_ppo_seed{42,123,2024}_0716/
        SAC:   checkpoints_sac_seed{42,123,2024}_0716/
        DDQN:  checkpoints_ddqn_seed{42,123,2024}_0716/

颜色约定：PPO=蓝(#1f77b4), SAC=红(#d62728), DDQN=黄(#FFC107)
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

fig1.savefig("method_comparison.png", dpi=300)
print("saved method_comparison.png")


# ============================================================================
# 图 3：三算法训练收敛曲线（PPO / SAC / DDQN，各 3 种子 mean ± std）
#        颜色约定：PPO=蓝  SAC=红  DDQN=黄
# ============================================================================
def load_curves(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(path, encoding="gbk"))


def smooth(y, window=5):
    """Edge-aware rolling average — reflect-pads edges to avoid zero-fill artifacts."""
    if window <= 1 or len(y) < window:
        return y
    # Asymmetric padding for both odd and even windows:
    # pad_left = (window-1)//2, pad_right = window//2
    # This ensures convolve 'valid' output length == len(y)
    pad_l = (window - 1) // 2
    pad_r = window // 2
    y_padded = np.pad(y, (pad_l, pad_r), mode="reflect")
    box = np.ones(window) / window
    return np.convolve(y_padded, box, mode="valid")


def read_algorithm_seeds(seed_dirs):
    """从多个种子目录读取 ep_reward，返回 (x_array, Y_matrix)。
    
    Y_matrix shape = (n_seeds, n_points)。
    不同算法的数据密度不同（PPO 每 rollout 一个点，SAC/DDQN 每 episode 一个点），
    返回各自的自然 x 网格，画图时 matplotlib 自动处理。
    """
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
        return None, None

    x_common = curves[0][0]
    Y = np.stack([cv[1] for cv in curves], axis=0)  # (n_seeds, n_points)
    return x_common, Y


# ---- 算法定义：名称 / 种子目录 / 颜色 / 平滑窗口 ----
ALGORITHMS = [
    {
        "name": "PPO",
        "seed_dirs": [
            "checkpoints_ppo_seed42_0716",
            "checkpoints_ppo_seed123_0716",
            "checkpoints_ppo_seed2024_0716",
        ],
        "color": "#1f77b4",   # 蓝
        "smooth_window": 50,
        "eval_step": 200,     # PPO旧版rollout=8 → LCM(8,50)=200；重跑rollout=10后改50
    },
    {
        "name": "SAC",
        "seed_dirs": [
            "checkpoints_sac_seed42_0716",
            "checkpoints_sac_seed123_0716",
            "checkpoints_sac_seed2024_0716",
        ],
        "color": "#d62728",   # 红
        "smooth_window": 50,
        "eval_step": 50,
    },
    {
        "name": "DDQN",
        "seed_dirs": [
            "checkpoints_ddqn_seed42_0716",
            "checkpoints_ddqn_seed123_0716",
            "checkpoints_ddqn_seed2024_0716",
        ],
        "color": "#FFC107",   # 黄
        "smooth_window": 50,
        "eval_step": 50,
    },
]


def plot_multi_algorithm_convergence(algorithms, save_name="figure3.png"):
    """三算法同图收敛曲线，各算法 3 种子 mean±std，正数平移。"""
    
    # 先收集所有有效算法的数据，算全局 offset
    algo_data = []
    global_min = float("inf")
    
    for algo in algorithms:
        x, Y = read_algorithm_seeds(algo["seed_dirs"])
        if x is None:
            print(f"[SKIP] {algo['name']}: no data found")
            continue
        mean_raw = Y.mean(axis=0)
        std_raw  = Y.std(axis=0)
        algo_min = Y.min()
        global_min = min(global_min, algo_min)
        algo_data.append({
            "name": algo["name"],
            "color": algo["color"],
            "smooth_window": algo["smooth_window"],
            "x": x,
            "mean_raw": mean_raw,
            "std_raw": std_raw,
        })

    if len(algo_data) == 0:
        print("[SKIP] no algorithm data found, skip figure3")
        return

    # 全局平移到正数区（所有算法共享同一个 offset，保证可比性）
    offset = -global_min + 5.0
    print(f"  global_min={global_min:.1f}  offset=+{offset:.1f}")

    fig3, ax3 = plt.subplots(figsize=(9, 5.5))

    for d in algo_data:
        mean = d["mean_raw"] + offset
        std  = d["std_raw"]
        mean_sm = smooth(mean, d["smooth_window"])

        # std 阴影带
        ax3.fill_between(d["x"], mean - std, mean + std,
                         color=d["color"], alpha=0.15)
        # 平滑主线
        ax3.plot(d["x"], mean_sm, color=d["color"], lw=2.2,
                 label=f"{d['name']} (mean, smoothed)")

    ax3.set_xlabel("Training episode", fontsize=13)
    ax3.set_ylabel("Episode reward", fontsize=13)
    ax3.set_title("Fig.3  Training reward convergence (random-order env, 3 seeds per algorithm)",
                  fontsize=14)
    ax3.legend(loc="lower right", framealpha=0.9)
    # x轴范围自动适配数据（支持smoke test短训练和正式长训练）
    max_ep = max(d["x"][-1] for d in algo_data)
    ax3.set_xlim(0, max_ep)

    # 底部小字注释
    ax3.text(0.99, 0.02,
             "Reward = −eff_peak_load (cost-based, shifted for readability)",
             transform=ax3.transAxes, ha="right", va="bottom",
             fontsize=8, style="italic", color="#666666")

    fig3.tight_layout()
    fig3.savefig(save_name, dpi=300)
    print(f"saved {save_name}  (n_algos={len(algo_data)}, offset=+{offset:.1f})")


plot_multi_algorithm_convergence(ALGORITHMS)


# ============================================================================
# 图 4：三算法 eff_peak 收敛曲线（跨算法直接可比，越低越好）
# ============================================================================
def read_algorithm_seeds_field(seed_dirs, field="ep_eff_peak", eval_step=None):
    """从多个种子目录读取指定字段，返回 (x_array, Y_matrix)。

    eval_step: 当读取 eval 字段时，eval 不是逐 episode 记录而是逐评估周期记录，
    需要用 eval_step 重建 x 轴。例如 eval_step=50 → x=[50,100,150,...]。
    """
    curves = []
    for d in seed_dirs:
        p = os.path.join(d, "training_curves.json")
        if not os.path.exists(p):
            print(f"[WARN] missing {p}, skip this seed")
            continue
        c = load_curves(p)
        if field not in c:
            print(f"[WARN] field '{field}' not in {p}, skip")
            continue
        ys = np.array(c[field], dtype=float)

        # 重建 x 轴
        if eval_step is not None:
            # eval 字段：x = [eval_step, 2*eval_step, ..., n*eval_step]
            n_evals = len(ys)
            xs = np.array([eval_step * i for i in range(1, n_evals + 1)], dtype=float)
        else:
            # 逐 episode 字段：直接用 episode 数组
            xs = np.array(c["episode"], dtype=float)

        curves.append((xs, ys))

    if len(curves) == 0:
        return None, None

    # eval 字段：各种子 eval 点数可能不同（早停/Ctrl+C），取最短的公共长度
    if eval_step is not None:
        min_len = min(len(cv[1]) for cv in curves)
        curves = [(cv[0][:min_len], cv[1][:min_len]) for cv in curves]

    x_common = curves[0][0]
    Y = np.stack([cv[1] for cv in curves], axis=0)
    return x_common, Y


def plot_eff_peak_convergence(algorithms, save_name="figure4_eff_peak_convergence.png"):
    """三算法 eff_peak 收敛曲线，y 轴统一为 effective peak load，越低越好。

    每个算法画两条线：
    - 浅色细线 = ep_eff_peak（训练中，带探索噪声，smoothed）
    - 深色粗线 = eval_eff_peak（评估时，贪婪策略，更稳定的收敛信号）
    """

    fig4, ax4 = plt.subplots(figsize=(9, 5.5))

    # 参考线
    ax4.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7,
                label="Random baseline (1.0)")
    ax4.axhline(0.545, color="#2ca02c", linestyle=":", linewidth=1.2, alpha=0.7,
                label="MILP optimal (0.545)")

    for algo in algorithms:
        # --- eval_eff_peak: 贪婪评估，更稳定 ---
        eval_step = algo.get("eval_step", 50)
        x_eval, Y_eval = read_algorithm_seeds_field(algo["seed_dirs"], "eval_eff_peak",
                                                     eval_step=eval_step)
        if x_eval is not None:
            mean_eval = Y_eval.mean(axis=0)
            std_eval  = Y_eval.std(axis=0) if Y_eval.shape[0] > 1 else None
            mean_eval_sm = smooth(mean_eval, min(algo["smooth_window"], len(mean_eval)))
            ax4.plot(x_eval, mean_eval_sm, color=algo["color"], lw=2.8,
                     label=f"{algo['name']} eval (greedy)")
            if std_eval is not None:
                ax4.fill_between(x_eval, mean_eval - std_eval, mean_eval + std_eval,
                                 color=algo["color"], alpha=0.12)

        # --- ep_eff_peak: 训练中，带噪声（浅色背景）---
        x_ep, Y_ep = read_algorithm_seeds_field(algo["seed_dirs"], "ep_eff_peak")
        if x_ep is not None:
            mean_ep = Y_ep.mean(axis=0)
            mean_ep_sm = smooth(mean_ep, min(algo["smooth_window"], len(mean_ep)))
            ax4.plot(x_ep, mean_ep_sm, color=algo["color"], lw=1.0, alpha=0.4,
                     linestyle="--")

    ax4.set_xlabel("Training episode", fontsize=13)
    ax4.set_ylabel("Effective peak load (lower = better)", fontsize=13)
    ax4.set_title("Fig.4  Eff-peak convergence (random-order env, greedy eval + train trace)",
                  fontsize=14)
    ax4.legend(loc="upper right", framealpha=0.9)

    # x轴范围
    all_xmax = []
    for algo in algorithms:
        x_eval, _ = read_algorithm_seeds_field(algo["seed_dirs"], "eval_eff_peak")
        if x_eval is not None:
            all_xmax.append(x_eval[-1])
        x_ep, _ = read_algorithm_seeds_field(algo["seed_dirs"], "ep_eff_peak")
        if x_ep is not None:
            all_xmax.append(x_ep[-1])
    if all_xmax:
        ax4.set_xlim(0, max(all_xmax))
    ax4.set_ylim(0.45, 1.08)

    # 注释：实线=eval, 虚线=train
    ax4.text(0.99, 0.02,
             "Solid = greedy eval (stable)  |  Dashed = train (with exploration noise)",
             transform=ax4.transAxes, ha="right", va="bottom",
             fontsize=8, style="italic", color="#666666")

    fig4.tight_layout()
    fig4.savefig(save_name, dpi=300)
    print(f"saved {save_name}  (n_algos={len(algorithms)})")





plot_eff_peak_convergence(ALGORITHMS)
print("DONE")

