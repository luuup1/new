# -*- coding: utf-8 -*-
"""生成论文图 3（奖励收敛）与图 4（eff_peak 收敛）。

图 1（方法对比柱状图）由 duibi.py 单独生成 -> figure1_method_comparison.png。

数据来源：各算法种子目录的 training_curves.json
        PPO:   checkpoints_ppo_cold_seed{42,123,2024}/
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

# 图 1（方法对比柱状图）由 duibi.py 单独生成 -> figure1_method_comparison.png
# 本脚本只负责图 3（奖励收敛）与图 4（eff_peak 收敛），不再产出 BC 旧口径图。


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
    不同算法/种子的 episode 长度与起点可能不同（早停、采样间隔不同），
    因此先插值到公共 x 网格，再求 mean/std，避免长度不一致报错。
    """
    raw = []
    for d in seed_dirs:
        p = os.path.join(d, "training_curves.json")
        if not os.path.exists(p):
            print(f"[WARN] missing {p}, skip this seed")
            continue
        c = load_curves(p)
        xs = np.array(c["episode"], dtype=float)
        ys = np.array(c["ep_reward"], dtype=float)
        if len(xs) != len(ys):
            n = min(len(xs), len(ys))
            xs, ys = xs[:n], ys[:n]
        raw.append((xs, ys))

    if len(raw) == 0:
        return None, None

    # 公共网格：从最大起点到最小终点，点数取最短种子的点数
    x_start = max(r[0][0] for r in raw)
    x_end = min(r[0][-1] for r in raw)
    n_min = min(len(r[1]) for r in raw)
    x_common = np.linspace(x_start, x_end, n_min)
    Y = np.stack([np.interp(x_common, r[0], r[1]) for r in raw], axis=0)
    return x_common, Y


# ---- 算法定义：名称 / 种子目录 / 颜色 / 平滑窗口 ----
ALGORITHMS = [
    {
        "name": "PPO",
        "seed_dirs": [
            "checkpoints_ppo_cold_seed42",
            "checkpoints_ppo_cold_seed123",
            "checkpoints_ppo_cold_seed2024",
        ],
        "color": "#1f77b4",   # 蓝
        "smooth_window": 50,
        "eval_step": 50,
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


def plot_multi_algorithm_convergence(algorithms, save_name="figure3_reward_convergence.png"):
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

        if len(xs) != len(ys):
            n = min(len(xs), len(ys))
            xs, ys = xs[:n], ys[:n]

        curves.append((xs, ys))

    if len(curves) == 0:
        return None, None

    # 不同种子长度可能不同（早停/采样间隔），插值到公共网格对齐
    x_start = max(cv[0][0] for cv in curves)
    x_end = min(cv[0][-1] for cv in curves)
    n_min = min(len(cv[1]) for cv in curves)
    x_common = np.linspace(x_start, x_end, n_min)
    Y = np.stack([np.interp(x_common, cv[0], cv[1]) for cv in curves], axis=0)
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

