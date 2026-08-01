# -*- coding: utf-8 -*-
"""三算法奖励曲线对比图（PPO / SAC / DDQN，各 3 种子 mean ± std）。

数据来源：各算法种子目录下的 training_curves.json 中的 ep_reward 字段。
  PPO:  checkpoints_ppo_seed{42,123,2024}_0716/
  SAC:  checkpoints_sac_seed{42,123,2024}_0716/
  DDQN: checkpoints_ddqn_seed{42,123,2024}_0716/

说明：
  - 同一算法内，不同种子 episode 长度/起点可能不同（早停、采样间隔不同），
    因此先线性插值到该算法内部的公共 x 网格，再求 mean/std，避免长度不一致报错。
  - 不同算法之间直接画在同一坐标轴（x = training episode），matplotlib 自动对齐。
  - 奖励 = −eff_peak_load（cost-based reward），数值越大（越接近 0）越好。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def smooth(y, window=5):
    """Edge-aware rolling average (reflect-pad, 输出长度 == len(y))."""
    if window <= 1 or len(y) < window:
        return y
    pad_l = (window - 1) // 2
    pad_r = window // 2
    y_padded = np.pad(y, (pad_l, pad_r), mode="reflect")
    box = np.ones(window) / window
    return np.convolve(y_padded, box, mode="valid")


def read_algo_reward(seed_dirs, smooth_window=50):
    """读取某算法多个种子的 ep_reward，插值到公共网格，返回 (x, mean, std, n_seeds)。"""
    raw = []
    for d in seed_dirs:
        p = os.path.join(d, "training_curves.json")
        if not os.path.exists(p):
            print(f"[WARN] missing {p}, skip this seed")
            continue
        c = load_curves(p)
        xs = np.asarray(c["episode"], dtype=float)
        ys = np.asarray(c["ep_reward"], dtype=float)
        if len(xs) != len(ys):
            n = min(len(xs), len(ys))
            xs, ys = xs[:n], ys[:n]
        raw.append((xs, ys))

    if not raw:
        return None, None, None

    # 公共网格：从所有种子中最大的起点，到最小终点，点数取最短种子的点数
    x_start = max(r[0][0] for r in raw)
    x_end = min(r[0][-1] for r in raw)
    n_min = min(len(r[1]) for r in raw)
    x_common = np.linspace(x_start, x_end, n_min)

    Y = np.stack([np.interp(x_common, r[0], r[1]) for r in raw], axis=0)  # (n_seeds, n_points)
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    if smooth_window and smooth_window > 1:
        mean = smooth(mean, smooth_window)
    return x_common, mean, std, len(raw)


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
    },
]


def main(save_name="reward_curves_comparison.png"):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    results = []  # (name, x, mean, std, n_seeds)
    for algo in ALGORITHMS:
        x, mean, std, n_seeds = read_algo_reward(algo["seed_dirs"], algo["smooth_window"])
        if x is None:
            print(f"[SKIP] {algo['name']}: no data")
            continue
        ax.fill_between(x, mean - std, mean + std,
                        color=algo["color"], alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=algo["color"], lw=2.4,
                label=f"{algo['name']} (mean ± std, {n_seeds} seed{'s' if n_seeds > 1 else ''})")
        results.append((algo["name"], x, mean, std, n_seeds))
        print(f"  {algo['name']:5s}: x=[{x[0]:.0f}, {x[-1]:.0f}]  "
              f"final reward={mean[-1]:.2f}  std={std[-1]:.2f}  seeds={n_seeds}")

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward  (= −eff_peak_load)")
    ax.set_title("Reward curve comparison: PPO vs SAC vs DDQN (random-order env, 3 seeds each)")
    ax.legend(loc="lower right", framealpha=0.9)
    if results:
        ax.set_xlim(0, max(r[1][-1] for r in results))

    fig.tight_layout()
    fig.savefig(save_name, dpi=300)
    print(f"saved {save_name}  (algorithms: {[r[0] for r in results]})")


if __name__ == "__main__":
    main()
