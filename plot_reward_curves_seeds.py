# -*- coding: utf-8 -*-
"""三算法奖励曲线：每个算法一幅子图，子图内画该算法 3 个种子的独立曲线。

数据：各算法种子目录 training_curves.json 的 ep_reward。
  PPO:  checkpoints_ppo_seed{42,123,2024}_0716/   (实际仅 seed42 有数据)
  SAC:  checkpoints_sac_seed{42,123,2024}_0716/
  DDQN: checkpoints_ddqn_seed{42,123,2024}_0716/
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # 每个种子的颜色
SEED_LABELS = ["seed 42", "seed 123", "seed 2024"]


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(p, encoding="gbk"))


def smooth(y, window=10):
    if window <= 1 or len(y) < window:
        return y
    pad_l, pad_r = (window - 1) // 2, window // 2
    yp = np.pad(y, (pad_l, pad_r), mode="reflect")
    return np.convolve(yp, np.ones(window) / window, mode="valid")


ALGORITHMS = [
    {"name": "PPO",  "dirs": ["checkpoints_ppo_seed42_0716",
                                "checkpoints_ppo_seed123_0716",
                                "checkpoints_ppo_seed2024_0716"],
     "color": "#1f77b4"},
    {"name": "SAC",  "dirs": ["checkpoints_sac_seed42_0716",
                                "checkpoints_sac_seed123_0716",
                                "checkpoints_sac_seed2024_0716"],
     "color": "#d62728"},
    {"name": "DDQN", "dirs": ["checkpoints_ddqn_seed42_0716",
                                "checkpoints_ddqn_seed123_0716",
                                "checkpoints_ddqn_seed2024_0716"],
     "color": "#FFC107"},
]


def main(save="reward_curves_by_seed.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, algo in zip(axes, ALGORITHMS):
        seeds = []  # 收集 (ep, rw) 元组
        for i, d in enumerate(algo["dirs"]):
            p = os.path.join(d, "training_curves.json")
            if not os.path.exists(p):
                print(f"[WARN] {algo['name']}: missing {p}")
                continue
            c = load(p)
            ep = np.asarray(c["episode"], float)
            rw = np.asarray(c["ep_reward"], float)
            if len(ep) != len(rw):
                n = min(len(ep), len(rw))
                ep, rw = ep[:n], rw[:n]
            ax.plot(ep, smooth(rw, 10), color=SEED_COLORS[i % 3], lw=1.8,
                    label=SEED_LABELS[i % 3])
            seeds.append((ep, rw))
            print(f"  {algo['name']:5s} {SEED_LABELS[i % 3]:9s}: "
                  f"x=[{ep[0]:.0f},{ep[-1]:.0f}] final={rw[-1]:.2f}")

        # 多种子时叠加均值虚线
        if len(seeds) > 1:
            x0 = max(s[0][0] for s in seeds)
            x1 = min(s[0][-1] for s in seeds)
            xs = np.linspace(x0, x1, 400)
            Y = [np.interp(xs, s[0], s[1]) for s in seeds]
            mean = np.mean(Y, axis=0)
            ax.plot(xs, smooth(mean, 10), color=algo["color"], lw=3.0,
                    linestyle="--", label="mean")

        ax.set_title(f"{algo['name']}  ({len(seeds)} seed"
                     f"{'s' if len(seeds) > 1 else ''})")
        ax.set_xlabel("Training episode")
        if seeds:
            xmax = max(s[0][-1] for s in seeds)
            ax.set_xlim(0, xmax)
        ax.legend(loc="lower right", framealpha=0.9)

    axes[0].set_ylabel("Episode reward  (= −eff_peak_load)")
    fig.suptitle("Reward curves per random seed: PPO / SAC / DDQN (random-order env)",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(save, dpi=300)
    print(f"saved {save}")


if __name__ == "__main__":
    main()
