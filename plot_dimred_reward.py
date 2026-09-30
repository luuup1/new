# -*- coding: utf-8 -*-
"""绘制降维(dimred)奖励曲线对比图（仅 PPO / SAC / TD3，一条指令出图）。

本脚本直接读取 data/ 下的 3 个 dimred 原始 *_reward.json 文件，在脚本内部自动完成：
  1. episode 采样间隔统一为 1：对 10ep 采样的用线性插值补到 1ep；
     对已经是 1ep 的直接使用。
  2. 统一截断到 800 轮（MAX_EPISODE=800）。
  3. 每组 3 个 seed 对齐到公共长度，计算 mean / std。
  4. 绘图：3 组同图，统一 y 轴，轻微滑动平均去噪，图例一列。

用法（一条指令）：
  RL/bin/python plot_dimred_reward.py                 # 默认输出 reward_dimred_comparison.png（自动平移为正数）
  RL/bin/python plot_dimred_reward.py --output xxx.png  # 指定输出文件名
  RL/bin/python plot_dimred_reward.py --no-smooth       # 不做滑动平均
  RL/bin/python plot_dimred_reward.py --window 15       # 调整滑动平均窗口（默认 7）
  RL/bin/python plot_dimred_reward.py --ylim=0,340      # 手动指定 y 轴范围（平移后的坐标）
  RL/bin/python plot_dimred_reward.py --shift 300       # 固定平移量
  RL/bin/python plot_dimred_reward.py --shift 0         # 不平移（保持原始负值）

说明：原始 reward = 累积 −effective_peak_load，恒为负；脚本默认自动整体上移一个
常数，使曲线最低点落在 1.0，便于阅读（形状与相对高低完全不变，std 不受影响）。
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体
for name in ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Songti SC", "Arial Unicode MS"]:
    if name in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "axes.linewidth": 1.0, "axes.edgecolor": "#333333",
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# 统一截断轮数
MAX_EPISODE = 800

# 3 组：(组id, 源文件, 算法) —— 仅 dimred
GROUPS = [
    ("08_PPO_dimred",  "8_ppo_dimred_reward.json",   "PPO"),
    ("09_SAC_dimred",  "9_sac_dimred_reward.json",   "SAC"),
    ("17_TD3_dimred",  "17_td3_dimred_reward.json",  "TD3"),
]

# 图例友好名 + 颜色（沿用统一配色：PPO 蓝 / SAC 红 / TD3 橙）
DISPLAY = {
    "08_PPO_dimred": ("PPO (dimred)", "#1f77b4"),
    "09_SAC_dimred": ("SAC (dimred)", "#d62728"),
    "17_TD3_dimred": ("TD3 (dimred)", "#ff7f0e"),
}


def load_group(gid):
    """读一个原始 reward 文件，返回 {seed: (episode_1d, reward_1d)}，内部完成插值+截断。"""
    src = dict((g[0], g[1]) for g in GROUPS)[gid]
    with open(os.path.join(DATA, src), encoding="utf-8") as f:
        d = json.load(f)

    seeds = {}
    for s in ["42", "123", "2024"]:
        v = d["seeds"].get(s, {})
        ep = v.get("episode", [])
        r = v.get("ep_reward", [])
        if not ep or not r or len(ep) != len(r):
            continue
        res = align_seed(ep, r)
        if res is not None:
            grid, rr = res
            n = min(len(grid), MAX_EPISODE)
            seeds[s] = (grid[:n], rr[:n])
    return seeds


def align_seed(ep, r):
    """把 (episode, reward) 对齐到 1ep 间隔、从 ep=1 开始的整数网格。

    步长 > 1 时用 np.interp 线性插值到 1ep；步长 == 1 时直接使用。
    """
    ep = np.array(ep, dtype=float)
    r = np.array(r, dtype=float)
    if len(ep) < 2:
        return None
    start = int(round(ep[0]))
    end = int(round(ep[-1]))
    grid = np.arange(1, end + 1, dtype=float)
    if len(ep) == end - start + 1 and np.allclose(np.diff(ep), 1.0):
        rr = r
        if start > 1:
            pad = np.full(start - 1, r[0])
            rr = np.concatenate([pad, r])
        return grid, rr
    rr = np.interp(grid, ep, r)
    return grid, rr


def light_smooth(y, window=7):
    """轻微滑动平均去噪。window<=1 表示不平滑。"""
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    out = np.convolve(y, kernel, mode="same")
    half = window // 2
    for k in range(half):
        out[k] = y[:k + half + 1].mean()
        out[-k - 1] = y[-k - half - 1:].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="exam1_reward.png")
    ap.add_argument("--no-smooth", action="store_true", help="不做任何平滑")
    ap.add_argument("--window", type=int, default=7, help="滑动平均窗口（默认7）")
    ap.add_argument("--ylim", default=None, help='y 轴范围，如 "0,340"（平移后的坐标）')
    ap.add_argument("--shift", default="auto",
                    help='纵坐标平移：auto=自动平移到正数（默认）；数字=固定平移量；0=不平移')
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # 先收集各组数据（不绘制），以便计算自动平移量
    collected = []
    for gid, _, algo in GROUPS:
        seeds = load_group(gid)
        if not seeds:
            print(f"[SKIP] {gid}: 无有效数据")
            continue
        common = min(len(v[1]) for v in seeds.values())
        Y = np.vstack([np.array(v[1][:common]) for v in seeds.values()])
        x = np.array(seeds["42"][0][:common], dtype=float)
        mean = Y.mean(axis=0)
        std = Y.std(axis=0)

        w = 0 if args.no_smooth else args.window
        mean_s = light_smooth(mean, w)
        std_s = light_smooth(std, w)
        collected.append((gid, x, mean_s, std_s))
        print(f"[OK] {gid}: n_seeds={len(seeds)}  ep={common}")

    if not collected:
        print("无有效数据，无法绘图。")
        return

    # 计算纵坐标平移量
    if str(args.shift).lower() in ("0", "none", "off"):
        shift = 0.0
    elif str(args.shift).lower() == "auto":
        global_min = min((m - s).min() for _, _, m, s in collected)
        shift = -global_min + 1.0  # 让曲线最低点平移到 1.0
        print(f"[shift] 自动平移 {shift:.2f}（原最低点 {global_min:.2f} -> 1.00）")
    else:
        shift = float(args.shift)

    # 绘制（整体上移 shift，std 不受平移影响）
    plotted = []
    for gid, x, mean_s, std_s in collected:
        mean_shifted = mean_s + shift
        label, color = DISPLAY[gid]
        ax.plot(x, mean_shifted, color=color, lw=2.6, label=label, alpha=0.95)
        ax.fill_between(x, mean_shifted - std_s, mean_shifted + std_s,
                        color=color, alpha=0.12, linewidth=0)
        plotted.append((x, mean_shifted, std_s))

    # 平移前的零奖励水平（虚线参考）
    ax.axhline(shift, color="black", linewidth=0.8, alpha=0.4, linestyle=":", zorder=1)

    ax.set_xlabel("episode (训练轮数)", fontsize=13)
    if shift != 0.0:
        ax.set_ylabel(f"ep_reward  (平移 +{shift:.1f})", fontsize=13)
    else:
        ax.set_ylabel("ep_reward  (累积 −effective_peak_load)", fontsize=13)
    ax.set_xlim(0, MAX_EPISODE)

    if args.ylim:
        lo, hi = [float(v) for v in args.ylim.split(",")]
        ax.set_ylim(lo, hi)
    else:
        all_vals = [v for _, m, s in plotted for v in (m + s)]
        all_vals += [v for _, m, s in plotted for v in (m - s)]
        lo, hi = min(all_vals), max(all_vals)
        pad = (hi - lo) * 0.05
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_title("奖励收敛曲线对比：dimred（PPO / SAC / TD3）", fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="lower right", ncol=1, framealpha=0.9, edgecolor="#999999",
              fancybox=True, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
