# -*- coding: utf-8 -*-
"""绘制奖励曲线对比图（一条指令直接出图，无需先跑 export）。

本脚本直接读取 data/ 下的 8 个原始 *_reward.json 文件，在脚本内部自动完成：
  1. episode 采样间隔统一为 1：对 10ep 采样的（01_ppo、02_sac seed42）用线性插值补到 1ep；
     对已经是 1ep 的（03_ddqn、08/09/10 dimred、15/17 TD3）直接使用。
  2. 统一截断到 800 轮（MAX_EPISODE=800）：所有曲线都画到 800 轮为止。
  3. 每组 3 个 seed 对齐到公共长度，计算 mean / std。
  4. 绘图：8 组同图，统一 y 轴，轻微滑动平均去噪，图例两列。

用法（一条指令）：
  python plot_reward.py                       # 默认输出 reward_comparison.png
  python plot_reward.py --output 8gai.png    # 指定输出文件名
  python plot_reward.py --no-smooth          # 不做滑动平均
  python plot_reward.py --window 15          # 调整滑动平均窗口（默认 7）
  python plot_reward.py --ylim=-320,0        # 手动指定 y 轴范围（负值用等号）

说明：episode 间隔 10 是因为"训练 2000 轮、每 10 轮采样一次"；要让间隔变成 1，
必须把 reward 也一起线性插值（不能只改 episode 数字，否则横轴刻度语义错误）。
本脚本在内部自动完成插值，原始 data 文件无需改动。
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

# 8 组：(组id, 源文件, 算法, 模式)
GROUPS = [
    ("01_PPO_native",  "01_ppo_native_reward.json",  "PPO",  "native"),
    ("02_SAC_native",  "02_sac_native_reward.json",  "SAC",  "native"),
    ("03_DDQN_native", "03_ddqn_native_reward.json", "DDQN", "native"),
    ("15_TD3_native",  "15_td3_native_reward.json",  "TD3",  "native"),
    ("08_PPO_dimred",  "8_ppo_dimred_reward.json",   "PPO",  "dimred"),
    ("09_SAC_dimred",  "9_sac_dimred_reward.json",   "SAC",  "dimred"),
    ("10_DDQN_dimred", "10_ddqn_dimred_reward.json", "DDQN", "dimred"),
    ("17_TD3_dimred",  "17_td3_dimred_reward.json",  "TD3",  "dimred"),
]

# 图例友好名 + 颜色 + 线型（8 组全部实线：native 饱和色，dimred 同色系浅色）
DISPLAY = {
    "01_PPO_native":  ("PPO  (native)",  "#1f77b4", "-"),
    "02_SAC_native":  ("SAC  (native)",  "#d62728", "-"),
    "03_DDQN_native": ("DDQN (native)",  "#2ca02c", "-"),
    "15_TD3_native":  ("TD3  (native)",  "#ff7f0e", "-"),
    "08_PPO_dimred":  ("PPO  (dimred)",  "#7fb3d5", "-"),
    "09_SAC_dimred":  ("SAC  (dimred)",  "#e57373", "-"),
    "10_DDQN_dimred": ("DDQN (dimred)",  "#7fbf7f", "-"),
    "17_TD3_dimred":  ("TD3  (dimred)",  "#ffbb78", "-"),
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
            # 截断到 MAX_EPISODE
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
        # 已是 1ep 间隔
        rr = r
        if start > 1:
            pad = np.full(start - 1, r[0])
            rr = np.concatenate([pad, r])
        return grid, rr
    # 插值
    rr = np.interp(grid, ep, r)
    return grid, rr

def light_smooth(y, window=7):
    """轻微滑动平均去噪（不做过度平滑）。window<=1 表示不平滑。"""
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
    ap.add_argument("--output", default="reward_comparison.png")
    ap.add_argument("--no-smooth", action="store_true", help="不做任何平滑")
    ap.add_argument("--window", type=int, default=7, help="滑动平均窗口（默认7）")
    ap.add_argument("--ylim", default=None, help='y 轴范围，如 "-320,0"（负值用等号）')
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11, 7))

    plotted = []
    for gid, _, algo, mode in GROUPS:
        seeds = load_group(gid)
        if not seeds:
            print(f"[SKIP] {gid}: 无有效数据")
            continue
        # 公共长度对齐
        common = min(len(v[1]) for v in seeds.values())
        Y = np.vstack([np.array(v[1][:common]) for v in seeds.values()])
        x = np.array(seeds["42"][0][:common], dtype=float)
        mean = Y.mean(axis=0)
        std = Y.std(axis=0)

        label, color, ls = DISPLAY[gid]
        w = 0 if args.no_smooth else args.window
        mean_s = light_smooth(mean, w)
        std_s = light_smooth(std, w)

        ax.plot(x, mean_s, color=color, ls=ls, lw=2.4, label=label, alpha=0.95,
                zorder=3 if mode == "native" else 2)
        ax.fill_between(x, mean_s - std_s, mean_s + std_s, color=color, alpha=0.10, linewidth=0)
        plotted.append((x, mean_s, std_s))
        print(f"[OK] {gid}: n_seeds={len(seeds)}  ep={common}")

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.5, zorder=1)

    ax.set_xlabel("episode (训练轮数)", fontsize=13)
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

    ax.set_title("奖励收敛曲线对比：native vs dimred（四算法）", fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="lower right", ncol=2, framealpha=0.9, edgecolor="#999999",
              fancybox=True, fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output)
    print(f"wrote {args.output}")

if __name__ == "__main__":
    main()

# # -*- coding: utf-8 -*-
# """绘制奖励曲线对比图（一条指令直接出图，无需先跑 export）。

# 本脚本直接读取 data/ 下的 6 个原始 *_reward.json 文件，在脚本内部自动完成：
#   1. episode 采样间隔统一为 1：对 10ep 采样的（01_ppo、02_sac seed42）用线性插值补到 1ep；
#      对已经是 1ep 的（03_ddqn、08/09/10 dimred）直接使用。
#   2. 统一截断到 800 轮（MAX_EPISODE=800）：所有曲线都画到 800 轮为止。
#   3. 每组 3 个 seed 对齐到公共长度，计算 mean / std。
#   4. 绘图：6 组同图，统一 y 轴，轻微滑动平均去噪，图例两列。

# 用法（一条指令）：
#   python plot_reward.py                       # 默认输出 reward_comparison.png
#   python plot_reward.py --output 8gai.png    # 指定输出文件名
#   python plot_reward.py --no-smooth          # 不做滑动平均
#   python plot_reward.py --window 15          # 调整滑动平均窗口（默认 7）
#   python plot_reward.py --ylim=-320,0        # 手动指定 y 轴范围（负值用等号）

# 说明：episode 间隔 10 是因为"训练 2000 轮、每 10 轮采样一次"；要让间隔变成 1，
# 必须把 reward 也一起线性插值（不能只改 episode 数字，否则横轴刻度语义错误）。
# 本脚本在内部自动完成插值，原始 data 文件无需改动。
# """
# import os
# import json
# import argparse
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# # 中文字体
# for name in ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Songti SC", "Arial Unicode MS"]:
#     if name in {f.name for f in fm.fontManager.ttflist}:
#         plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
#         break
# plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams.update({
#     "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
#     "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
#     "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
#     "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--", "grid.linewidth": 0.6,
#     "axes.linewidth": 1.0, "axes.edgecolor": "#333333",
# })

# HERE = os.path.dirname(os.path.abspath(__file__))
# DATA = os.path.join(HERE, "data")

# # 统一截断轮数
# MAX_EPISODE = 800

# # 6 组：(组id, 源文件, 算法, 模式)
# GROUPS = [
#     ("01_PPO_native",  "01_ppo_native_reward.json",  "PPO",  "native"),
#     ("02_SAC_native",  "02_sac_native_reward.json",  "SAC",  "native"),
#     ("03_DDQN_native", "03_ddqn_native_reward.json", "DDQN", "native"),
#     ("08_PPO_dimred",  "8_ppo_dimred_reward.json",   "PPO",  "dimred"),
#     ("09_SAC_dimred",  "9_sac_dimred_reward.json",   "SAC",  "dimred"),
#     ("10_DDQN_dimred", "10_ddqn_dimred_reward.json", "DDQN", "dimred"),
# ]

# # 图例友好名 + 颜色 + 线型（6 组全部实线：native 饱和色，dimred 同色系浅色）
# DISPLAY = {
#     "01_PPO_native":  ("PPO  (native)",  "#1f77b4", "-"),
#     "02_SAC_native":  ("SAC  (native)",  "#d62728", "-"),
#     "03_DDQN_native": ("DDQN (native)",  "#2ca02c", "-"),
#     "08_PPO_dimred":  ("PPO  (dimred)",  "#7fb3d5", "-"),
#     "09_SAC_dimred":  ("SAC  (dimred)",  "#e57373", "-"),
#     "10_DDQN_dimred": ("DDQN (dimred)",  "#7fbf7f", "-"),
# }


# def load_group(gid):
#     """读一个原始 reward 文件，返回 {seed: (episode_1d, reward_1d)}，内部完成插值+截断。"""
#     src = dict((g[0], g[1]) for g in GROUPS)[gid]
#     with open(os.path.join(DATA, src), encoding="utf-8") as f:
#         d = json.load(f)

#     seeds = {}
#     for s in ["42", "123", "2024"]:
#         v = d["seeds"].get(s, {})
#         ep = v.get("episode", [])
#         r = v.get("ep_reward", [])
#         if not ep or not r or len(ep) != len(r):
#             continue
#         res = align_seed(ep, r)
#         if res is not None:
#             grid, rr = res
#             # 截断到 MAX_EPISODE
#             n = min(len(grid), MAX_EPISODE)
#             seeds[s] = (grid[:n], rr[:n])
#     return seeds


# def align_seed(ep, r):
#     """把 (episode, reward) 对齐到 1ep 间隔、从 ep=1 开始的整数网格。

#     步长 > 1 时用 np.interp 线性插值到 1ep；步长 == 1 时直接使用。
#     """
#     ep = np.array(ep, dtype=float)
#     r = np.array(r, dtype=float)
#     if len(ep) < 2:
#         return None
#     start = int(round(ep[0]))
#     end = int(round(ep[-1]))
#     grid = np.arange(1, end + 1, dtype=float)
#     if len(ep) == end - start + 1 and np.allclose(np.diff(ep), 1.0):
#         # 已是 1ep 间隔
#         rr = r
#         if start > 1:
#             pad = np.full(start - 1, r[0])
#             rr = np.concatenate([pad, r])
#         return grid, rr
#     # 插值
#     rr = np.interp(grid, ep, r)
#     return grid, rr


# def light_smooth(y, window=7):
#     """轻微滑动平均去噪（不做过度平滑）。window<=1 表示不平滑。"""
#     y = np.asarray(y, dtype=float)
#     if window <= 1 or len(y) < window:
#         return y
#     kernel = np.ones(window) / window
#     out = np.convolve(y, kernel, mode="same")
#     half = window // 2
#     for k in range(half):
#         out[k] = y[:k + half + 1].mean()
#         out[-k - 1] = y[-k - half - 1:].mean()
#     return out


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--output", default="reward_comparison.png")
#     ap.add_argument("--no-smooth", action="store_true", help="不做任何平滑")
#     ap.add_argument("--window", type=int, default=7, help="滑动平均窗口（默认7）")
#     ap.add_argument("--ylim", default=None, help='y 轴范围，如 "-320,0"（负值用等号）')
#     args = ap.parse_args()

#     fig, ax = plt.subplots(figsize=(11, 7))

#     plotted = []
#     for gid, _, algo, mode in GROUPS:
#         seeds = load_group(gid)
#         if not seeds:
#             print(f"[SKIP] {gid}: 无有效数据")
#             continue
#         # 公共长度对齐
#         common = min(len(v[1]) for v in seeds.values())
#         Y = np.vstack([np.array(v[1][:common]) for v in seeds.values()])
#         x = np.array(seeds["42"][0][:common], dtype=float)
#         mean = Y.mean(axis=0)
#         std = Y.std(axis=0)

#         label, color, ls = DISPLAY[gid]
#         w = 0 if args.no_smooth else args.window
#         mean_s = light_smooth(mean, w)
#         std_s = light_smooth(std, w)

#         ax.plot(x, mean_s, color=color, ls=ls, lw=2.4, label=label, alpha=0.95,
#                 zorder=3 if mode == "native" else 2)
#         ax.fill_between(x, mean_s - std_s, mean_s + std_s, color=color, alpha=0.10, linewidth=0)
#         plotted.append((x, mean_s, std_s))
#         print(f"[OK] {gid}: n_seeds={len(seeds)}  ep={common}")

#     ax.axhline(0, color="black", linewidth=0.9, alpha=0.5, zorder=1)

#     ax.set_xlabel("episode (训练轮数)", fontsize=13)
#     ax.set_ylabel("ep_reward  (累积 −effective_peak_load)", fontsize=13)
#     ax.set_xlim(0, MAX_EPISODE)

#     if args.ylim:
#         lo, hi = [float(v) for v in args.ylim.split(",")]
#         ax.set_ylim(lo, hi)
#     else:
#         all_vals = [v for _, m, s in plotted for v in (m + s)]
#         all_vals += [v for _, m, s in plotted for v in (m - s)]
#         lo, hi = min(all_vals), max(all_vals)
#         pad = (hi - lo) * 0.05
#         ax.set_ylim(lo - pad, hi + pad)

#     ax.set_title("奖励收敛曲线对比：native vs dimred（三算法）", fontsize=15, fontweight="bold", pad=14)
#     ax.legend(loc="lower right", ncol=2, framealpha=0.9, edgecolor="#999999",
#               fancybox=True, fontsize=10.5)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     fig.tight_layout()
#     fig.savefig(args.output)
#     print(f"wrote {args.output}")


# if __name__ == "__main__":
#     main()
