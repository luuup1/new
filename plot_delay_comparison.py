# -*- coding: utf-8 -*-
"""三算法 DRL 时延对比图（PPO / SAC / TD3）。

数据来源：data/ 下的 6 个 *_delay.json，读取每个 seed 的 delay_mean_ms 字段
（= average_delay_ms，每 episode 调度完成的平均时延，单位 ms）。

输出三幅图：
  - figure_delay_native_comparison.png   未降维(native)三算法时延对比
  - figure_delay_dimred_comparison.png   降维(dimred)三算法时延对比
  - figure_delay_native_vs_dimred.png    native vs dimred 三算法并排对比（第三幅）

口径说明：
  - 每个算法 3 个 seed 的 delay_mean_ms 求 mean ± std（ddof=1，跨 seed 标准差）。
  - 时延越低越好。
  - 部分算法（PPO/SAC native）的 ep_delay 训练曲线缺失，delay_mean_ms 为
    best checkpoint 贪婪评估补录值；TD3 及 dimred 组为训练曲线平均。
    画图统一使用 delay_mean_ms 字段，两种来源的绝对水平可比较。
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
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# 算法顺序 + 统一配色（native / dimred / 并排 三幅图保持一致，便于对照）
ALGORITHMS = [
    ("PPO", "#1f77b4"),
    ("SAC", "#d62728"),
    ("TD3", "#2ca02c"),
]

# data 文件名映射：algorithm -> (native_file, dimred_file)
DELAY_FILES = {
    "PPO": ("05_ppo_native_delay.json",  "12_ppo_dimred_delay.json"),
    "SAC": ("06_sac_native_delay.json",  "13_sac_dimred_delay.json"),
    "TD3": ("16_td3_native_delay.json",  "18_td3_dimred_delay.json"),
}

# native / dimred 的配色（并排图里用）
NATIVE_COLOR = "#1f77b4"   # 蓝
DIMRED_COLOR = "#ff7f0e"   # 橙


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_seed_means(filename):
    """读取一个 delay json，返回 {seed -> delay_mean_ms(float or None)}。"""
    p = os.path.join(DATA, filename)
    if not os.path.exists(p):
        print(f"  [MISSING] {filename}")
        return {}
    d = load_json(p)
    seeds = d.get("seeds", {})
    out = {}
    for seed, v in seeds.items():
        m = v.get("delay_mean_ms", None)
        out[seed] = m
    return out


def algo_stats(filename):
    """返回 (means_list, mean, std) —— 跨 3 seed 的 mean±std。"""
    seed_means = collect_seed_means(filename)
    vals = [m for m in seed_means.values() if m is not None]
    if not vals:
        return [], 0.0, 0.0
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return vals, mean, std


def plot_group(grp, out_name, title):
    """画一个分组（native 或 dimred）的柱状图。"""
    names, colors, means, stds, n_seeds = [], [], [], [], []
    for algo, color in ALGORITHMS:
        filename = DELAY_FILES[algo][0 if grp == "native" else 1]
        vals, mean, std = algo_stats(filename)
        print(f"  [{grp}] {algo}: mean={mean:.4f}ms  std={std:.4f}  "
              f"seeds={len(vals)}  per_seed={[round(v,4) for v in vals]}")
        if not vals:
            print(f"     -> 无有效时延数据，跳过 {algo}")
            continue
        names.append(algo)
        colors.append(color)
        means.append(mean)
        stds.append(std)
        n_seeds.append(len(vals))

    if not names:
        print(f"  [{grp}] 没有任何算法有时延数据，无法绘图。")
        return

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.2,
           yerr=stds, capsize=6,
           error_kw=dict(elinewidth=1.5, ecolor="black"))

    for xi, (mv, sd, ns) in enumerate(zip(means, stds, n_seeds)):
        txt = f"{mv:.3f}\n±{sd:.3f}\n(n={ns})"
        ax.text(xi, mv + sd + 0.01, txt, ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Average delay (ms)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    top = max(means) + max(stds) + 0.3
    ax.set_ylim(0, top)

    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")
    return {n: f"{m:.3f}±{s:.3f}ms (n={ns})"
            for n, m, s, ns in zip(names, means, stds, n_seeds)}


def plot_native_vs_dimred(out_name, title):
    """第三幅图：native vs dimred 三算法并排对比（grouped bar）。"""
    names, native_means, native_stds, dimred_means, dimred_stds = [], [], [], [], []
    for algo, _ in ALGORITHMS:
        nf, df = DELAY_FILES[algo]
        n_vals, n_mean, n_std = algo_stats(nf)
        d_vals, d_mean, d_std = algo_stats(df)
        if not n_vals or not d_vals:
            print(f"  [combined] 跳过 {algo}（native={len(n_vals)} seeds, "
                  f"dimred={len(d_vals)} seeds）")
            continue
        names.append(algo)
        native_means.append(n_mean)
        native_stds.append(n_std)
        dimred_means.append(d_mean)
        dimred_stds.append(d_std)
        print(f"  [combined] {algo}: native={n_mean:.4f}±{n_std:.4f}  "
              f"dimred={d_mean:.4f}±{d_std:.4f}")

    if not names:
        print("  [combined] 没有可并排对比的算法，无法绘图。")
        return

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    b1 = ax.bar(x - width/2, native_means, width, color=NATIVE_COLOR,
                edgecolor="black", linewidth=1.2, label="native",
                yerr=native_stds, capsize=5,
                error_kw=dict(elinewidth=1.5, ecolor="black"))
    b2 = ax.bar(x + width/2, dimred_means, width, color=DIMRED_COLOR,
                edgecolor="black", linewidth=1.2, label="dimred",
                yerr=dimred_stds, capsize=5,
                error_kw=dict(elinewidth=1.5, ecolor="black"))

    # 柱顶标注均值
    for xi, (nm, ns) in enumerate(zip(native_means, native_stds)):
        ax.text(xi - width/2, nm + ns + 0.01, f"{nm:.3f}", ha="center",
                va="bottom", fontsize=9, fontweight="bold", color=NATIVE_COLOR)
    for xi, (dm, ds) in enumerate(zip(dimred_means, dimred_stds)):
        ax.text(xi + width/2, dm + ds + 0.01, f"{dm:.3f}", ha="center",
                va="bottom", fontsize=9, fontweight="bold", color=DIMRED_COLOR)

    ax.set_ylabel("Average delay (ms)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    all_means = native_means + dimred_means
    all_stds = native_stds + dimred_stds
    top = max(all_means) + max(all_stds) + 0.3
    ax.set_ylim(0, top)

    fig.tight_layout()
    out = os.path.join(HERE, out_name)
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")
    return {n: {"native": f"{nm:.3f}±{nstd:.3f}", "dimred": f"{dm:.3f}±{dstd:.3f}"}
            for n, nm, nstd, dm, dstd in zip(
                names, native_means, native_stds, dimred_means, dimred_stds)}


def main():
    print("=== 三算法 DRL 时延数据收集（PPO / SAC / TD3）===")

    print("\n--- 未降维 (native) ---")
    summary_native = plot_group(
        "native",
        "figure_delay_native_comparison.png",
        "Average packet delay - native (no reduction) | 50 flows, simple set, random-order",
    )

    print("\n--- 降维 (dimred) ---")
    summary_dimred = plot_group(
        "dimred",
        "figure_delay_dimred_comparison.png",
        "Average packet delay - dimred (reduced) | 50 flows, simple set, random-order",
    )

    print("\n--- native vs dimred 并排对比 ---")
    summary_combined = plot_native_vs_dimred(
        "figure_delay_native_vs_dimred.png",
        "Average packet delay - native vs dimred | 50 flows, simple set, random-order",
    )

    print("\n=== summary (native) ===")
    print(json.dumps(summary_native or {}, ensure_ascii=False, indent=2))
    print("=== summary (dimred) ===")
    print(json.dumps(summary_dimred or {}, ensure_ascii=False, indent=2))
    print("=== summary (native vs dimred) ===")
    print(json.dumps(summary_combined or {}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
