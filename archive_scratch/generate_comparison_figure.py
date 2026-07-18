"""Generate 4-strategy comparison figure only."""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsn_sim.config import SimulationConfig, default_config
from tsn_sim.experiment import compare_heuristics

# ── global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 12,
    "axes.facecolor": "#1a1d29",
    "figure.facecolor": "#0f1117",
    "axes.edgecolor": "#3a3f55",
    "axes.labelcolor": "#e4e6eb",
    "xtick.color": "#9ca3af",
    "ytick.color": "#9ca3af",
    "text.color": "#e4e6eb",
    "axes.grid": True,
    "grid.color": "#2d3142",
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "green":  "#22c55e",
    "red":    "#ef4444",
    "yellow": "#f59e0b",
    "orange": "#f97316",
    "cyan":   "#06b6d4",
    "purple": "#818cf8",
    "blue":   "#3b82f6",
    "gray":   "#6b7280",
}

STRATEGY_COLORS = {
    "random_feasible":       COLORS["red"],
    "edf_min_load":          COLORS["green"],
    "urgency_lexicographic": COLORS["yellow"],
    "edf_min_peak":          COLORS["cyan"],
}

STRATEGY_LABELS = {
    "random_feasible":       "Random",
    "edf_min_load":          "EDF+MinLoad",
    "urgency_lexicographic": "Urgency+Lex",
    "edf_min_peak":          "EDF+MinPeak",
}

OUTPUT_DIR = Path(__file__).resolve().parent / "tu"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_heuristic_compare(config: SimulationConfig):
    """Generate 4-strategy comparison figure."""
    print("Generating heuristic strategy comparison figure...")
    rows = compare_heuristics(config)
    strategies = [r["strategy"] for r in rows]
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    peaks = [r["peak_load"] for r in rows]
    eff_peaks = [r["effective_peak_load"] for r in rows]
    delays = [r["average_delay_ms"] for r in rows]
    utils = [r["resource_utilization"] * 100 for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # — Peak Load —
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.38
    bars1 = ax.bar(x - w/2, peaks, w, label="Peak Load",
                   color=[STRATEGY_COLORS[s] for s in strategies], alpha=0.9)
    bars2 = ax.bar(x + w/2, eff_peaks, w, label="Effective Peak",
                   color=[STRATEGY_COLORS[s] for s in strategies], alpha=0.45,
                   edgecolor=[STRATEGY_COLORS[s] for s in strategies], linewidth=1.5)
    ax.set_ylabel("Normalized Load")
    ax.set_title("Peak Load", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, loc="upper right")
    for bar, val in zip(bars1, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, color="#e4e6eb")

    # — Average Delay —
    ax = axes[1]
    bars = ax.bar(x, delays, 0.5,
                  color=[STRATEGY_COLORS[s] for s in strategies], alpha=0.85)
    ax.set_ylabel("Delay (ms)")
    ax.set_title("Average Delay", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    for bar, val in zip(bars, delays):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    # — Resource Utilization —
    ax = axes[2]
    bars = ax.bar(x, utils, 0.5,
                  color=[STRATEGY_COLORS[s] for s in strategies], alpha=0.85)
    ax.set_ylabel("Utilization (%)")
    ax.set_title("Resource Utilization", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    for bar, val in zip(bars, utils):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.15, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=9)

    fig.suptitle("启发式策略对比 (20 flows, seed=7)", fontsize=16, fontweight="bold",
                 color="#818cf8", y=1.02)
    fig.tight_layout()
    
    output_path = OUTPUT_DIR / "fig_heuristic_compare.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"✓ Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    config = default_config()
    plot_heuristic_compare(config)
    print(f"\nDone. Figure saved in {OUTPUT_DIR}")
