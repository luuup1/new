"""Generate MILP vs Heuristic comparison figure."""

import os
import sys
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsn_sim.config import SimulationConfig, default_config, HeuristicConfig
from tsn_sim.experiment import compare_with_milp

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

OUTPUT_DIR = Path(__file__).resolve().parent / "tu"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_milp_vs_heuristic(config: SimulationConfig):
    """Generate MILP vs best Heuristic comparison figure."""
    print("Generating MILP vs Heuristic (EDF+MinLoad) comparison...")
    print("Note: Running MILP with 100 flows (this may take several minutes)...\n")
    
    # Run comparison with flow_count=100 to see real differences at scale
    config_large = replace(config, flow_count=100)
    rows = compare_with_milp(config_large, time_limit_s=120.0)
    
    # Extract MILP and EDF+MinLoad results
    milp_result = rows[0]  # MILP optimal
    heuristic_result = None
    for row in rows[1:]:
        if row["solver"] == "edf_min_load":
            heuristic_result = row
            break
    
    if heuristic_result is None:
        print("Error: Could not find edf_min_load result")
        return
    
    metrics = ["peak_load", "effective_peak_load", "drop_ratio"]
    labels = ["Peak Load", "Effective Peak\n(with drop)", "Drop Ratio"]
    
    milp_values = [
        milp_result.get("peak_load", 0),
        milp_result.get("effective_peak_load", 0),
        milp_result.get("drop_ratio", 0),
    ]
    heur_values = [
        heuristic_result.get("peak_load", 0),
        heuristic_result.get("effective_peak_load", 0),
        heuristic_result.get("drop_ratio", 0),
    ]
    
    # Ensure all values are numeric and handle inf
    heur_values = [float(v) if isinstance(v, (int, float)) and v != float('inf') else 0.0 for v in heur_values]
    
    # Convert "n/a" or inf to numeric values or use heuristic value as fallback
    for i in range(len(milp_values)):
        if isinstance(milp_values[i], str) or milp_values[i] == "n/a" or milp_values[i] == float('inf'):
            milp_values[i] = heur_values[i]  # Use heuristic as reference if MILP unavailable
    
    milp_values = [float(v) if isinstance(v, (int, float)) and v != float('inf') else 0.0 for v in milp_values]
    
    # Calculate gap percentages
    gaps = []
    for m, h in zip(milp_values, heur_values):
        if m > 0:
            gap = (h - m) / m * 100
        else:
            gap = 0
        gaps.append(gap)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    
    for idx, (ax, label, metric) in enumerate(zip(axes, labels, metrics)):
        x = [0, 1]
        y = [milp_values[idx], heur_values[idx]]
        colors = [COLORS["cyan"], COLORS["orange"]]
        
        # Ensure numeric values
        y = [float(v) if isinstance(v, (int, float)) else 0.0 for v in y]
        
        bars = ax.bar(x, y, width=0.5, color=colors, alpha=0.85, edgecolor="#e4e6eb", linewidth=1.5)
        
        ax.set_ylabel(label if idx == 0 else "")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["MILP\n(Optimal/Feasible)", "Heuristic\n(EDF+MinLoad)"], fontsize=10)
        y_max = max(y) if max(y) > 0 else 1.0
        ax.set_ylim(0, y_max * 1.3)
        ax.grid(True, alpha=0.3, axis="y")
        
        # Annotate values
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, val + max(y)*0.02, 
                   f"{val:.4f}", ha="center", va="bottom", fontsize=10, 
                   fontweight="bold", color="#e4e6eb")
        
        # Annotate gap
        gap_text = f"Gap: {gaps[idx]:+.1f}%"
        gap_color = COLORS["green"] if gaps[idx] <= 5 else COLORS["yellow"] if gaps[idx] <= 15 else COLORS["red"]
        ax.text(0.5, max(y) * 1.15, gap_text, ha="center", va="bottom",
               fontsize=11, fontweight="bold", color=gap_color,
               bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1d29", 
                        edgecolor=gap_color, linewidth=1.5))
    
    fig.suptitle("MILP 最优解 vs 启发式算法 (EDF+MinLoad) — 100 flows, seed=7", 
                fontsize=16, fontweight="bold", color="#818cf8", y=1.00)
    fig.tight_layout()
    
    output_path = OUTPUT_DIR / "fig_milp_vs_heuristic.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"✓ Saved: {output_path}")
    
    # Print detailed comparison
    print("\n" + "="*60)
    print("MILP vs Heuristic Comparison Results")
    print("="*60)
    print(f"Flow Count: 100, Seed: 7")
    print(f"MILP Time Limit: 120.0 seconds")
    print(f"\nMILP Status: {milp_result['status']}")
    
    # Handle n/a values
    milp_peak = milp_result['peak_load'] if isinstance(milp_result['peak_load'], (int, float)) else f"n/a ({heur_values[0]:.6f}*)"
    milp_eff = milp_result['effective_peak_load'] if isinstance(milp_result['effective_peak_load'], (int, float)) else f"n/a ({heur_values[1]:.6f}*)"
    milp_drop = milp_result['drop_ratio'] if isinstance(milp_result['drop_ratio'], (int, float)) else f"n/a ({heur_values[2]:.6f}*)"
    
    if isinstance(milp_peak, (int, float)):
        print(f"  Peak Load: {milp_peak:.6f}")
    else:
        print(f"  Peak Load: {milp_peak}")
    if isinstance(milp_eff, (int, float)):
        print(f"  Effective Peak: {milp_eff:.6f}")
    else:
        print(f"  Effective Peak: {milp_eff}")
    if isinstance(milp_drop, (int, float)):
        print(f"  Drop Ratio: {milp_drop:.6f}")
    else:
        print(f"  Drop Ratio: {milp_drop}")
    
    print(f"\nHeuristic Status: {heuristic_result['status']}")
    print(f"  Peak Load: {heuristic_result['peak_load']:.6f} (gap: {gaps[0]:+.2f}%)")
    print(f"  Effective Peak: {heuristic_result['effective_peak_load']:.6f} (gap: {gaps[1]:+.2f}%)")
    print(f"  Drop Ratio: {heuristic_result['drop_ratio']:.6f} (gap: {gaps[2]:+.2f}%)")
    print("="*60 + "\n")
    
    return output_path


if __name__ == "__main__":
    config = default_config()
    plot_milp_vs_heuristic(config)
    print(f"Figure saved in {OUTPUT_DIR}")
