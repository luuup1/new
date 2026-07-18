"""Diagnose the BC teacher ceiling on the RANDOM-order env.

The DRL MDP is order_mode="random" (evaluated "on the baseline"). Before
committing to BC, we must know what the min-load placement rule alone can
achieve under *random order* (not EDF order). If it is already near 1.0,
order is the real lever (Plan B); if it is clearly below 1.0, BC is worth it.

Prints mean/std eff_peak of "random order + min-load placement" over seeds.
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from tsn_sim import TSNSchedulingEnv, SimulationConfig
from tsn_sim.config import HeuristicConfig
from tsn_sim.heuristics import schedule_with_heuristic


def run_teacher(order_mode: str, period_mode: str, seed: int) -> float:
    sim = SimulationConfig(seed=seed, period_mode=period_mode)
    env = TSNSchedulingEnv(config=sim, order_mode=order_mode, reward_mode="mixed")
    state, _ = env.reset()
    while True:
        a = env.expert_min_load_action()
        state, _, done, _, info = env.step(a)
        if done:
            break
    return float(info.get("effective_peak_load", 0.0))


def run_random_feasible(period_mode: str, seed: int) -> float:
    sim = SimulationConfig(seed=seed, period_mode=period_mode)
    base = schedule_with_heuristic(
        sim and __import__("tsn_sim.scenario").build_scenario(sim),
        seed=seed,
        heuristic=HeuristicConfig(strategy="random_feasible"),
    )
    return float(base.metrics["effective_peak_load"])


if __name__ == "__main__":
    period_mode = sys.argv[1] if len(sys.argv) > 1 else "simple"
    seeds = list(range(1000, 1020))  # 20 seeds
    teacher = [run_teacher("random", period_mode, s) for s in seeds]
    base = [run_random_feasible(period_mode, s) for s in seeds]
    print(f"period_mode={period_mode}  (random-order env)")
    print(f"  teacher (random order + min-load placement): "
          f"eff_peak = {np.mean(teacher):.4f} +/- {np.std(teacher):.4f}")
    print(f"  random_feasible baseline (random order + random placement): "
          f"eff_peak = {np.mean(base):.4f} +/- {np.std(base):.4f}")
    gap = (np.mean(base) - np.mean(teacher)) / np.mean(base) * 100
    print(f"  teacher headroom vs baseline: {gap:+.2f}%")
