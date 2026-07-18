"""Verify the corrected baseline setup:
1. order_mode='random' in env reproduces random_feasible's packet order (paired).
2. random_feasible peak on the random-order scenario is the TRUE baseline (~1.0).
3. A random agent in the random-order env matches random_feasible (sanity).
"""
import sys
import random
import numpy as np

sys.path.insert(0, ".")
from tsn_sim import TSNSchedulingEnv, SimulationConfig
from tsn_sim.heuristics import schedule_with_heuristic
from tsn_sim.candidate import instantiate_packets

SEED = 42
PERIOD = "simple"

# --- (1) order match between env(random) and random_feasible ---
cfg = SimulationConfig(seed=SEED, period_mode=PERIOD, order_mode="random")
env = TSNSchedulingEnv(cfg, order_mode="random")
env_order = list(env._packet_order)

base = schedule_with_heuristic(env.scenario, seed=SEED)
# recover random_feasible packet order via its own shuffle
rf_order = list(instantiate_packets(env.scenario).keys())
rng = random.Random(SEED)
rng.shuffle(rf_order)
print("env random order == random_feasible order:", tuple(rf_order) == env_order)

# --- (2) TRUE baseline: random_feasible peak on this scenario ---
print(f"random_feasible eff_peak (seed={SEED}): {base.metrics['effective_peak_load']:.4f}")
print(f"  scheduling_success_rate: {base.metrics.get('scheduling_success_rate', 'n/a')}")

# --- (3) random agent in random-order env (should ~= baseline) ---
random.seed(0)
e = TSNSchedulingEnv(cfg, order_mode="random", reward_mode="load_balance")
obs, info = e.reset()
total = 0.0
while True:
    m = e.action_masks()
    acts = np.where(m)[0]
    a = int(np.random.choice(acts)) if len(acts) else 0
    o, r, d, _, info = e.step(a)
    total += r
    if d:
        break
print(f"random agent (random-order env) eff_peak: {info['effective_peak_load']:.4f}")

# --- (4) edf order env for reference ---
cfg_edf = SimulationConfig(seed=SEED, period_mode=PERIOD, order_mode="edf")
env_edf = TSNSchedulingEnv(cfg_edf, order_mode="edf")
print("env edf order[0:5]:", env_edf._packet_order[:5])
print("env random order[0:5]:", env_order[:5])
