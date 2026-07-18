"""True naive baseline on the random-order env: random placement (random_feasible
behavior) averaged over several random seeds, in the SAME env DRL will use.

This is the fair baseline for "DRL conducted on the baseline":
  baseline = random order (from env) + random placement
  DRL     = random order (from env) + learned placement
Same env, same order distribution, only the placement policy differs.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from tsn_sim import TSNSchedulingEnv, SimulationConfig

SEED = 42
PERIOD = "simple"
N = 20

cfg = SimulationConfig(seed=SEED, period_mode=PERIOD, order_mode="random")
peaks = []
for i in range(N):
    rng = np.random.RandomState(i)
    e = TSNSchedulingEnv(cfg, order_mode="random", reward_mode="load_balance")
    e.reset()
    while True:
        m = e.action_masks()
        acts = np.where(m)[0]
        a = int(rng.choice(acts)) if len(acts) else 0
        o, r, d, _, info = e.step(a)
        if d:
            break
    peaks.append(info["effective_peak_load"])

peaks = np.array(peaks)
print(f"[{PERIOD}] random-baseline eff_peak over {N} seeds: "
      f"mean={peaks.mean():.4f}  std={peaks.std():.4f}  min={peaks.min():.4f}  max={peaks.max():.4f}")
