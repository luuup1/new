"""Verify whether load_balance reward separates good vs bad policies."""
import math
import numpy as np

from tsn_sim import SimulationConfig, TSNSchedulingEnv
from tsn_sim.heuristics import schedule_with_heuristic
from tsn_sim.ppo import PPOAgent

cfg = SimulationConfig(seed=42, period_mode="simple")


def total_lb_reward_from_schedule(schedule_entries):
    e = TSNSchedulingEnv(cfg, reward_mode="load_balance")
    obs, _ = e.reset()
    order = e._packet_order
    cell_of = {(se.flow_id, se.packet_index): (se.link, se.slot) for se in schedule_entries}
    total = 0.0
    for pk in order:
        if pk not in cell_of:
            a = 0
        else:
            l, s = cell_of[pk]
            a = (l - 1) * e.S + s
        o, r, d, tr, info = e.step(a)
        total += r
        if d:
            break
    return total, info["effective_peak_load"]


h = schedule_with_heuristic(SimulationConfig(seed=42, period_mode="simple")._build_episode_data if False else TSNSchedulingEnv(cfg, reward_mode="load_balance").scenario, seed=42)
h_total, h_peak = total_lb_reward_from_schedule(h.schedule)
print(f"[heuristic edf_min_load] eff_peak={h_peak:.4f}  load_balance_total={h_total:.3f}")

# random
rng = np.random.default_rng(0)
e = TSNSchedulingEnv(cfg, reward_mode="load_balance")
obs, _ = e.reset()
total = 0.0
while True:
    m = e.action_masks()
    acts = np.where(m)[0]
    a = int(acts[rng.integers(len(acts))]) if len(acts) else 0
    o, r, d, tr, info = e.step(a)
    total += r
    if d:
        break
print(f"[random policy      ] eff_peak={info['effective_peak_load']:.4f}  load_balance_total={total:.3f}")

# PPO best
agent = PPOAgent(obs_dim=199, n_actions=96, device="cpu")
agent.load("checkpoints_ppo_simple/ppo_best.pth")
e = TSNSchedulingEnv(cfg, reward_mode="load_balance")
obs, _ = e.reset()
total = 0.0
while True:
    m = e.action_masks()
    a, _, _ = agent.select_action(obs, m, deterministic=True)
    o, r, d, tr, info = e.step(a)
    total += r
    if d:
        break
print(f"[PPO best           ] eff_peak={info['effective_peak_load']:.4f}  load_balance_total={total:.3f}")
