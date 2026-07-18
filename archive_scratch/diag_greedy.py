"""Diagnostic: does PPO's greedy policy match the min-post-load greedy at each step?

For every scheduling step, we compute:
  - post-load of the cell PPO chose
  - minimum post-load among ALL feasible cells (the heuristic's choice)
and count how often PPO == min (or within a small tolerance).
If PPO rarely matches the min, it simply has NOT learned the greedy rule that
the load_balance/mixed reward directly rewards -> cold-start failure -> needs IL.
If PPO matches locally but the episode still peaks at 1.0, the failure is global
coordination (peak is an episode-level MAX) -> needs MDP change.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from tsn_sim import TSNSchedulingEnv, PPOAgent, SimulationConfig

MODEL = "checkpoints_ppo_simple_mixed_g09/ppo_best.pth"
cfg = SimulationConfig(seed=42, period_mode="simple")
env = TSNSchedulingEnv(cfg, reward_mode="mixed")

agent = PPOAgent(obs_dim=env.observation_space.shape[0],
                 n_actions=env.action_space.n, device="cpu")
agent.load(MODEL)

n_match = 0
n_steps = 0
worst_gap_sum = 0.0
chosen_post_list = []
min_post_list = []

state, info = env.reset()
mask = env.action_masks()
while True:
    action, _, _ = agent.select_action(state, mask, deterministic=True)
    # compute post-load of chosen cell
    link = action // env.S + 1
    slot = action % env.S
    cell = (link, slot)
    cap = env.scenario.rb_capacity[cell]
    used = env._used_rb[cell]
    pkt = env._packets[env._packet_order[env._current_idx]]
    flow = env._flows[pkt.flow_id]
    bits_per_rb = env.scenario.rb_bits[(flow.flow_id, link, slot)]
    req = np.ceil(flow.packet_size_bits / bits_per_rb)
    chosen_post = (used + req) / cap

    # min post-load among feasible cells
    feasmask = mask
    acts = np.where(feasmask)[0]
    min_post = float("inf")
    if len(acts) > 0:
        for a in acts:
            ll = a // env.S + 1
            ss = a % env.S
            cc = (ll, ss)
            ccap = env.scenario.rb_capacity[cc]
            u = env._used_rb[cc]
            bpr = env.scenario.rb_bits[(flow.flow_id, ll, ss)]
            rr = np.ceil(flow.packet_size_bits / bpr)
            post = (u + rr) / ccap
            if post < min_post:
                min_post = post

    chosen_post_list.append(chosen_post)
    min_post_list.append(min_post)
    n_steps += 1
    if min_post < float("inf") and chosen_post <= min_post + 1e-6:
        n_match += 1
    worst_gap_sum += max(0.0, chosen_post - min_post)

    state, reward, done, _, info = env.step(action)
    mask = env.action_masks()
    if done:
        break

print(f"Total steps: {n_steps}")
print(f"PPO matched min-post-load cell: {n_match}/{n_steps} = {n_match/n_steps*100:.1f}%")
print(f"Mean gap (chosen - min): {worst_gap_sum/n_steps:.4f}")
print(f"Final eff_peak: {info['effective_peak_load']:.4f}  peak={info['peak_load']:.4f}  drop={info['drop_ratio']:.4f}")
print(f"Chosen-post-load mean={np.mean(chosen_post_list):.3f}  Min-feasible-post mean={np.mean(min_post_list):.3f}")
