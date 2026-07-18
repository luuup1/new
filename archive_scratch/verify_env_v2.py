"""Verify env changes: obs modes, reward modes, dimensions."""
import math
import numpy as np
from tsn_sim import TSNSchedulingEnv

print("=== Test 1: full obs mode (199-dim) ===")
env1 = TSNSchedulingEnv(obs_mode="full", reward_mode="terminal")
obs1, info1 = env1.reset()
print(f"obs_dim={obs1.shape[0]} (expected 199)")
print(f"action_space={env1.action_space.n} (expected 96)")
assert obs1.shape[0] == 199, f"Expected 199, got {obs1.shape[0]}"

print()
print("=== Test 2: compact obs mode (111-dim) ===")
env2 = TSNSchedulingEnv(obs_mode="compact", reward_mode="exponential")
obs2, info2 = env2.reset()
print(f"obs_dim={obs2.shape[0]} (expected 111)")
assert obs2.shape[0] == 111, f"Expected 111, got {obs2.shape[0]}"

print()
print("=== Test 3: exponential reward (first step) ===")
env3 = TSNSchedulingEnv(obs_mode="full", reward_mode="exponential",
                         reward_delta=0.1, reward_zeta=3.0)
obs3, info3 = env3.reset()
mask3 = env3.action_masks()
valid3 = np.where(mask3)[0]
action3 = int(valid3[0]) if len(valid3) > 0 else 0
next_obs3, reward3, done3, _, next_info3 = env3.step(action3)
expected_r = 0.1 * math.exp(-3.0 * env3._compute_current_peak())
print(f"step_reward={reward3:.6f} (expected ~{expected_r:.6f})")
print(f"reward_range: [{0.1:.4f} (peak=0), {0.1*math.exp(-3):.6f} (peak=1)]")
assert reward3 > 0, f"Exponential reward should be positive, got {reward3}"

print()
print("=== Test 4: terminal reward mode (no step reward) ===")
env4 = TSNSchedulingEnv(obs_mode="full", reward_mode="terminal")
obs4, info4 = env4.reset()
mask4 = env4.action_masks()
valid4 = np.where(mask4)[0]
action4 = int(valid4[0]) if len(valid4) > 0 else 0
_, reward4, done4, _, _ = env4.step(action4)
print(f"step_reward={reward4:.6f} (expected 0.0 for non-terminal step)")
assert reward4 == 0.0, f"Terminal mode step reward should be 0, got {reward4}"

print()
print("=== Test 5: full episode (compact + exponential) ===")
env5 = TSNSchedulingEnv(obs_mode="compact", reward_mode="exponential")
obs5, info5 = env5.reset()
total_r = 0.0
steps = 0
mask5 = env5.action_masks()
while True:
    valid5 = np.where(mask5)[0]
    a = int(np.random.choice(valid5)) if len(valid5) > 0 else 0
    obs5, r, done5, _, info5 = env5.step(a)
    total_r += r
    steps += 1
    mask5 = env5.action_masks()
    if done5:
        break

terminal_r = -info5["effective_peak_load"]
print(f"episode: steps={steps}, total_reward={total_r:.4f}")
print(f"terminal_reward={terminal_r:.4f}")
print(f"eff_peak={info5['effective_peak_load']:.4f}, peak={info5['peak_load']:.4f}")
print(f"step_rewards_sum={total_r - terminal_r:.4f} ({steps} steps)")

print()
print("=== Test 6: SAC agent with compact obs ===")
from tsn_sim import SACAgent
agent = SACAgent(111, 96, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
print(f"SAC agent created: obs=111, actions=96, device={agent.device}")

print()
print("ALL TESTS PASSED")
