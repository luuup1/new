"""Verify env works with 50-flow + uniform-size config."""
from tsn_sim.config import default_config
from tsn_sim.env import TSNSchedulingEnv

sc = default_config()
print(f"Config: flow_count={sc.flow_count}, packet_size_bits={sc.packet_size_bits}")

env = TSNSchedulingEnv(sc)
print(f"Env created: obs_space={env.observation_space.shape}, action_space={env.action_space.n}")

obs, info = env.reset()
print(f"Reset OK: obs_shape={obs.shape}, info_keys={list(info.keys())}")

# Run a full episode with random valid actions
total_reward = 0
steps = 0
done = False
truncated = False

while not (done or truncated):
    mask = info.get("action_mask", None)
    if mask is not None:
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            print(f"Step {steps}: NO valid actions! Breaking.")
            break
        action = np.random.choice(valid_actions)
    else:
        action = env.action_space.sample()
    
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

print(f"Episode done: steps={steps}, total_reward={total_reward:.4f}")
print(f"Final info: peak_load={info.get('peak_load', 'N/A')}, "
      f"eff_peak={info.get('effective_peak_load', 'N/A')}")
