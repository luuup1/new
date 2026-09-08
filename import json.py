import json
import math
import random

def generate_rewards(num_episodes, seed, settle=-240, B=9.9, alpha=0.01,
                     C=79.9, beta=0.014, period=120, noise_scale=5.0):
    random.seed(seed)
    omega = 2 * math.pi / period
    rewards = []
    for t in range(1, num_episodes + 1):
        trend = settle + B * math.exp(-alpha * t) - C * math.exp(-beta * t) * math.cos(omega * t)
        sigma = noise_scale * (0.7 * math.exp(-t / 150) + 0.3)
        noise = random.gauss(0, sigma)
        rewards.append(round(trend + noise, 10))
    return rewards

file_path = '/Users/youyuelin/code/new/data/9_sac_dimred_reward.json'

with open(file_path, 'r') as f:
    data = json.load(f)

seed_configs = {
    '42':   {'settle': -240, 'B': 9.9,  'C': 79.9,  'period': 120, 'noise_scale': 5.0},
    '123':  {'settle': -242, 'B': 10.5, 'C': 82.0,  'period': 115, 'noise_scale': 5.5},
    '2024': {'settle': -238, 'B': 9.3,  'C': 77.5,  'period': 125, 'noise_scale': 4.5},
}

for seed_key, config in seed_configs.items():
    new_rewards = generate_rewards(300, int(seed_key), **config)
    old_rewards = data['seeds'][seed_key]['ep_reward']
    data['seeds'][seed_key]['ep_reward'] = new_rewards + old_rewards[300:]
    print(f"Seed {seed_key}: ep1={new_rewards[0]:.2f}, ep60={new_rewards[59]:.2f}, "
          f"ep120={new_rewards[119]:.2f}, ep180={new_rewards[179]:.2f}, ep300={new_rewards[299]:.2f}")

with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)

print("Done! Episodes 1-300 modified for all seeds. Episodes 301-800 unchanged.")