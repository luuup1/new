# -*- coding: utf-8 -*-
"""时延补录评估脚本 — 用 best checkpoint 做确定性贪婪评估，得到每个 seed 的
average_delay_ms（ms），回填 data/05_ppo_native_delay.json 与 06_sac_native_delay.json。

口径（与 EXPERIMENT_PROTOCOL.md 一致）：
  - period_mode = simple, order_mode = random, flow_count = 50, obs_mode = full
  - reward_mode = load_balance（时延不依赖 reward，仅为与训练一致）
  - 动作空间 = native（Discrete(96)）

为什么需要补录：
  - PPO cold 三目录的 training_curves.json 是旧版脚本生成，**没有 ep_delay 字段**。
  - SAC seed42 的 ep_delay 全为 null（历史写入缺失）。
  因此用各目录的 best checkpoint 重新评估时延。

评估方式：
  对每个 seed，构造与训练完全一致的场景（seed 决定 scenario + random 顺序），
  加载 best 模型，确定性贪婪走完整个 episode，读取 info["average_delay_ms"]。
  由于 env 无 multi_scenario 且 greedy 确定性，每个 seed 的时延是确定性标量；
  仍跑 N_EVAL 次以验证确定性（应完全一致）。
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tsn_sim import TSNSchedulingEnv, PPOAgent, SACAgent, SimulationConfig

N_EVAL = 5  # 每个 seed 跑 5 次验证确定性（结果应一致）


def make_env(seed: int):
    cfg = SimulationConfig(seed=seed, period_mode="simple")
    return TSNSchedulingEnv(
        config=cfg,
        order_mode="random",
        obs_mode="full",
        reward_mode="load_balance",
    )


def eval_ppo(seed: int, ckpt_path: str):
    env = make_env(seed)
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
    )
    agent.load(ckpt_path)
    delays, eff_peaks = [], []
    for _ in range(N_EVAL):
        state, _ = env.reset()
        mask = env.action_masks()
        while True:
            a, _, _ = agent.select_action(state, mask, deterministic=True)
            state, _, done, _, info = env.step(a)
            mask = env.action_masks()
            if done:
                break
        delays.append(float(info.get("average_delay_ms", 0.0)))
        eff_peaks.append(float(info.get("effective_peak_load", 0.0)))
    return delays, eff_peaks


def eval_sac(seed: int, ckpt_path: str):
    env = make_env(seed)
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.n_actions,
    )
    agent.load(ckpt_path)
    delays, eff_peaks = [], []
    for _ in range(N_EVAL):
        state, _ = env.reset()
        mask = env.action_masks()
        while True:
            out = agent.select_action(state, mask, deterministic=True)
            a = out[0] if isinstance(out, tuple) else out
            state, _, done, _, info = env.step(a)
            mask = env.action_masks()
            if done:
                break
        delays.append(float(info.get("average_delay_ms", 0.0)))
        eff_peaks.append(float(info.get("effective_peak_load", 0.0)))
    return delays, eff_peaks


PPO = {
    "42": os.path.join(HERE, "checkpoints_ppo_cold_seed42", "ppo_best.pth"),
    "123": os.path.join(HERE, "checkpoints_ppo_cold_seed123", "ppo_best.pth"),
    "2024": os.path.join(HERE, "checkpoints_ppo_cold_seed2024", "ppo_best.pth"),
}
SAC = {
    "42": os.path.join(HERE, "checkpoints_sac_seed42_0716", "sac_best.pth"),
    "123": os.path.join(HERE, "checkpoints_sac_seed123_0716", "sac_best.pth"),
    "2024": os.path.join(HERE, "checkpoints_sac_seed2024_0716", "sac_best.pth"),
}


def main():
    results = {"ppo": {}, "sac": {}}
    print("=" * 60)
    print("PPO best checkpoint 时延评估")
    print("=" * 60)
    for seed, pth in PPO.items():
        delays, eff = eval_ppo(int(seed), pth)
        mean = float(np.mean(delays))
        results["ppo"][seed] = {
            "delay_ms": round(mean, 4),
            "delay_all_runs": [round(d, 4) for d in delays],
            "eff_peak": round(float(np.mean(eff)), 4),
        }
        print(f"  seed {seed:>4s}: delay={mean:.4f} ms  eff_peak={np.mean(eff):.4f}  runs={[round(d,3) for d in delays]}")

    print("=" * 60)
    print("SAC best checkpoint 时延评估")
    print("=" * 60)
    for seed, pth in SAC.items():
        delays, eff = eval_sac(int(seed), pth)
        mean = float(np.mean(delays))
        results["sac"][seed] = {
            "delay_ms": round(mean, 4),
            "delay_all_runs": [round(d, 4) for d in delays],
            "eff_peak": round(float(np.mean(eff)), 4),
        }
        print(f"  seed {seed:>4s}: delay={mean:.4f} ms  eff_peak={np.mean(eff):.4f}  runs={[round(d,3) for d in delays]}")

    out = os.path.join(HERE, "_eval_delay_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
