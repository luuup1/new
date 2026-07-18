"""Double-DQN training script for 5G-TSN scheduling.

Phase 3: value-based RL counterpart to PPO / SAC for the three-way
discrete-RL comparison.

Usage:
    python train_ddqn.py --period-mode simple --order-mode random
    python train_ddqn.py --period-mode simple --order-mode random --bc-pretrain
    python train_ddqn.py --eval-only --model ddqn_best.pth

Protocol compliance (see EXPERIMENT_PROTOCOL.md):
    - default order_mode="random"  (DRL evaluated "on the baseline")
    - comparison is against random_feasible (~1.0), never the heuristics
    - loads best checkpoint for final reporting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, ".")

from tsn_sim import (
    TSNSchedulingEnv,
    DDQNAgent,
    ReplayBuffer,
    SimulationConfig,
)
from tsn_sim.sac import Transition
from tsn_sim.bc import generate_demonstrations, bc_train


# =====================================================================
#  Training Loop
# =====================================================================

def train(
    env: TSNSchedulingEnv,
    agent: DDQNAgent,
    buffer: ReplayBuffer,
    *,
    episodes: int = 2000,
    warmup_steps: int = 1000,
    batch_size: int = 128,
    update_every: int = 1,
    eval_interval: int = 50,
    save_interval: int = 200,
    save_dir: str = "checkpoints_ddqn",
    log_interval: int = 10,
    verbose: bool = True,
    patience: int = 5,
    min_episodes: int = 100,
) -> Dict[str, List]:
    os.makedirs(save_dir, exist_ok=True)
    history = defaultdict(list)
    total_steps = 0
    best_eval_eff_peak = float("inf")
    no_improve_count = 0
    early_stopped = False

    for ep in range(1, episodes + 1):
        ep_start = time.time()
        state, info = env.reset()
        mask = env.action_masks()
        ep_reward = 0.0
        ep_steps = 0
        ep_q_value = 0.0
        ep_update_count = 0

        while True:
            # --- Action selection ---
            if total_steps < warmup_steps:
                valid = np.where(mask)[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0
            else:
                action = agent.select_action(state, mask, deterministic=False)

            agent.increment_total_steps(1)

            # --- Step ---
            next_state, reward, done, truncated, next_info = env.step(action)
            next_mask = env.action_masks()
            ep_reward += reward
            ep_steps += 1
            total_steps += 1

            # --- Store transition ---
            buffer.push(_make_transition(
                state, action, reward, next_state, done, mask, next_mask,
            ))

            # --- Gradient update ---
            if total_steps >= warmup_steps and total_steps % update_every == 0:
                metrics = agent.update(buffer, batch_size)
                ep_q_value += metrics.get("q_value", 0.0)
                ep_update_count += 1
            else:
                metrics = {}

            state = next_state
            mask = next_mask

            if done:
                break

        # --- Episode end ---
        ep_eff_peak = next_info.get("effective_peak_load", 0.0)
        ep_peak = next_info.get("peak_load", 0.0)
        ep_drop = next_info.get("drop_ratio", 0.0)
        ep_delay = next_info.get("average_delay_ms", 0.0)

        history["episode"].append(ep)
        history["ep_reward"].append(ep_reward)
        history["ep_eff_peak"].append(ep_eff_peak)
        history["ep_peak"].append(ep_peak)
        history["ep_drop"].append(ep_drop)
        history["ep_delay"].append(ep_delay)
        history["ep_steps"].append(ep_steps)
        history["epsilon"].append(agent.epsilon)

        avg_q = ep_q_value / max(1, ep_update_count)
        history["q_value"].append(avg_q)

        if ep % log_interval == 0 and verbose:
            n = min(log_interval, len(history["ep_reward"]))
            avg_reward = np.mean(history["ep_reward"][-n:])
            avg_peak = np.mean(history["ep_eff_peak"][-n:])
            avg_raw = np.mean(history["ep_peak"][-n:])
            avg_drop = np.mean(history["ep_drop"][-n:])
            avg_qval = np.mean(history["q_value"][-n:])
            ep_time = time.time() - ep_start
            print(
                f"Ep {ep:>5d}/{episodes}  "
                f"R={avg_reward:>7.3f}  "
                f"eff_peak={avg_peak:.4f}  "
                f"drop={avg_drop:.3f}  "
                f"eps={agent.epsilon:.3f}  "
                f"Q={avg_qval:>7.3f}  "
                f"buf={len(buffer):>6d}  "
                f"({ep_time:.1f}s)"
            )

        # --- Evaluation + Early Stopping ---
        if ep % eval_interval == 0:
            eval_metrics = evaluate(env, agent, num_episodes=5)
            eval_eff_peak = eval_metrics["eff_peak_mean"]
            history["eval_eff_peak"].append(eval_eff_peak)
            history["eval_peak"].append(eval_metrics["peak_mean"])
            history["eval_drop"].append(eval_metrics["drop_mean"])
            history["eval_episode"].append(ep)

            if eval_eff_peak < best_eval_eff_peak:
                best_eval_eff_peak = eval_eff_peak
                no_improve_count = 0
                save_path = os.path.join(save_dir, "ddqn_best.pth")
                agent.save(save_path)
                if verbose:
                    print(f"  >>> New best! eff_peak={eval_eff_peak:.4f}  "
                          f"saved to {save_path}")
            else:
                no_improve_count += 1

            if verbose:
                print(f"  [EVAL] ep={ep}  "
                      f"eff_peak={eval_metrics['eff_peak_mean']:.4f} +/- "
                      f"{eval_metrics['eff_peak_std']:.4f}  "
                      f"peak={eval_metrics['peak_mean']:.4f}  "
                      f"drop={eval_metrics['drop_mean']:.3f}  "
                      f"no_improve={no_improve_count}/{patience}")

            if ep >= min_episodes and no_improve_count >= patience:
                if verbose:
                    print(f"\n  [EARLY STOP] No improvement for {patience} "
                          f"evals. Stopping at ep={ep}.")
                early_stopped = True
                break

        # --- Periodic save ---
        if ep % save_interval == 0:
            save_path = os.path.join(save_dir, f"ddqn_ep{ep}.pth")
            agent.save(save_path)

    final_path = os.path.join(save_dir, "ddqn_checkpoint.pth")
    agent.save(final_path)
    if verbose:
        status = "early_stopped" if early_stopped else "completed"
        print(f"\nTraining {status} at ep={ep}/{episodes}")
        print(f"Final model saved to {final_path}")
        print(f"Best eval eff_peak: {best_eval_eff_peak:.4f}")

    return dict(history)


# =====================================================================
#  Evaluation
# =====================================================================

def evaluate(
    env: TSNSchedulingEnv,
    agent: DDQNAgent,
    num_episodes: int = 10,
) -> Dict[str, float]:
    eff_peaks, raw_peaks, drops, delays, rewards = [], [], [], [], []

    for _ in range(num_episodes):
        state, info = env.reset()
        mask = env.action_masks()
        ep_reward = 0.0
        while True:
            action = agent.select_action(state, mask, deterministic=True)
            state, reward, done, _, info = env.step(action)
            mask = env.action_masks()
            ep_reward += reward
            if done:
                break
        eff_peaks.append(info.get("effective_peak_load", 0.0))
        raw_peaks.append(info.get("peak_load", 0.0))
        drops.append(info.get("drop_ratio", 0.0))
        delays.append(info.get("average_delay_ms", 0.0))
        rewards.append(ep_reward)

    return {
        "eff_peak_mean": float(np.mean(eff_peaks)),
        "eff_peak_std": float(np.std(eff_peaks)),
        "peak_mean": float(np.mean(raw_peaks)),
        "peak_std": float(np.std(raw_peaks)),
        "drop_mean": float(np.mean(drops)),
        "delay_mean": float(np.mean(delays)),
        "reward_mean": float(np.mean(rewards)),
    }


# =====================================================================
#  Helpers
# =====================================================================

def _make_transition(state, action, reward, next_state, done, mask, next_mask):
    return Transition(
        state=state.astype(np.float32),
        action=int(action),
        reward=float(reward),
        next_state=next_state.astype(np.float32),
        done=bool(done),
        action_mask=mask.copy(),
        next_mask=next_mask.copy(),
    )


def build_scenario_safe(sim_config):
    from tsn_sim.scenario import build_scenario
    return build_scenario(sim_config)


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Double-DQN training for 5G-TSN scheduling")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.005,
                        help="0 = hard update every --target-update-freq; >0 = soft Polyak")
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=10000)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--reward-mode", type=str, default="load_balance",
                        choices=["terminal", "shaping", "exponential", "load_balance", "mixed"])
    parser.add_argument("--obs-mode", type=str, default="full", choices=["full", "compact"])
    parser.add_argument("--reward-delta", type=float, default=0.1)
    parser.add_argument("--reward-zeta", type=float, default=3.0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--period-mode", type=str, default="cyclic",
                        choices=["cyclic", "simple", "random"])
    parser.add_argument("--order-mode", type=str, default="random",
                        choices=["edf", "random"],
                        help="edf=固定EDF顺序(历史); random=随机顺序(在基线上进行,默认)")
    parser.add_argument("--bc-pretrain", action="store_true",
                        help="BC暖启动：用随机顺序min-load teacher预训练Q-net，再DQN微调")
    parser.add_argument("--bc-episodes", type=int, default=150)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="checkpoints_ddqn")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    period_mode = args.period_mode

    sim_config = SimulationConfig(seed=args.seed, period_mode=period_mode)

    if args.reward_mode == "shaping":
        env_reward_mode, env_shaping = "terminal", True
    elif args.reward_mode == "exponential":
        env_reward_mode, env_shaping = "exponential", False
    elif args.reward_mode == "load_balance":
        env_reward_mode, env_shaping = "load_balance", False
    elif args.reward_mode == "mixed":
        env_reward_mode, env_shaping = "mixed", False
    else:
        env_reward_mode, env_shaping = "terminal", args.shaping

    env = TSNSchedulingEnv(
        config=sim_config,
        use_step_shaping=env_shaping,
        shaping_weight=0.01,
        reward_mode=env_reward_mode,
        obs_mode=args.obs_mode,
        reward_delta=args.reward_delta,
        reward_zeta=args.reward_zeta,
        order_mode=args.order_mode,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    agent = DDQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay_steps=args.eps_decay_steps,
        target_update_freq=args.target_update_freq,
        device=device,
    )

    print(f"{'='*60}")
    print(f"Double-DQN Training for 5G-TSN Scheduling (period_mode={period_mode})")
    print(f"{'='*60}")
    print(f"  Obs dim:        {env.observation_space.shape[0]}")
    print(f"  Action dim:     {env.action_space.n}")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Warmup steps:   {args.warmup}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Buffer size:    {args.buffer_size}")
    print(f"  LR:             {args.lr}")
    print(f"  Gamma:          {args.gamma}")
    print(f"  Tau:            {args.tau}  (target_update_freq={args.target_update_freq})")
    print(f"  Epsilon:        {args.eps_start} -> {args.eps_end} over {args.eps_decay_steps} steps")
    print(f"  Reward mode:    {env_reward_mode}")
    print(f"  Order mode:     {args.order_mode}")
    print(f"  Seed:           {args.seed}")
    print(f"  Early stop:     {'OFF' if args.no_early_stop else f'patience={args.patience}'}")
    print(f"  BC pretrain:    {'ON' if args.bc_pretrain else 'OFF'}"
          + (f" (ep={args.bc_episodes}, epochs={args.bc_epochs}, lr={args.bc_lr})"
             if args.bc_pretrain else ""))
    print(f"{'='*60}")

    # --- BC warm-start (Route C) ---------------------------------------
    if args.bc_pretrain:
        print(f"\n{'='*60}")
        print("BC Pretraining (random-order min-load teacher)")
        print(f"{'='*60}")
        bc_env = TSNSchedulingEnv(
            config=sim_config,
            use_step_shaping=env_shaping,
            reward_mode=env_reward_mode,
            obs_mode=args.obs_mode,
            order_mode=args.order_mode,
            multi_scenario=True,
        )
        demo = generate_demonstrations(bc_env, args.bc_episodes)
        print(f"  demo episodes: {args.bc_episodes}  samples: {len(demo['action'])}")
        # For DQN, BC warms up the Q-net's action preferences via behavior cloning
        # on the *online* network (we clone the actor-equivalent: argmax of Q).
        stats = bc_train_dqn(agent, demo, epochs=args.bc_epochs, lr=args.bc_lr)
        os.makedirs(args.save_dir, exist_ok=True)
        agent.save(os.path.join(args.save_dir, "bc_ddqn.pth"))
        print(f"  BC done. greedy_match={stats['greedy_match']:.3f}  "
              f"saved {os.path.join(args.save_dir, 'bc_ddqn.pth')}")

    if args.eval_only:
        if args.model:
            agent.load(args.model)
            print(f"Loaded model from {args.model}")
        em = evaluate(env, agent, num_episodes=20)
        print(f"\nEvaluation (20 episodes):")
        for k, v in em.items():
            print(f"  {k:>16s}: {v:.4f}")
        _compare_with_baseline(sim_config, em)
        return

    buffer = ReplayBuffer(capacity=args.buffer_size)
    history = train(
        env=env,
        agent=agent,
        buffer=buffer,
        episodes=args.episodes,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        patience=9999 if args.no_early_stop else args.patience,
        min_episodes=args.min_episodes,
    )

    curves_path = os.path.join(args.save_dir, "training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining curves saved to {curves_path}")

    print(f"\n{'='*60}")
    print("Final Evaluation (20 episodes, BEST checkpoint, greedy policy)")
    print(f"{'='*60}")
    best_path = os.path.join(args.save_dir, "ddqn_best.pth")
    if os.path.exists(best_path):
        agent.load(best_path)
        print(f"  Loaded best checkpoint from {best_path}")
    em = evaluate(env, agent, num_episodes=20)
    for k, v in em.items():
        print(f"  {k:>16s}: {v:.4f}")

    _compare_with_baseline(sim_config, em)


def _compare_with_baseline(sim_config, em):
    """Compare DQN result against the TRUE naive baseline (random_feasible)."""
    print(f"\n{'='*60}")
    print("Comparison (same scenario, same seed)")
    print(f"{'='*60}")
    from tsn_sim.heuristics import schedule_with_heuristic
    from tsn_sim.config import HeuristicConfig
    scenario = build_scenario_safe(sim_config)
    base = schedule_with_heuristic(
        scenario, seed=sim_config.seed,
        heuristic=HeuristicConfig(strategy="random_feasible"),
    )
    base_peak = base.metrics["effective_peak_load"]
    dqn_peak = em["eff_peak_mean"]
    improvement = (base_peak - dqn_peak) / base_peak * 100
    print(f"  Baseline (random_feasible):  eff_peak = {base_peak:.4f}")
    print(f"  DQN (greedy, 20 ep):         eff_peak = {dqn_peak:.4f}")
    print(f"  Improvement vs baseline:     {improvement:+.2f}%")


if __name__ == "__main__":
    main()
