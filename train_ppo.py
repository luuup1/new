"""PPO training script for 5G-TSN scheduling.

Mirror of train_sac.py but on-policy:
    - collect a rollout of N episodes, then do K epochs of PPO updates
    - evaluate greedy policy every eval_interval episodes
    - early-stop when eval eff_peak stops improving

Usage:
    python train_ppo.py --period-mode cyclic     # sanity vs SAC (best 0.727)
    python train_ppo.py --period-mode simple
    python train_ppo.py --period-mode random
    python train_ppo.py --eval-only --model ppo_best.pth
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

from tsn_sim import TSNSchedulingEnv, PPOAgent, SimulationConfig
from tsn_sim.bc import generate_demonstrations, bc_train


# =====================================================================
#  Rollout collection
# =====================================================================

def collect_rollout(
    env: TSNSchedulingEnv,
    agent: PPOAgent,
    episodes: int,
) -> List[Dict[str, float]]:
    """Collect `episodes` full episodes into the agent's rollout buffer.

    Returns a list of per-episode metric dicts for logging.
    """
    ep_metrics: List[Dict[str, float]] = []

    for _ in range(episodes):
        state, info = env.reset()
        mask = env.action_masks()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            action, logp, value = agent.select_action(state, mask, deterministic=False)
            next_state, reward, done, truncated, next_info = env.step(action)

            if done:
                next_value = 0.0
            else:
                next_value = agent.value(next_state)

            agent.store_step(state, action, logp, reward, value,
                             next_value, done, mask)

            state = next_state
            mask = env.action_masks()
            ep_reward += reward
            ep_steps += 1

            if done:
                break

        ep_metrics.append({
            "eff_peak": next_info.get("effective_peak_load", 0.0),
            "peak": next_info.get("peak_load", 0.0),
            "drop": next_info.get("drop_ratio", 0.0),
            "delay": next_info.get("average_delay_ms", 0.0),
            "reward": ep_reward,
            "steps": ep_steps,
        })

    return ep_metrics


# =====================================================================
#  Training loop
# =====================================================================

def _save_curves(history, save_dir):
    """Incrementally save training curves to JSON (survives Ctrl+C)."""
    curves_path = os.path.join(save_dir, "training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(dict(history), f, indent=2)


def train(
    env: TSNSchedulingEnv,
    agent: PPOAgent,
    *,
    total_episodes: int = 2000,
    rollout_episodes: int = 10,  # 改为10：LCM(10,10)=10打印, LCM(10,50)=50评估, 和SAC频率一致
    epochs: int = 4,
    minibatch: int = 256,
    eval_interval: int = 50,
    save_interval: int = 200,
    save_dir: str = "checkpoints_ppo",
    log_interval: int = 10,  # rollout=10 → LCM(10,10)=10, 每10ep打印（和SAC一致）
    verbose: bool = True,
    patience: int = 5,
    min_episodes: int = 100,
) -> Dict[str, List]:
    os.makedirs(save_dir, exist_ok=True)
    history = defaultdict(list)
    episodes_done = 0
    best_eval_eff_peak = float("inf")
    no_improve_count = 0
    early_stopped = False

    try:
      while episodes_done < total_episodes and not early_stopped:
        ep_metrics = collect_rollout(env, agent, rollout_episodes)
        metrics = agent.update(epochs=epochs, minibatch=minibatch)

        # ---- Per-episode recording (unified format with SAC/DDQN) ----
        for m in ep_metrics:
            episodes_done += 1
            history["episode"].append(episodes_done)
            history["ep_reward"].append(m["reward"])
            history["ep_eff_peak"].append(m["eff_peak"])
            history["ep_peak"].append(m["peak"])
            history["ep_drop"].append(m["drop"])
            history["ep_delay"].append(m["delay"])
            history["ep_steps"].append(m["steps"])

        # PPO rollout-level metrics (fewer entries than per-episode data)
        history["policy_loss"].append(metrics.get("policy_loss", 0.0))
        history["value_loss"].append(metrics.get("value_loss", 0.0))
        history["entropy"].append(metrics.get("entropy", 0.0))

        if episodes_done % log_interval == 0 and verbose:
            n = min(log_interval, len(history["ep_reward"]))
            avg_reward = float(np.mean(history["ep_reward"][-n:]))
            avg_eff = float(np.mean(history["ep_eff_peak"][-n:]))
            avg_drop = float(np.mean(history["ep_drop"][-n:]))
            print(
                f"Ep {episodes_done:>5d}/{total_episodes}  "
                f"R={avg_reward:>7.3f}  "
                f"eff_peak={avg_eff:.4f}  "
                f"drop={avg_drop:.3f}  "
                f"polL={metrics.get('policy_loss', 0):.3f}  "
                f"valL={metrics.get('value_loss', 0):.3f}  "
                f"H={metrics.get('entropy', 0):.3f}"
            )

        # --- Evaluation + early stopping ---
        if episodes_done % eval_interval == 0:
            eval_metrics = evaluate(env, agent, num_episodes=5)
            eval_eff_peak = eval_metrics["eff_peak_mean"]
            history["eval_episode"].append(episodes_done)
            history["eval_eff_peak"].append(eval_eff_peak)
            history["eval_peak"].append(eval_metrics["peak_mean"])
            history["eval_drop"].append(eval_metrics["drop_mean"])

            if eval_eff_peak < best_eval_eff_peak:
                best_eval_eff_peak = eval_eff_peak
                no_improve_count = 0
                save_path = os.path.join(save_dir, "ppo_best.pth")
                agent.save(save_path)
                if verbose:
                    print(f"  >>> New best! eff_peak={eval_eff_peak:.4f}  "
                          f"saved to {save_path}")
            else:
                no_improve_count += 1

            # --- Incremental save: write training_curves.json every eval ---
            _save_curves(history, save_dir)

            if verbose:
                print(f"  [EVAL] ep={episodes_done}  "
                      f"eff_peak={eval_metrics['eff_peak_mean']:.4f} +/- "
                      f"{eval_metrics['eff_peak_std']:.4f}  "
                      f"peak={eval_metrics['peak_mean']:.4f}  "
                      f"drop={eval_metrics['drop_mean']:.3f}  "
                      f"no_improve={no_improve_count}/{patience}")

            if episodes_done >= min_episodes and no_improve_count >= patience:
                if verbose:
                    print(f"\n  [EARLY STOP] No improvement for {patience} "
                          f"evals. Stopping at ep={episodes_done}.")
                early_stopped = True

        if episodes_done % save_interval == 0:
            agent.save(os.path.join(save_dir, f"ppo_ep{episodes_done}.pth"))

    except KeyboardInterrupt:
        if verbose:
            print(f"\n[INTERRUPTED] Saving training curves at ep={episodes_done}...")
        _save_curves(history, save_dir)
        agent.save(os.path.join(save_dir, "ppo_checkpoint.pth"))
        if verbose:
            print(f"  Saved to {save_dir}")
        early_stopped = True

    final_path = os.path.join(save_dir, "ppo_checkpoint.pth")
    agent.save(final_path)
    if verbose:
        status = "early_stopped" if early_stopped else "completed"
        print(f"\nTraining {status} at ep={episodes_done}/{total_episodes}")
        print(f"Final model saved to {final_path}")
        print(f"Best eval eff_peak: {best_eval_eff_peak:.4f}")

    return dict(history)


# =====================================================================
#  Evaluation
# =====================================================================

def evaluate(env: TSNSchedulingEnv, agent: PPOAgent, num_episodes: int = 10):
    eff_peaks, raw_peaks, drops, delays, rewards = [], [], [], [], []
    for _ in range(num_episodes):
        state, info = env.reset()
        mask = env.action_masks()
        ep_reward = 0.0
        while True:
            action, _, _ = agent.select_action(state, mask, deterministic=True)
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
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="PPO training for 5G-TSN scheduling")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--rollout-episodes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="折扣因子；提高到0.99以改善长程信用分配(Phase 1.5)")
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--n-steps", type=int, default=5,
                        help="n-step bootstrapping 步数；>1 时启用(信用分配修复)")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--reward-mode", type=str, default="load_balance",
                        choices=["terminal", "shaping", "exponential", "load_balance", "mixed"])
    parser.add_argument("--obs-mode", type=str, default="full", choices=["full", "compact"])
    parser.add_argument("--reward-delta", type=float, default=0.1)
    parser.add_argument("--reward-zeta", type=float, default=3.0)
    parser.add_argument("--shaping-weight", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--period-mode", type=str, default="cyclic",
                        choices=["cyclic", "simple", "random"],
                        help="cyclic=历史复现; simple=Simple集(2固定周期); random=Random集")
    parser.add_argument("--order-mode", type=str, default="edf",
                        choices=["edf", "random"],
                        help="edf=固定EDF顺序(历史); random=随机顺序(native/dimred统一口径)")
    parser.add_argument("--action-mode", type=str, default="native",
                        choices=["native", "dimred"],
                        help="native=Discrete(96)动作; dimred=结构化7位二值动作(动作空间降维)")
    parser.add_argument("--bc-pretrain", action="store_true",
                        help="BC暖启动：用随机顺序min-load teacher预训练actor，再PPO微调")
    parser.add_argument("--bc-episodes", type=int, default=150,
                        help="BC演示生成的episode数（用multi_scenario多样性）")
    parser.add_argument("--bc-epochs", type=int, default=30,
                        help="BC预训练轮数")
    parser.add_argument("--bc-lr", type=float, default=1e-3,
                        help="BC预训练学习率")
    parser.add_argument("--save-dir", type=str, default="checkpoints_ppo")
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
        shaping_weight=args.shaping_weight,
        reward_mode=env_reward_mode,
        obs_mode=args.obs_mode,
        reward_delta=args.reward_delta,
        reward_zeta=args.reward_zeta,
        order_mode=args.order_mode,
        action_mode=args.action_mode,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.n_actions,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_steps=args.n_steps,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        value_coef=args.value_coef,
        device=device,
        action_mode=args.action_mode,
    )

    print(f"{'='*60}")
    print(f"PPO Training for 5G-TSN Scheduling (period_mode={period_mode})")
    print(f"{'='*60}")
    print(f"  Obs dim:       {env.observation_space.shape[0]}")
    print(f"  Action dim:    {env.n_actions} ({env.action_mode})")
    print(f"  Episodes:      {args.episodes} (rollout={args.rollout_episodes} ep/update)")
    print(f"  PPO epochs:    {args.epochs}  minibatch={args.minibatch}")
    print(f"  LR:            {args.lr}")
    print(f"  Gamma:         {args.gamma}  GAE-lambda={args.gae_lambda}  n-steps={args.n_steps}")
    print(f"  Clip-eps:      {args.clip_eps}  Ent-coef={args.ent_coef}")
    print(f"  Reward mode:   {env_reward_mode}")
    print(f"  Seed:          {args.seed}")
    print(f"  Early stop:    {'OFF' if args.no_early_stop else f'patience={args.patience}'}")
    print(f"  BC pretrain:   {'ON' if args.bc_pretrain else 'OFF'}"
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
        stats = bc_train(agent, demo, epochs=args.bc_epochs, lr=args.bc_lr)
        os.makedirs(args.save_dir, exist_ok=True)
        agent.save(os.path.join(args.save_dir, "bc_actor.pth"))
        print(f"  BC done. greedy_match={stats['greedy_match']:.3f}  "
              f"saved {os.path.join(args.save_dir, 'bc_actor.pth')}")

    if args.eval_only:
        if args.model:
            agent.load(args.model)
            print(f"Loaded model from {args.model}")
        em = evaluate(env, agent, num_episodes=20)
        print(f"\nEvaluation (20 episodes):")
        for k, v in em.items():
            print(f"  {k:>16s}: {v:.4f}")
        # Reference scale: random_feasible (random lower bound), NOT the baseline.
        # The comparison baseline for dimred is the native (non-reduced) version.
        from tsn_sim.heuristics import schedule_with_heuristic
        from tsn_sim.config import HeuristicConfig
        scenario = build_scenario_safe(sim_config)
        base = schedule_with_heuristic(
            scenario, seed=args.seed,
            heuristic=HeuristicConfig(strategy="random_feasible"),
        )
        base_peak = base.metrics["effective_peak_load"]
        improvement = (base_peak - em["eff_peak_mean"]) / base_peak * 100
        print(f"\n  Reference (random_feasible): eff_peak = {base_peak:.4f}")
        print(f"  PPO (greedy, 20 ep):        eff_peak = {em['eff_peak_mean']:.4f}")
        print(f"  Gain vs random reference:   {improvement:+.2f}%")
        return

    history = train(
        env=env, agent=agent,
        total_episodes=args.episodes,
        rollout_episodes=args.rollout_episodes,
        epochs=args.epochs,
        minibatch=args.minibatch,
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
    best_path = os.path.join(args.save_dir, "ppo_best.pth")
    if os.path.exists(best_path):
        agent.load(best_path)
        print(f"  Loaded best checkpoint from {best_path}")
    em = evaluate(env, agent, num_episodes=20)
    for k, v in em.items():
        print(f"  {k:>16s}: {v:.4f}")

    # --- Reference scale: random_feasible (random lower bound) ---
    # NOTE: the comparison baseline for the dimred method is the native
    # (non-reduced) version of the SAME algorithm. random_feasible / heuristics
    # / MILP are reference scales only, used to mark the absolute level.
    print(f"\n{'='*60}")
    print("Reference scale (same scenario, same seed)")
    print(f"{'='*60}")
    from tsn_sim.heuristics import schedule_with_heuristic
    from tsn_sim.config import HeuristicConfig
    scenario = build_scenario_safe(sim_config)
    base = schedule_with_heuristic(
        scenario, seed=args.seed,
        heuristic=HeuristicConfig(strategy="random_feasible"),
    )
    base_peak = base.metrics["effective_peak_load"]
    ppo_peak = em["eff_peak_mean"]
    improvement = (base_peak - ppo_peak) / base_peak * 100
    print(f"  Reference (random_feasible): eff_peak = {base_peak:.4f}")
    print(f"  PPO (greedy, 20 ep):         eff_peak = {ppo_peak:.4f}")
    print(f"  Gain vs random reference:    {improvement:+.2f}%")


def build_scenario_safe(sim_config):
    from tsn_sim.scenario import build_scenario
    return build_scenario(sim_config)


if __name__ == "__main__":
    main()
