"""SAC training script for 5G-TSN scheduling.

Usage:
    python train_sac.py                          # default config
    python train_sac.py --episodes 3000          # more episodes
    python train_sac.py --shaping                # enable step reward shaping
    python train_sac.py --eval-only --model sac_checkpoint.pth  # eval only

Training loop:
    1. Warmup:   random actions to fill replay buffer (warmup_steps)
    2. Train:    SAC sampling + gradient updates each step
    3. Evaluate: greedy policy every eval_interval episodes
    4. Save:     checkpoint every save_interval episodes
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

from tsn_sim import TSNSchedulingEnv, SACAgent, ReplayBuffer, SimulationConfig


# =====================================================================
#  Training Loop
# =====================================================================

def train(
    env: TSNSchedulingEnv,
    agent: SACAgent,
    buffer: ReplayBuffer,
    *,
    episodes: int = 2000,
    warmup_steps: int = 1000,
    batch_size: int = 128,
    update_every: int = 1,
    eval_interval: int = 50,
    save_interval: int = 200,
    save_dir: str = "checkpoints",
    log_interval: int = 10,
    verbose: bool = True,
    patience: int = 5,
    min_episodes: int = 100,
    alpha_floor: float = 0.02,
    alpha_collapse_patience: int = 3,
) -> Dict[str, List]:
    """Run SAC training loop with early stopping.

    Early stopping triggers when either:
    - eval eff_peak hasn't improved for `patience` consecutive evals (after min_episodes)
    - alpha stays below `alpha_floor` for `alpha_collapse_patience` consecutive evals

    Returns a dict of training curves for plotting.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Training curves
    history = defaultdict(list)
    total_steps = 0
    best_eval_eff_peak = float("inf")
    no_improve_count = 0
    alpha_collapse_count = 0
    early_stopped = False

    for ep in range(1, episodes + 1):
        ep_start = time.time()
        state, info = env.reset(seed=None)
        mask = env.action_masks()
        ep_reward = 0.0
        ep_steps = 0
        dropped = 0
        ep_entropy = 0.0
        ep_q_value = 0.0
        ep_update_count = 0

        while True:
            # --- Action selection ---
            if total_steps < warmup_steps:
                # Warmup: random valid action
                valid = np.where(mask)[0]
                if len(valid) > 0:
                    action = int(np.random.choice(valid))
                else:
                    action = 0
            else:
                # SAC policy
                action = agent.select_action(state, mask, deterministic=False)

            # --- Step ---
            next_state, reward, done, truncated, next_info = env.step(action)
            next_mask = env.action_masks()
            ep_reward += reward
            ep_steps += 1
            total_steps += 1

            if action == 0 and not mask.any():
                dropped += 1

            # --- Store transition ---
            buffer.push(_make_transition(
                state, action, reward, next_state, done, mask, next_mask,
            ))

            # --- Gradient update ---
            if total_steps >= warmup_steps and total_steps % update_every == 0:
                metrics = agent.update(buffer, batch_size)
                ep_entropy += metrics.get("entropy", 0.0)
                ep_q_value += metrics.get("q1_value", 0.0)
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
        history["alpha"].append(agent.alpha)

        # Diagnostics: average entropy and Q-value over the episode
        avg_entropy = ep_entropy / max(1, ep_update_count)
        avg_q = ep_q_value / max(1, ep_update_count)
        history["entropy"].append(avg_entropy)
        history["q_value"].append(avg_q)

        if ep % log_interval == 0 and verbose:
            # Recent averages
            n = min(log_interval, len(history["ep_reward"]))
            avg_reward = np.mean(history["ep_reward"][-n:])
            avg_peak = np.mean(history["ep_eff_peak"][-n:])
            avg_raw = np.mean(history["ep_peak"][-n:])
            avg_drop = np.mean(history["ep_drop"][-n:])
            avg_delay = np.mean(history["ep_delay"][-n:])
            avg_ent = np.mean(history["entropy"][-n:])
            avg_qval = np.mean(history["q_value"][-n:])
            ep_time = time.time() - ep_start
            print(
                f"Ep {ep:>5d}/{episodes}  "
                f"R={avg_reward:>7.3f}  "
                f"eff_peak={avg_peak:.4f}  "
                f"drop={avg_drop:.3f}  "
                f"a={agent.alpha:.3f}  "
                f"H={avg_ent:.3f}  "
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
                save_path = os.path.join(save_dir, "sac_best.pth")
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

            # --- Early stopping: alpha collapse ---
            if agent.alpha < alpha_floor:
                alpha_collapse_count += 1
                if verbose:
                    print(f"  [WARN] alpha={agent.alpha:.4f} < {alpha_floor} "
                          f"({alpha_collapse_count}/{alpha_collapse_patience})")
            else:
                alpha_collapse_count = 0

            # --- Trigger early stopping ---
            if ep >= min_episodes:
                if no_improve_count >= patience:
                    if verbose:
                        print(f"\n  [EARLY STOP] No improvement for "
                              f"{patience} evals. Stopping at ep={ep}.")
                    early_stopped = True
                elif alpha_collapse_count >= alpha_collapse_patience:
                    if verbose:
                        print(f"\n  [EARLY STOP] Alpha collapsed below "
                              f"{alpha_floor} for {alpha_collapse_patience} "
                              f"consecutive evals. Stopping at ep={ep}.")
                    early_stopped = True

            if early_stopped:
                break

        # --- Periodic save ---
        if ep % save_interval == 0:
            save_path = os.path.join(save_dir, f"sac_ep{ep}.pth")
            agent.save(save_path)

    # Final save
    final_path = os.path.join(save_dir, "sac_checkpoint.pth")
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
    agent: SACAgent,
    num_episodes: int = 10,
) -> Dict[str, float]:
    """Run greedy evaluation episodes."""
    eff_peaks = []
    raw_peaks = []
    drops = []
    delays = []
    rewards = []

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
    """Create a Transition for the replay buffer."""
    from tsn_sim.sac import Transition
    return Transition(
        state=state.astype(np.float32),
        action=int(action),
        reward=float(reward),
        next_state=next_state.astype(np.float32),
        done=bool(done),
        action_mask=mask.copy(),
        next_mask=next_mask.copy(),
    )


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="SAC training for 5G-TSN scheduling")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha-init", type=float, default=0.1)
    parser.add_argument("--target-entropy-ratio", type=float, default=0.1)
    parser.add_argument("--max-alpha", type=float, default=10.0)
    parser.add_argument("--auto-alpha", action="store_true",
                        help="Enable automatic alpha tuning (default: fixed alpha=0.1)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--shaping", action="store_true",
                        help="Enable step-level reward shaping (legacy mode)")
    parser.add_argument("--shaping-weight", type=float, default=0.01)
    parser.add_argument("--reward-mode", type=str, default="load_balance",
                        choices=["terminal", "shaping", "exponential", "load_balance", "mixed"],
                        help="terminal: only terminal reward; "
                             "shaping: terminal + weak step shaping; "
                             "exponential: delta*exp(-zeta*peak) per step + terminal; "
                             "load_balance: -cell_load per step + terminal (dense, strongly correlated); "
                             "mixed: -cell_load - peak_increase per step + terminal (local + global)")
    parser.add_argument("--obs-mode", type=str, default="full",
                        choices=["full", "compact"],
                        help="full: 199-dim; compact: 111-dim (channel stats + global stats)")
    parser.add_argument("--reward-delta", type=float, default=0.1,
                        help="Scale of exponential per-step reward")
    parser.add_argument("--reward-zeta", type=float, default=3.0,
                        help="Sensitivity of exponential reward to peak load")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stop after N evals without improvement")
    parser.add_argument("--min-episodes", type=int, default=100,
                        help="Minimum episodes before early stopping")
    parser.add_argument("--alpha-floor", type=float, default=0.02,
                        help="Alpha below this = exploration dead")
    parser.add_argument("--alpha-collapse-patience", type=int, default=3,
                        help="Consecutive evals with alpha < floor before stop")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--period-mode", type=str, default="cyclic",
                        choices=["cyclic", "simple", "random"],
                        help="cyclic=历史基线复现; simple=Simple集(2固定周期); random=Random集")
    parser.add_argument("--order-mode", type=str, default="random",
                        choices=["edf", "random"],
                        help="edf=固定EDF顺序(历史); random=随机顺序(在基线上进行,默认)")
    parser.add_argument("--multi-scenario", action="store_true",
                        help="Train with diverse scenarios: each episode uses seed+ep_count")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    # --- Build env ---
    sim_config = SimulationConfig(seed=args.seed, period_mode=args.period_mode)
    # Map reward-mode → env reward_mode + legacy shaping flag
    if args.reward_mode == "shaping":
        env_reward_mode = "terminal"
        env_shaping = True
    elif args.reward_mode == "exponential":
        env_reward_mode = "exponential"
        env_shaping = False
    elif args.reward_mode == "load_balance":
        env_reward_mode = "load_balance"
        env_shaping = False
    elif args.reward_mode == "mixed":
        env_reward_mode = "mixed"
        env_shaping = False
    else:  # terminal
        env_reward_mode = "terminal"
        env_shaping = args.shaping

    env = TSNSchedulingEnv(
        config=sim_config,
        use_step_shaping=env_shaping,
        shaping_weight=args.shaping_weight,
        reward_mode=env_reward_mode,
        obs_mode=args.obs_mode,
        reward_delta=args.reward_delta,
        reward_zeta=args.reward_zeta,
        order_mode=args.order_mode,
        multi_scenario=args.multi_scenario,
    )

    # --- Build agent ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha_init=args.alpha_init,
        auto_alpha=args.auto_alpha,
        target_entropy_ratio=args.target_entropy_ratio,
        max_alpha=args.max_alpha,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"{'='*60}")
    print(f"SAC Training for 5G-TSN Scheduling")
    print(f"{'='*60}")
    print(f"  Obs dim:       {env.observation_space.shape[0]}")
    print(f"  Action dim:    {env.action_space.n}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Warmup steps:  {args.warmup}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Buffer size:   {args.buffer_size}")
    print(f"  LR:            {args.lr}")
    print(f"  Gamma:         {args.gamma}")
    print(f"  Tau:           {args.tau}")
    print(f"  Alpha init:    {args.alpha_init}")
    print(f"  Target H ratio:{args.target_entropy_ratio}")
    print(f"  Hidden dims:   {args.hidden_dims}")
    print(f"  Step shaping:  {env_shaping}")
    print(f"  Reward mode:   {env_reward_mode}")
    print(f"  Obs mode:      {args.obs_mode} (dim={env.observation_space.shape[0]})")
    if env_reward_mode == "exponential":
        print(f"  Reward delta:  {args.reward_delta}")
        print(f"  Reward zeta:   {args.reward_zeta}")
    print(f"  Seed:          {args.seed}")
    print(f"  Multi-scenario: {args.multi_scenario}")
    print(f"  Early stop:    {'OFF' if args.no_early_stop else f'patience={args.patience}, alpha_floor={args.alpha_floor}'}")
    print(f"{'='*60}")

    # --- Eval only ---
    if args.eval_only:
        if args.model:
            agent.load(args.model)
            print(f"Loaded model from {args.model}")
        eval_metrics = evaluate(env, agent, num_episodes=20)
        print(f"\nEvaluation (20 episodes):")
        for k, v in eval_metrics.items():
            print(f"  {k:>16s}: {v:.4f}")
        return

    # --- Train ---
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
        alpha_floor=args.alpha_floor,
        alpha_collapse_patience=args.alpha_collapse_patience,
    )

    # --- Save training curves ---
    curves_path = os.path.join(args.save_dir, "training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining curves saved to {curves_path}")

    # --- Final evaluation ---
    print(f"\n{'='*60}")
    print("Final Evaluation (20 episodes, greedy policy)")
    print(f"{'='*60}")
    eval_metrics = evaluate(env, agent, num_episodes=20)
    for k, v in eval_metrics.items():
        print(f"  {k:>16s}: {v:.4f}")

    # --- Compare with the TRUE naive baseline (random_feasible) ---
    # NOTE: the real baseline is random_feasible (random order + random
    # placement), NOT the handcrafted min-load heuristic. When the env uses
    # order_mode='random', DRL is evaluated "on the baseline" and its gain is
    # attributable to RL, not to a hand-picked EDF ordering.
    print(f"\n{'='*60}")
    print("Comparison (same scenario, same seed)")
    print(f"{'='*60}")
    from tsn_sim.heuristics import schedule_with_heuristic
    from tsn_sim.config import HeuristicConfig
    scenario = build_scenario(sim_config)
    base = schedule_with_heuristic(
        scenario, seed=args.seed,
        heuristic=HeuristicConfig(strategy="random_feasible"),
    )
    base_peak = base.metrics["effective_peak_load"]
    sac_peak = eval_metrics["eff_peak_mean"]
    improvement = (base_peak - sac_peak) / base_peak * 100
    print(f"  Baseline (random_feasible):  eff_peak = {base_peak:.4f}")
    print(f"  SAC (greedy, 20 ep):         eff_peak = {sac_peak:.4f}")
    print(f"  Improvement vs baseline:     {improvement:+.2f}%")


if __name__ == "__main__":
    main()
