"""Quick BC ceiling check (no PPO fine-tune).

Trains the actor with BC at several epoch budgets and reports:
  - greedy_match: fraction of demo steps where argmax == teacher action
  - BC-only eff_peak: greedy rollout on the fixed eval env
Goal: confirm how high BC alone can reach vs the teacher (0.7073).
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from tsn_sim import TSNSchedulingEnv, PPOAgent, SimulationConfig
from tsn_sim.bc import generate_demonstrations, bc_train


def greedy_eval(env, agent, n=20):
    effs = []
    for _ in range(n):
        s, _ = env.reset()
        while True:
            a, _, _ = agent.select_action(s, env.action_masks(), deterministic=True)
            s, _, done, _, info = env.step(a)
            if done:
                break
        effs.append(info["effective_peak_load"])
    return float(np.mean(effs))


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "simple"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    sim = SimulationConfig(seed=seed, period_mode=period)
    demo_env = TSNSchedulingEnv(
        config=sim, order_mode="random", reward_mode="mixed",
        obs_mode="full", multi_scenario=True,
    )
    demo = generate_demonstrations(demo_env, 300)
    print(f"demo samples: {len(demo['action'])}   teacher(random-order min-load)=0.7073")

    eval_env = TSNSchedulingEnv(
        config=sim, order_mode="random", reward_mode="mixed", obs_mode="full",
    )

    for epochs in [30, 60, 100, 150]:
        agent = PPOAgent(
            obs_dim=eval_env.observation_space.shape[0],
            n_actions=eval_env.action_space.n, device="cuda",
        )
        stats = bc_train(agent, demo, epochs=epochs, lr=1e-3, verbose=False)
        eff = greedy_eval(eval_env, agent)
        print(f"  BC epochs={epochs:>3d}  greedy_match={stats['greedy_match']:.3f}  "
              f"BC-only eff_peak={eff:.4f}  (gap to teacher={eff-0.7073:+.4f})")
