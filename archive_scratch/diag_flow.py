"""Concrete flow trace: heuristic edf_min_load vs trained PPO greedy, step by step.

目的：让用户看到调度流程到底怎么跑的，以及 PPO 在哪一步开始偏离 min-load 启发式。
场景：simple 集 + 50 流 + seed=42（与训练一致），只打印前若干步。
"""
import math
import numpy as np

from tsn_sim import SimulationConfig, TSNSchedulingEnv
from tsn_sim.heuristics import schedule_with_heuristic
from tsn_sim.ppo import PPOAgent


def build_env():
    cfg = SimulationConfig(seed=42, period_mode="simple")
    env = TSNSchedulingEnv(cfg, reward_mode="load_balance")
    return env


def min_load_feasible_cell(env, packet_key):
    """复刻 edf_min_load 的选法：在可行格里挑 post-load 最小的。"""
    packet = env._packets[packet_key]
    flow = env._flows[packet.flow_id]
    mask = env.action_masks()
    best = None
    best_load = float("inf")
    for a in np.where(mask)[0]:
        link = a // env.S + 1
        slot = a % env.S
        bprb = env.scenario.rb_bits[(flow.flow_id, link, slot)]
        req = math.ceil(flow.packet_size_bits / bprb)
        cap = env.scenario.rb_capacity[(link, slot)]
        post = (env._used_rb[(link, slot)] + req) / cap
        if post < best_load:
            best_load = post
            best = (a, link, slot, post)
    return best


def run_heuristic():
    cfg = SimulationConfig(seed=42, period_mode="simple")
    env = TSNSchedulingEnv(cfg, reward_mode="load_balance")
    res = schedule_with_heuristic(env.scenario, seed=42)
    print(f"[启发式 edf_min_load] eff_peak = {res.objective:.4f}  丢弃 = {res.metrics['infeasible_packet_count']:.0f}")
    return res.objective


def run_agent_trace(agent, n_show=15):
    env = build_env()
    obs, _ = env.reset()
    print(f"\n[PPO 贪婪策略] 逐步追踪（前 {n_show} 步）")
    print(f"{'步':>3} | {'PPO选格(link,slot)':>18} | {'PPO.post_load':>13} | {'启发式min-load格':>18} | {'min.post_load':>12} | {'当前峰值':>8}")
    print("-" * 95)
    steps = 0
    while True:
        packet_key = env.current_packet
        if packet_key is None:
            break
        mask = env.action_masks()
        a, _, _ = agent.select_action(obs, mask, deterministic=True)
        link = a // env.S + 1
        slot = a % env.S
        # PPO 选中后的 post-load
        flow = env._flows[packet_key[0]]
        bprb = env.scenario.rb_bits[(flow.flow_id, link, slot)]
        req = math.ceil(flow.packet_size_bits / bprb)
        cap = env.scenario.rb_capacity[(link, slot)]
        ppo_post = (env._used_rb[(link, slot)] + req) / cap

        # 启发式会选的格子
        mlf = min_load_feasible_cell(env, packet_key)
        if mlf is None:
            print(f"{steps:>3} | (无可行动作)")
            break
        m_a, m_link, m_slot, m_post = mlf

        diverge = "  <-- 不同" if (link, slot) != (m_link, m_slot) else ""
        if steps < n_show:
            print(f"{steps:>3} | ({link},{slot})".rjust(3) + f" | {ppo_post:>13.3f} | ({m_link},{m_slot})".rjust(18) +
                  f" | {m_post:>12.3f} | {env._compute_current_peak():>8.3f}{diverge}")

        obs, r, done, _, info = env.step(a)
        steps += 1
        if done:
            break

    final = info.get("effective_peak_load", float("nan"))
    print(f"\n[PPO 贪婪] 总步数 = {steps}, eff_peak = {final:.4f}")
    return final


def main():
    h_peak = run_heuristic()
    # 加载训练好的 PPO（simple 集 best checkpoint）
    cfg = SimulationConfig(seed=42, period_mode="simple")
    probe = TSNSchedulingEnv(cfg, reward_mode="load_balance")
    agent = PPOAgent(obs_dim=probe.observation_space.shape[0], n_actions=probe.action_space.n, device="cpu")
    try:
        agent.load("checkpoints_ppo_simple/ppo_best.pth")
        print("\n(已加载 checkpoints_ppo_simple/ppo_best.pth)")
    except Exception as e:
        print(f"\n(未找到 best 模型，用随机初始化: {e})")
    a_peak = run_agent_trace(agent, n_show=15)
    print(f"\n=== 对比 ===")
    print(f"启发式 eff_peak = {h_peak:.4f}")
    print(f"PPO    eff_peak = {a_peak:.4f}")
    print(f"差距 = {(a_peak - h_peak):+.4f}  (正值=PPO 更差)")


if __name__ == "__main__":
    main()
