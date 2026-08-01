"""临时验证：SAC dimred 端到端训练不崩。跑完即删。"""
import os
import shutil
import numpy as np
import torch
from tsn_sim import TSNSchedulingEnv, SACAgent, ReplayBuffer, SimulationConfig
from train_sac import train, evaluate

TMP = "_tmp_sac_dimred"

try:
    cfg = SimulationConfig(seed=42, period_mode="simple")
    env = TSNSchedulingEnv(
        cfg, order_mode="random", action_mode="dimred", reward_mode="load_balance",
    )
    print("n_actions:", env.n_actions, "action_space:", env.action_space)
    assert env.n_actions == 7

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0], n_actions=env.n_actions,
        action_mode="dimred", gamma=0.5, alpha_init=0.1,
        target_entropy_ratio=0.1, device=device,
    )
    buffer = ReplayBuffer(capacity=20000)
    hist = train(
        env, agent, buffer, episodes=12, warmup_steps=150, batch_size=64,
        eval_interval=6, save_dir=TMP, log_interval=3,
        patience=9999, min_episodes=5,
    )
    ev = hist.get("eval_eff_peak", [])
    print("eval_eff_peak history:", ev)
    assert len(ev) > 0, "no eval recorded"
    assert all(np.isfinite(x) for x in ev), "eval has NaN/Inf"

    em = evaluate(env, agent, num_episodes=5)
    print("EVAL:", {k: round(v, 4) for k, v in em.items()})
    assert np.isfinite(em["eff_peak_mean"])

    # 检查一个训练后的 checkpoint 能 load
    agent.save(os.path.join(TMP, "sac_best.pth"))
    agent2 = SACAgent(
        obs_dim=env.observation_space.shape[0], n_actions=7,
        action_mode="dimred", gamma=0.5, device=device,
    )
    agent2.load(os.path.join(TMP, "sac_best.pth"))
    em2 = evaluate(env, agent2, num_episodes=3)
    print("RELOAD EVAL:", {k: round(v, 4) for k, v in em2.items()})

    print("SAC DIMRED TRAIN TEST PASSED")
finally:
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
        print("cleaned", TMP)
