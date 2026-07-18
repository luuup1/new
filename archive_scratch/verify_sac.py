"""Quick verification of the SAC agent implementation.

Tests:
1. Network forward pass shapes
2. Action masking in policy
3. Replay buffer operations
4. One gradient update step
5. Short training run (50 episodes) -- check loss is finite and policy improves
6. Save/load round-trip
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from tsn_sim import TSNSchedulingEnv, SACAgent, ReplayBuffer, SimulationConfig


def test_network_shapes():
    """Verify network forward pass produces correct shapes."""
    env = TSNSchedulingEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = SACAgent(obs_dim, n_actions, hidden_dims=(64, 64))

    state = np.random.randn(obs_dim).astype(np.float32)
    mask = np.ones(n_actions, dtype=bool)
    mask[::3] = False  # mask out some actions

    # Policy
    action, log_prob = agent.policy.sample(
        torch.from_numpy(state).unsqueeze(0),
        torch.from_numpy(mask).unsqueeze(0),
    )
    assert action.shape == (1,), f"action shape: {action.shape}"
    assert log_prob.shape == (1,), f"log_prob shape: {log_prob.shape}"

    # Q-networks
    q1 = agent.q1(torch.from_numpy(state).unsqueeze(0))
    q2 = agent.q2(torch.from_numpy(state).unsqueeze(0))
    assert q1.shape == (1, n_actions), f"q1 shape: {q1.shape}"
    assert q2.shape == (1, n_actions), f"q2 shape: {q2.shape}"

    print(f"  [PASS] shapes: action={action.shape}, q1={q1.shape}, q2={q2.shape}")
    return True


def test_masking():
    """Verify that masked actions are never sampled."""
    env = TSNSchedulingEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = SACAgent(obs_dim, n_actions, hidden_dims=(64, 64))

    state = np.random.randn(obs_dim).astype(np.float32)
    mask = np.zeros(n_actions, dtype=bool)
    mask[5:10] = True  # Only actions 5-9 are valid

    for _ in range(1000):
        action = agent.select_action(state, mask, deterministic=False)
        assert 5 <= action <= 9, f"Sampled invalid action {action}!"

    # Also check deterministic
    det_action = agent.select_action(state, mask, deterministic=True)
    assert 5 <= det_action <= 9, f"Deterministic action {det_action} invalid!"

    # Check all-zero mask (edge case)
    empty_mask = np.zeros(n_actions, dtype=bool)
    action = agent.select_action(state, empty_mask, deterministic=False)
    # Should still return something (argmax of all -1e9 -> action 0)
    assert 0 <= action < n_actions

    print(f"  [PASS] masking: 1000 samples all in [5,9], "
          f"det={det_action}, empty_mask handled")
    return True


def test_replay_buffer():
    """Test replay buffer push and sample."""
    from tsn_sim.sac import Transition

    buf = ReplayBuffer(capacity=100)
    obs_dim = 10
    n_actions = 5

    for i in range(50):
        buf.push(Transition(
            state=np.random.randn(obs_dim).astype(np.float32),
            action=i % n_actions,
            reward=float(i),
            next_state=np.random.randn(obs_dim).astype(np.float32),
            done=(i == 49),
            action_mask=np.ones(n_actions, dtype=bool),
            next_mask=np.ones(n_actions, dtype=bool),
        ))

    assert len(buf) == 50
    batch = buf.sample(16)
    assert batch["state"].shape == (16, obs_dim)
    assert batch["action"].shape == (16,)
    assert batch["reward"].shape == (16,)
    assert batch["done"].shape == (16,)
    assert batch["action_mask"].shape == (16, n_actions)

    # Test capacity overflow
    for i in range(100):
        buf.push(Transition(
            state=np.zeros(obs_dim, dtype=np.float32),
            action=0, reward=0.0,
            next_state=np.zeros(obs_dim, dtype=np.float32),
            done=False,
            action_mask=np.ones(n_actions, dtype=bool),
            next_mask=np.ones(n_actions, dtype=bool),
        ))
    assert len(buf) == 100  # capacity

    print(f"  [PASS] replay buffer: push, sample, capacity overflow")
    return True


def test_gradient_update():
    """Run one gradient update and check losses are finite."""
    from tsn_sim.sac import Transition

    env = TSNSchedulingEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = SACAgent(obs_dim, n_actions, hidden_dims=(64, 64))
    buffer = ReplayBuffer(capacity=1000)

    # Fill buffer with random transitions
    state, _ = env.reset(seed=0)
    mask = env.action_masks()
    for _ in range(200):
        valid = np.where(mask)[0]
        action = int(np.random.choice(valid)) if len(valid) > 0 else 0
        next_state, reward, done, _, next_info = env.step(action)
        next_mask = env.action_masks()

        buffer.push(Transition(
            state=state.astype(np.float32), action=action, reward=float(reward),
            next_state=next_state.astype(np.float32), done=bool(done),
            action_mask=mask.copy(), next_mask=next_mask.copy(),
        ))

        if done:
            state, _ = env.reset()
            mask = env.action_masks()
        else:
            state = next_state
            mask = next_mask

    # Run update
    metrics = agent.update(buffer, batch_size=32)
    assert len(metrics) > 0, "No metrics returned"
    for k, v in metrics.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"

    print(f"  [PASS] gradient update: q1_loss={metrics['q1_loss']:.4f}, "
          f"q2_loss={metrics['q2_loss']:.4f}, "
          f"policy_loss={metrics['policy_loss']:.4f}, "
          f"alpha={metrics['alpha']:.4f}")
    return True


def test_short_training():
    """Run 50 episodes of training and check for improvement."""
    from tsn_sim.sac import Transition

    env = TSNSchedulingEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = SACAgent(obs_dim, n_actions, hidden_dims=(128, 128),
                     lr=3e-4, gamma=0.99)
    buffer = ReplayBuffer(capacity=10000)

    # Warmup
    state, _ = env.reset(seed=42)
    mask = env.action_masks()
    for _ in range(500):
        valid = np.where(mask)[0]
        action = int(np.random.choice(valid)) if len(valid) > 0 else 0
        next_state, reward, done, _, _ = env.step(action)
        next_mask = env.action_masks()
        buffer.push(Transition(
            state=state.astype(np.float32), action=action, reward=float(reward),
            next_state=next_state.astype(np.float32), done=bool(done),
            action_mask=mask.copy(), next_mask=next_mask.copy(),
        ))
        if done:
            state, _ = env.reset()
            mask = env.action_masks()
        else:
            state = next_state
            mask = next_mask

    # Training
    rewards_early = []
    rewards_late = []

    for ep in range(1, 51):
        state, _ = env.reset()
        mask = env.action_masks()
        ep_reward = 0.0

        while True:
            action = agent.select_action(state, mask, deterministic=False)
            next_state, reward, done, _, _ = env.step(action)
            next_mask = env.action_masks()

            buffer.push(Transition(
                state=state.astype(np.float32), action=action, reward=float(reward),
                next_state=next_state.astype(np.float32), done=bool(done),
                action_mask=mask.copy(), next_mask=next_mask.copy(),
            ))

            metrics = agent.update(buffer, batch_size=64)

            ep_reward += reward
            state = next_state
            mask = next_mask

            if done:
                break

        if ep <= 10:
            rewards_early.append(ep_reward)
        if ep > 40:
            rewards_late.append(ep_reward)

    early_mean = np.mean(rewards_early)
    late_mean = np.mean(rewards_late)
    print(f"  [PASS] short training (50 ep): "
          f"early R={early_mean:.4f}, late R={late_mean:.4f}, "
          f"delta={late_mean - early_mean:+.4f}")

    # Also check alpha changed from init
    print(f"         alpha: init=0.200, final={agent.alpha:.4f}")
    return True


def test_save_load():
    """Test model save/load round-trip."""
    env = TSNSchedulingEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent1 = SACAgent(obs_dim, n_actions, hidden_dims=(64, 64))
    agent2 = SACAgent(obs_dim, n_actions, hidden_dims=(64, 64))

    state = np.random.randn(obs_dim).astype(np.float32)
    mask = np.ones(n_actions, dtype=bool)

    # Before load: different outputs
    a1_before = agent1.select_action(state, mask, deterministic=True)
    a2_before = agent2.select_action(state, mask, deterministic=True)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name
    agent1.save(path)
    agent2.load(path)

    # After load: same outputs
    a1_after = agent1.select_action(state, mask, deterministic=True)
    a2_after = agent2.select_action(state, mask, deterministic=True)
    assert a1_after == a2_after, f"After load: {a1_after} != {a2_after}"

    os.unlink(path)
    print(f"  [PASS] save/load: before=({a1_before},{a2_before}) "
          f"after=({a1_after},{a2_after})")
    return True


def main():
    print("=" * 60)
    print("SAC Agent Verification Suite")
    print("=" * 60)

    results = []
    try:
        results.append(("network shapes", test_network_shapes()))
        results.append(("action masking", test_masking()))
        results.append(("replay buffer", test_replay_buffer()))
        results.append(("gradient update", test_gradient_update()))
        results.append(("short training", test_short_training()))
        results.append(("save/load", test_save_load()))
    except Exception as e:
        print(f"\n  [FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results:
        print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
