"""Verify multi-scenario mode works correctly."""
import sys
sys.path.insert(0, ".")

from tsn_sim import TSNSchedulingEnv, SimulationConfig
import numpy as np

def test_multi_scenario():
    print("=" * 60)
    print("Multi-Scenario Verification")
    print("=" * 60)

    config = SimulationConfig(seed=42)

    # --- Test 1: Fixed scenario (original behavior) ---
    print("\n--- Test 1: Fixed scenario (multi_scenario=False) ---")
    env_fixed = TSNSchedulingEnv(config=config, multi_scenario=False)
    obs1, _ = env_fixed.reset()
    obs2, _ = env_fixed.reset()
    obs3, _ = env_fixed.reset()

    # All 3 resets should produce identical first observation
    same1 = np.allclose(obs1, obs2)
    same2 = np.allclose(obs2, obs3)
    print(f"  reset 1 vs 2: identical = {same1}")
    print(f"  reset 2 vs 3: identical = {same2}")
    print(f"  Expected: True, True (same scenario every time)")

    # --- Test 2: Multi-scenario ---
    print("\n--- Test 2: Multi-scenario (multi_scenario=True) ---")
    env_multi = TSNSchedulingEnv(config=config, multi_scenario=True,
                                  reward_mode="exponential", obs_mode="compact")

    obs_list = []
    peak_list = []
    total_pkt_list = []

    for ep in range(1, 6):
        obs, info = env_multi.reset()
        obs_list.append(obs.copy())

        # Run episode with random valid actions
        total_r = 0
        steps = 0
        mask = env_multi.action_masks()
        while True:
            valid = np.where(mask)[0]
            a = int(np.random.choice(valid)) if len(valid) > 0 else 0
            obs, r, done, _, ep_info = env_multi.step(a)
            total_r += r
            steps += 1
            mask = env_multi.action_masks()
            if done:
                break

        peak_list.append(ep_info.get("peak_load", 0))
        total_pkt_list.append(env_multi._total_packets)

        print(f"  Ep {ep}: total_packets={env_multi._total_packets}, "
              f"peak={ep_info.get('peak_load', 0):.4f}, "
              f"eff_peak={ep_info.get('effective_peak_load', 0):.4f}, "
              f"obs_dim={len(obs_list[-1])}")

    # Check: different peaks across episodes (different scenarios)
    unique_peaks = len(set(round(p, 3) for p in peak_list))
    print(f"\n  Unique peak values across 5 eps: {unique_peaks}")
    print(f"  Expected: >1 (scenarios should differ)")

    # Check: packet count is constant
    pkt_counts = set(total_pkt_list)
    print(f"  Packet counts across 5 eps: {pkt_counts}")
    print(f"  Expected: {1} (always same count)")

    # Check: obs dimensions are constant
    obs_dims = set(len(o) for o in obs_list)
    print(f"  Obs dimensions across 5 eps: {obs_dims}")
    print(f"  Expected: {1} (always 111 for compact mode)")

    # Check: first obs differs across episodes (different scenarios → different grid at reset)
    diff_count = 0
    for i in range(len(obs_list)):
        for j in range(i+1, len(obs_list)):
            if not np.allclose(obs_list[i], obs_list[j]):
                diff_count += 1
    print(f"  Pairs of different initial obs: {diff_count}/{len(obs_list)*(len(obs_list)-1)//2}")
    print(f"  Expected: most pairs differ (different scenarios)")

    # --- Test 3: Reproducibility ---
    print("\n--- Test 3: Reproducibility (same seed → same scenario sequence) ---")
    env_multi2 = TSNSchedulingEnv(config=config, multi_scenario=True)
    obs_a, _ = env_multi2.reset()  # ep 1: seed=42+1=43
    obs_b, _ = env_multi2.reset()  # ep 2: seed=42+2=44

    env_multi3 = TSNSchedulingEnv(config=config, multi_scenario=True)
    obs_c, _ = env_multi3.reset()  # ep 1: seed=43 (same as env_multi2 ep1)
    obs_d, _ = env_multi3.reset()  # ep 2: seed=44 (same as env_multi2 ep2)

    repro1 = np.allclose(obs_a, obs_c)
    repro2 = np.allclose(obs_b, obs_d)
    print(f"  Ep1 reproducible: {repro1}")
    print(f"  Ep2 reproducible: {repro2}")
    print(f"  Expected: True, True (seed sequence is deterministic)")

    # --- Summary ---
    all_pass = (
        same1 and same2  # fixed mode
        and unique_peaks > 1  # multi mode produces different scenarios
        and len(pkt_counts) == 1  # packet count stable
        and len(obs_dims) == 1  # obs dim stable
        and repro1 and repro2  # reproducible
    )

    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED: {all_pass}")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_multi_scenario()
