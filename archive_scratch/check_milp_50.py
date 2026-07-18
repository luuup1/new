"""MILP gap for 50-flow config with default seed=7."""
from tsn_sim.config import default_config
from tsn_sim.scenario import build_scenario
from tsn_sim.heuristics import schedule_with_heuristic
from tsn_sim.milp import solve_optimal_milp
from tsn_sim.config import HeuristicConfig

sc = default_config()
print(f"Config: flow_count={sc.flow_count}, packet_size_bits={sc.packet_size_bits}")

scenario = build_scenario(sc)

# Heuristic baselines
for strategy in ["edf_min_load", "urgency_lexicographic"]:
    heu = HeuristicConfig(strategy=strategy)
    result = schedule_with_heuristic(scenario, heu, seed=42)
    print(f"{strategy}: peak={result.metrics['peak_load']:.4f}, "
          f"eff_peak={result.metrics['effective_peak_load']:.4f}")

# MILP
print("Running MILP (time_limit=120s)...")
milp_result = solve_optimal_milp(scenario, time_limit_s=120)
if milp_result and milp_result.metrics and "peak_load" in milp_result.metrics:
    milp_peak = milp_result.metrics["peak_load"]
    print(f"MILP: peak={milp_peak:.4f}")
    heu_peak = 0.816  # edf_min_load from above
    gap = (heu_peak - milp_peak) / milp_peak * 100
    print(f"Gap (edf vs MILP): {gap:.1f}%")
else:
    print(f"MILP: no solution or timeout")
    if milp_result:
        print(f"MILP status: {milp_result.status}, message: {milp_result.message}")
