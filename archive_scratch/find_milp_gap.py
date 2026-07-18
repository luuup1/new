"""MILP gap comparison for viable 50-flow configurations."""
from tsn_sim.config import SimulationConfig, HeuristicConfig
from tsn_sim.scenario import build_scenario
from tsn_sim.candidate import instantiate_packets, build_candidates
from tsn_sim.heuristics import schedule_with_heuristic
from tsn_sim.milp import solve_optimal_milp

# Only test viable configs
configs = {
    "D: 50流+均匀size(800-2400)": SimulationConfig(
        seed=42, flow_count=50,
        periods_ms=(2, 4, 8, 16, 32),
        packet_size_bits=(800, 1200, 1800, 2400),
    ),
    "A: 50流+原始size(800-9600)": SimulationConfig(
        seed=42, flow_count=50,
        periods_ms=(2, 4, 8, 16, 32),
        packet_size_bits=(800, 1200, 2400, 4800, 9600),
    ),
}

for name, cfg in configs.items():
    sc = build_scenario(cfg)
    pkts = instantiate_packets(sc)
    cands, order = build_candidates(sc, pkts)
    
    # Heuristic baseline
    heu = HeuristicConfig(strategy="edf_min_load")
    heu_result = schedule_with_heuristic(sc, heu, seed=42)
    heu_peak = heu_result.metrics["peak_load"]
    
    # MILP
    print(f"\n{name}")
    print(f"  Heuristic peak = {heu_peak:.4f}")
    print(f"  Running MILP (time_limit=120s)...")
    milp_result = solve_optimal_milp(sc, time_limit_s=120)
    if milp_result and milp_result.metrics:
        milp_peak = milp_result.metrics["peak_load"]
        gap = (heu_peak - milp_peak) / milp_peak * 100
        print(f"  MILP peak = {milp_peak:.4f}")
        print(f"  MILP gap = {gap:.1f}%")
    else:
        print(f"  MILP: no solution or timeout")
