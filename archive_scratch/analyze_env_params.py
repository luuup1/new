"""Quick diagnostic: compare DRL learnability across parameter configurations."""

import math
from tsn_sim.config import SimulationConfig, HeuristicConfig
from tsn_sim.scenario import build_scenario
from tsn_sim.candidate import instantiate_packets
from tsn_sim.heuristics import schedule_with_heuristic
from tsn_sim.milp import solve_optimal_milp

configs = {
    "A: 20流+32slot(当前)": SimulationConfig(seed=42, flow_count=20, periods_ms=(2,4,8,16,32)),
    "B: 50流+32slot": SimulationConfig(seed=42, flow_count=50, periods_ms=(2,4,8,16,32)),
    "C: 50流+16slot": SimulationConfig(seed=42, flow_count=50, periods_ms=(4,8,16)),
    "D: 50流+8slot": SimulationConfig(seed=42, flow_count=50, periods_ms=(4,8)),
    "E: 100流+16slot": SimulationConfig(seed=42, flow_count=100, periods_ms=(4,8,16)),
}

print("=" * 80)
print("DRL Learnability Diagnostic: Parameter Configuration Comparison")
print("=" * 80)

for name, cfg in configs.items():
    sc = build_scenario(cfg)
    pkts = instantiate_packets(sc)
    n_pkts = len(pkts)
    n_cells = cfg.link_count * sc.hyperperiod_ms
    action_dim = n_cells
    
    # Heuristic baseline
    res = schedule_with_heuristic(sc, heuristic=HeuristicConfig(strategy='edf_min_load'), seed=42)
    m = res.metrics
    heu_peak = m['peak_load']
    heu_eff = m['effective_peak_load']
    
    # Count forced decisions (only 1-2 valid candidates)
    from tsn_sim.candidate import build_candidates
    cand_dict, order = build_candidates(sc, pkts)
    forced_count = 0
    total_candidates = 0
    for pkt_key in order:
        n_valid = len(cand_dict[pkt_key])
        total_candidates += n_valid
        if n_valid <= 2:
            forced_count += 1
    
    forced_ratio = forced_count / n_pkts if n_pkts > 0 else 0
    avg_candidates = total_candidates / n_pkts if n_pkts > 0 else 0
    
    # Link balance
    link_peaks = {}
    for link in sc.links:
        link_loads = []
        for slot in range(sc.hyperperiod_ms):
            cell_key = (link, slot)
            used = m.get('cell_loads', {}).get(cell_key, 0)
            cap = sc.rb_capacity[cell_key]
            link_loads.append(used / cap if cap > 0 else 0)
        link_peaks[link] = max(link_loads) if link_loads else 0
    
    # MILP (only for small problems)
    milp_peak = None
    if n_pkts <= 200:
        try:
            milp_res = solve_optimal_milp(sc, time_limit=60)
            if milp_res and milp_res.metrics:
                milp_peak = milp_res.metrics['peak_load']
        except Exception as e:
            milp_peak = None
    
    gap = ((heu_peak - milp_peak) / milp_peak * 100) if milp_peak is not None else None
    
    print(f"\n{'─' * 70}")
    print(f"  {name}")
    print(f"{'─' * 70}")
    print(f"  Hyperperiod:    {sc.hyperperiod_ms} ms")
    print(f"  Slots:          {sc.hyperperiod_ms}")
    print(f"  Action space:   {action_dim} (3×{sc.hyperperiod_ms})")
    print(f"  Packets/episode: {n_pkts}")
    print(f"  Forced steps:   {forced_count}/{n_pkts} = {forced_ratio:.1%}")
    print(f"  Avg candidates: {avg_candidates:.1f}")
    print(f"  Heuristic peak: {heu_peak:.4f}")
    print(f"  Heuristic eff:  {heu_eff:.4f}")
    print(f"  MILP peak:      {milp_peak if milp_peak else 'N/A (>60s)'}")
    print(f"  MILP gap:       {gap:.1f}%") if gap is not None else print(f"  MILP gap:       N/A")
    print(f"  Link peaks:     L1={link_peaks.get(1,0):.3f}  L2={link_peaks.get(2,0):.3f}  L3={link_peaks.get(3,0):.3f}")
    print(f"  Peak/avg ratio: {heu_peak / m['resource_utilization']:.1f}:1") if m['resource_utilization'] > 0 else None
    
    # DRL suitability score (heuristic)
    # Higher gap + lower forced ratio + moderate episode length = better for DRL
    if gap is not None:
        score = gap * (1 - forced_ratio) / (n_pkts / 100)
        print(f"  DRL suitability: {score:.2f} (gap×freedom/length)")

print(f"\n{'=' * 80}")
print("Recommendation: Choose config with highest DRL suitability score")
print("=" * 80)
