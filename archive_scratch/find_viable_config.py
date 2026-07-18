"""Test different packet_size configurations to find a viable 50-flow setup."""
from tsn_sim.config import SimulationConfig, HeuristicConfig
from tsn_sim.scenario import build_scenario
from tsn_sim.candidate import instantiate_packets, build_candidates
from tsn_sim.heuristics import schedule_with_heuristic

# Try different packet_size ranges with 50 flows
configs = {
    "A: 50流+原始size(800-9600)": SimulationConfig(
        seed=42, flow_count=50,
        periods_ms=(2, 4, 8, 16, 32),
        packet_size_bits=(800, 1200, 2400, 4800, 9600),
    ),
    "B: 50流+缩窄size(800-4800)": SimulationConfig(
        seed=42, flow_count=50,
        periods_ms=(2, 4, 8, 16, 32),
        packet_size_bits=(800, 1600, 3200, 4800),
    ),
    "C: 50流+小size(400-3200)": SimulationConfig(
        seed=42, flow_count=50,
        periods_ms=(2, 4, 8, 16, 32),
        packet_size_bits=(400, 800, 1600, 3200),
    ),
    "D: 50流+均匀size(800-2400)": SimulationConfig(
        seed=42, flow_count=50,
        periods_ms=(2, 4, 8, 16, 32),
        packet_size_bits=(800, 1200, 1800, 2400),
    ),
    "E: 30流+原始size(800-9600)": SimulationConfig(
        seed=42, flow_count=30,
        periods_ms=(2, 4, 8, 16, 32),
        packet_size_bits=(800, 1200, 2400, 4800, 9600),
    ),
}

print("=" * 80)
print("Finding viable 50-flow configuration")
print("=" * 80)

for name, cfg in configs.items():
    sc = build_scenario(cfg)
    pkts = instantiate_packets(sc)
    cands, order = build_candidates(sc, pkts)
    
    heu = HeuristicConfig(strategy="edf_min_load")
    result = schedule_with_heuristic(sc, heu, seed=42)
    m = result.metrics
    
    sched = int(m["scheduled_packet_count"])
    infeas = int(m["infeasible_packet_count"])
    total = sched + infeas
    
    print(f"\n{name}")
    print(f"  Flows={len(sc.flows)}, Packets={len(pkts)}, Cells={len(sc.rb_capacity)}")
    print(f"  peak={m['peak_load']:.3f}, eff_peak={m['effective_peak_load']:.3f}")
    print(f"  deadline_sat={m['true_deadline_satisfaction_rate']:.2%}")
    print(f"  scheduled={sched}/{total}, drop={infeas}")
    print(f"  util={m['resource_utilization']:.2%}")
    print(f"  avg choices={sum(len(cands[k]) for k in cands)/len(cands):.1f}")
    print(f"  Episode steps={len(order)}")
    
    if m["peak_load"] < 0.95:
        print(f"  [VIABLE] peak < 0.95, suitable for DRL")
    elif m["peak_load"] >= 1.0:
        print(f"  [OVERLOADED] peak >= 1.0, not suitable")
    else:
        print(f"  [BORDERLINE] 0.95 <= peak < 1.0")
