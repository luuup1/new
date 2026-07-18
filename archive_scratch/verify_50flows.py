"""Quick verification of 50-flow configuration."""
from tsn_sim.config import default_config
from tsn_sim.scenario import build_scenario
from tsn_sim.candidate import instantiate_packets, build_candidates
from tsn_sim.heuristics import schedule_with_heuristic
from tsn_sim.metrics import compute_metrics
from tsn_sim.config import HeuristicConfig

# Test with default seed=7 (as used in training)
sc = default_config()
hyper_slots = int(sc.periods_ms[-1] / sc.slot_ms)
print(f"Config: flow_count={sc.flow_count}, slots={hyper_slots}, links={sc.link_count}")
print(f"Packet sizes: {sc.packet_size_bits}")
print(f"Action space: {sc.link_count * hyper_slots}")

scenario = build_scenario(sc)
pkts = instantiate_packets(scenario)
print(f"Scenario: {len(scenario.flows)} flows, {len(pkts)} packet instances")

cands, order = build_candidates(scenario, pkts)
avg_choices = sum(len(cands[k]) for k in cands) / len(cands)
print(f"Candidates: {len(order)} packets, avg choices={avg_choices:.1f}")

strategies = [
    ("edf_min_load", HeuristicConfig(strategy="edf_min_load")),
    ("urgency_lexicographic", HeuristicConfig(strategy="urgency_lexicographic")),
    ("random_feasible", HeuristicConfig(strategy="random_feasible")),
]

for name, heu_cfg in strategies:
    result = schedule_with_heuristic(scenario, heu_cfg, seed=42)
    m = compute_metrics(scenario, result.schedule, pkts, result.infeasible_packets)
    peak = m["peak_load"]
    eff = m["effective_peak_load"]
    dsr = m["true_deadline_satisfaction_rate"]
    sched = int(m["scheduled_packet_count"])
    infeas = int(m["infeasible_packet_count"])
    total = sched + infeas
    print(f"{name}: peak={peak:.3f}, eff_peak={eff:.3f}, "
          f"deadline_sat={dsr:.2%}, scheduled={sched}/{total}")

print(f"\nEpisode steps (for DRL): {len(order)}")
print(f"Action space size: {sc.link_count * hyper_slots}")
