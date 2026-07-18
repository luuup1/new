"""Physical config health check for 50-flow setting (seed=7, default config).

验证 period_mode 三种模式（cyclic / simple / random）下：
  - 超周期恒定 32（env 96 格动作空间不受影响）
  - 不过载(peak<1.0)、不丢包、deadline 可满足
  - Simple 集周期多样性低（默认 2 个固定值），Random 集多样性高
"""
import numpy as np
from tsn_sim.config import default_config, HeuristicConfig
from tsn_sim.scenario import build_scenario
from tsn_sim.candidate import instantiate_packets
from tsn_sim.heuristics import schedule_with_heuristic

MODES = ["cyclic", "simple", "random"]
sc0 = default_config()
L = sc0.link_count
S = int(sc0.periods_ms[-1] / sc0.slot_ms)  # = lcm = 32
print("=== Grid (fixed) ===")
print(f"  Links={L}, Slots(hyperperiod)={S}, Cells={L*S}, slot_ms={sc0.slot_ms}")
print(f"  RB base by link = {sc0.rb_base_by_link}, packet_sizes = {sc0.packet_size_bits}")
print(f"  periods_ms(full set) = {sc0.periods_ms}, simple_periods = {sc0.simple_periods}\n")

summary = []
for mode in MODES:
    sc = default_config(period_mode=mode, seed=sc0.seed)
    scenario = build_scenario(sc)
    distinct_periods = sorted({f.period_ms for f in scenario.flows})
    caps = np.array([scenario.rb_capacity[(l, s)] for l in range(1, L + 1) for s in range(S)])
    total_cap = int(caps.sum())

    pkts = instantiate_packets(scenario)
    result = schedule_with_heuristic(scenario, HeuristicConfig(strategy="edf_min_load"))
    sched = result.schedule
    infeas = result.infeasible_packets
    used = {}
    for e in sched:
        used[(e.link, e.slot)] = used.get((e.link, e.slot), 0) + e.required_rb
    cell_loads = np.array(
        [used.get((l, s), 0) / scenario.rb_capacity[(l, s)]
         for l in range(1, L + 1) for s in range(S)]
    )
    peak = float(cell_loads.max())
    avg = float(cell_loads.mean())
    total_used = int(sum(e.required_rb for e in sched))
    util = total_used / total_cap

    per_link = ", ".join(
        f"L{l}={cell_loads[l-1::L].max():.3f}" for l in range(1, L + 1)
    )
    ok = (peak < 1.0) and (len(infeas) == 0)
    print(f"=== [{mode}] ===")
    print(f"  distinct periods = {distinct_periods}  (count={len(distinct_periods)})")
    print(f"  Flow={len(scenario.flows)}, Packet instances={len(pkts)}")
    print(f"  Scheduled={len(sched)}, Dropped(infeasible)={len(infeas)}")
    print(f"  Peak={peak:.3f}, Avg={avg:.3f}, Peak/Avg={peak/avg:.2f}, Util={util:.3f}")
    print(f"  Per-link peak: {per_link}")
    print(f"  Verdict: {'OK (no overload, no drop)' if ok else 'PROBLEM'}\n")

    summary.append((mode, len(distinct_periods), len(pkts), len(infeas), peak, util, ok))

print("=== Summary (edf_min_load) ===")
print(f"{'mode':<9}{'#periods':>9}{'#pkts':>7}{'dropped':>9}{'peak':>8}{'util':>8}  verdict")
for mode, np_, nk, dr, pk, ut, ok in summary:
    print(f"{mode:<9}{np_:>9}{nk:>7}{dr:>9}{pk:>8.3f}{ut:>8.3f}  {'OK' if ok else 'PROBLEM'}")
