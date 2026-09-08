# 5G-TSN Multi-Link Heuristic Simulation

This project implements the first heuristic version of the 5G-TSN multi-link
scheduling model. Each packet chooses one available link and one 1 ms slot. The
scheduler minimizes peak normalized RB load (with drop-penalty) greedily while
respecting arrival time, deadline, link availability, and RB capacity constraints.

## Run

```powershell
python tsn_scheduler.py
```

Useful variants:

```powershell
python tsn_scheduler.py --flows 30 --load-scale 1.2
python tsn_scheduler.py --compare-links
python tsn_scheduler.py --compare-heuristics
python tsn_scheduler.py --flows 5 --compare-milp --milp-time-limit 10
python tsn_scheduler.py --sweep-load 0.8 1.0 1.2 1.5
python tsn_scheduler.py --schedule-csv schedule.csv --metrics-json metrics.json
```

Run tests:

```powershell
python -m unittest -v
```

## Structure

- `tsn_sim/config.py`: simulation and heuristic parameters.
- `tsn_sim/models.py`: flow, packet, candidate, scenario, and result models.
- `tsn_sim/scenario.py`: default scenario generation.
- `tsn_sim/candidate.py`: packet instantiation and feasible candidate filtering.
- `tsn_sim/heuristics.py`: heuristic schedulers (random, EDF, urgency ordering with lexicographic scoring).
- `tsn_sim/milp.py`: small-scale MILP optimum solver for heuristic gap checks.
- `tsn_sim/metrics.py`: load, delay, drop, and export helpers.
- `tsn_sim/experiment.py`: link comparison and load-scale sweeps.
- `tsn_scheduler.py`: command-line entry point.

## Heuristic And Optimum Checks

Available heuristic strategies:

- `random_feasible`: random feasible reference (random lower bound).
- `edf_min_load`: earliest-deadline-first order, choose the lowest loaded slot.
- `urgency_lexicographic`: narrowest feasible window first, lexicographic scoring (peak → slot-load → tiebreakers).
- `edf_min_peak`: earliest-deadline-first order, same lexicographic scoring.

Compare them:

```powershell
python tsn_scheduler.py --flows 20 --compare-heuristics
```

For small scenarios, compare heuristics with the MILP optimum:

```powershell
python tsn_scheduler.py --flows 5 --compare-milp --milp-time-limit 10
```

Use small flow counts for MILP. The exact solver is for quality validation, not
large-scale experiments.

## Constraint Handling

The C4 deadline constraint is handled during candidate generation:

```text
finish_time = (slot + 1) * slot_ms
delay = finish_time - arrival_time
candidate is feasible only if delay <= D_i
```

The earliest transmission constraint is also handled there:

```text
slot >= ceil(arrival_time / slot_ms)
```

This keeps the heuristic scheduler focused on choosing among already-feasible
candidates.
