"""Shared scheduling framework for heuristic algorithms.

Every heuristic in this package produces a :class:`SimulationResult` by
placing each packet onto a feasible (link, slot) cell while respecting the
per-cell RB capacity.  The common machinery — packet instantiation, candidate
generation, capacity bookkeeping, metric computation and result assembly —
lives here so each algorithm only has to specify *how* it orders packets and
*picks* a cell.

Two building styles are supported:

* :class:`SequentialScheduler` — greedy, packet-by-packet placement.  Used by
  ``random_feasible``, ``greedy`` and ``proportional_fair``.  Subclasses only
  override ``order_packets`` and ``select_candidate``.
* Standalone solvers — algorithms that search over whole schedules (e.g.
  ``genetic_algorithm``) import the helpers below and build their result via
  :func:`build_result`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from ..candidate import build_candidates, instantiate_packets
from ..metrics import compute_metrics
from ..models import (
    Candidate,
    CellKey,
    Packet,
    PacketKey,
    Scenario,
    ScheduleEntry,
    SimulationResult,
)

#: Running capacity bookkeeping.  ``remaining`` counts free RBs per cell;
#: ``used`` counts RBs already committed per cell.
RemainingRb = Dict[CellKey, int]
UsedRb = Dict[CellKey, int]


@dataclass
class State:
    """Mutable placement state passed between scheduling steps."""

    remaining: RemainingRb
    used: UsedRb


def init_state(scenario: Scenario) -> State:
    """Create a fresh bookkeeping state with all capacity free."""
    return State(
        remaining=dict(scenario.rb_capacity),
        used={cell: 0 for cell in scenario.rb_capacity},
    )


def place(state: State, candidate: Candidate) -> None:
    """Commit ``candidate`` to the schedule, updating capacity bookkeeping."""
    cell = (candidate.link, candidate.slot)
    state.remaining[cell] -= candidate.required_rb
    state.used[cell] += candidate.required_rb


def feasible_candidates(
    candidates: Sequence[Candidate], state: State
) -> List[Candidate]:
    """Filter ``candidates`` down to those still fitting in remaining capacity."""
    return [
        candidate
        for candidate in candidates
        if state.remaining[(candidate.link, candidate.slot)] >= candidate.required_rb
    ]


def slot_load_after(
    candidate: Candidate, scenario: Scenario, state: State
) -> float:
    """Normalized load of the candidate's cell *after* placement."""
    cell = (candidate.link, candidate.slot)
    return (state.used[cell] + candidate.required_rb) / scenario.rb_capacity[cell]


def current_peak(scenario: Scenario, state: State) -> float:
    """Highest normalized cell load across the whole schedule so far."""
    return max(
        state.used[cell] / scenario.rb_capacity[cell]
        for cell in scenario.rb_capacity
    )


def build_result(
    scenario: Scenario,
    packets: Mapping[PacketKey, Packet],
    schedule: Sequence[ScheduleEntry],
    infeasible: Sequence[PacketKey],
) -> SimulationResult:
    """Assemble the canonical :class:`SimulationResult` from raw schedule rows."""
    schedule_tuple = tuple(
        sorted(schedule, key=lambda item: (item.slot, item.link, item.flow_id))
    )
    infeasible_tuple = tuple(infeasible)
    metrics = compute_metrics(scenario, schedule_tuple, packets, infeasible_tuple)
    status = "feasible" if not infeasible_tuple else "partial"
    return SimulationResult(
        status=status,
        objective=metrics["effective_peak_load"],
        schedule=schedule_tuple,
        metrics=metrics,
        infeasible_packets=infeasible_tuple,
        message=(
            "All packets scheduled."
            if status == "feasible"
            else "Some packets were left unscheduled."
        ),
    )


class SequentialScheduler:
    """Base class for greedy packet-by-packet heuristics.

    Subclasses implement:

    * :meth:`order_packets` — the order in which packets are considered.
    * :meth:`select_candidate` — which feasible cell a packet is placed on
      (return ``None`` to drop the packet).
    """

    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        self.scenario = scenario
        self.rng = random.Random(seed)
        self.packets = instantiate_packets(scenario)
        self.candidates_by_packet, self.initially_infeasible = build_candidates(
            scenario, self.packets
        )

    # -- overridable hooks -------------------------------------------------

    def order_packets(self) -> List[PacketKey]:
        raise NotImplementedError

    def select_candidate(
        self, packet_key: PacketKey, candidates: List[Candidate], state: State
    ) -> Candidate | None:
        raise NotImplementedError

    # -- shared driver -----------------------------------------------------

    def run(self) -> SimulationResult:
        state = init_state(self.scenario)
        schedule: List[ScheduleEntry] = []
        infeasible: List[PacketKey] = list(self.initially_infeasible)

        for packet_key in self.order_packets():
            if packet_key in self.initially_infeasible:
                continue
            candidates = feasible_candidates(
                self.candidates_by_packet[packet_key], state
            )
            selected = self.select_candidate(packet_key, candidates, state)
            if selected is None:
                infeasible.append(packet_key)
                continue

            place(state, selected)
            packet = self.packets[packet_key]
            schedule.append(
                ScheduleEntry(
                    flow_id=packet.flow_id,
                    packet_index=packet.packet_index,
                    arrival_ms=packet.arrival_ms,
                    deadline_ms=packet.deadline_ms,
                    link=selected.link,
                    slot=selected.slot,
                    required_rb=selected.required_rb,
                    bits_per_rb=selected.bits_per_rb,
                    delay_ms=selected.delay_ms,
                )
            )

        return build_result(
            self.scenario, self.packets, schedule, infeasible
        )
