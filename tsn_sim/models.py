"""Core data models for the 5G-TSN scheduling simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


Link = int
Slot = int
PacketKey = Tuple[int, int]
CellKey = Tuple[Link, Slot]


@dataclass(frozen=True)
class Flow:
    flow_id: int
    period_ms: int
    packet_size_bits: int
    deadline_ms: float
    available_links: Tuple[Link, ...]
    offset_ms: float
    user_class: str


@dataclass(frozen=True)
class Packet:
    flow_id: int
    packet_index: int
    arrival_ms: float
    earliest_slot: int
    latest_slot: int
    deadline_ms: float
    period_ms: int


@dataclass(frozen=True)
class Candidate:
    packet: PacketKey
    link: Link
    slot: Slot
    required_rb: int
    bits_per_rb: int
    delay_ms: float


@dataclass(frozen=True)
class ScheduleEntry:
    flow_id: int
    packet_index: int
    arrival_ms: float
    deadline_ms: float
    link: Link
    slot: Slot
    required_rb: int
    bits_per_rb: int
    delay_ms: float


@dataclass(frozen=True)
class Scenario:
    flows: Tuple[Flow, ...]
    links: Tuple[Link, ...]
    rb_capacity: Mapping[CellKey, int]
    rb_bits: Mapping[Tuple[int, Link, Slot], int]
    hyperperiod_ms: int
    slot_ms: float


@dataclass(frozen=True)
class SimulationResult:
    status: str
    objective: float
    schedule: Tuple[ScheduleEntry, ...]
    metrics: Mapping[str, float]
    infeasible_packets: Tuple[PacketKey, ...]
    message: str
