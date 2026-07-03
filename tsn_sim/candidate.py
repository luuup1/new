"""Packet instantiation and feasible candidate generation."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Tuple

from .models import Candidate, Packet, PacketKey, Scenario


def ceil_slot(value: float, eps: float = 1e-9) -> int:
    if abs(value - round(value)) <= eps:
        return int(round(value))
    return math.ceil(value)


def floor_slot(value: float, eps: float = 1e-9) -> int:
    if abs(value - round(value)) <= eps:
        return int(round(value))
    return math.floor(value)


def instantiate_packets(scenario: Scenario) -> Dict[PacketKey, Packet]:
    packets: Dict[PacketKey, Packet] = {}
    for flow in scenario.flows:
        count = scenario.hyperperiod_ms // flow.period_ms
        for packet_index in range(count):
            arrival = flow.offset_ms + packet_index * flow.period_ms
            earliest = ceil_slot(arrival / scenario.slot_ms)
            latest = floor_slot((arrival + flow.deadline_ms) / scenario.slot_ms - 1.0)
            packets[(flow.flow_id, packet_index)] = Packet(
                flow_id=flow.flow_id,
                packet_index=packet_index,
                arrival_ms=arrival,
                earliest_slot=earliest,
                latest_slot=latest,
                deadline_ms=flow.deadline_ms,
                period_ms=flow.period_ms,
            )
    return packets


def build_candidates(
    scenario: Scenario, packets: Mapping[PacketKey, Packet]
) -> Tuple[Dict[PacketKey, List[Candidate]], Tuple[PacketKey, ...]]:
    flows = {flow.flow_id: flow for flow in scenario.flows}
    candidates: Dict[PacketKey, List[Candidate]] = {key: [] for key in packets}

    for packet_key, packet in packets.items():
        flow = flows[packet.flow_id]
        for link in flow.available_links:
            for slot in range(scenario.hyperperiod_ms):
                if slot < packet.earliest_slot or slot > packet.latest_slot:
                    continue
                if (link, slot) not in scenario.rb_capacity:
                    continue

                finish_time = (slot + 1) * scenario.slot_ms
                delay = finish_time - packet.arrival_ms
                if delay > flow.deadline_ms:
                    continue

                bits_per_rb = scenario.rb_bits[(flow.flow_id, link, slot)]
                required_rb = math.ceil(flow.packet_size_bits / bits_per_rb)
                if required_rb <= scenario.rb_capacity[(link, slot)]:
                    candidates[packet_key].append(
                        Candidate(
                            packet=packet_key,
                            link=link,
                            slot=slot,
                            required_rb=required_rb,
                            bits_per_rb=bits_per_rb,
                            delay_ms=delay,
                        )
                    )

    infeasible = tuple(key for key, values in candidates.items() if not values)
    return candidates, infeasible
