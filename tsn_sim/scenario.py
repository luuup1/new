"""Scenario generation for configurable 5G-TSN experiments."""

from __future__ import annotations

import math
import random
from functools import reduce
from math import gcd
from typing import Dict, Iterable, List

from .config import SimulationConfig
from .models import CellKey, Flow, Link, Scenario


def lcm(values: Iterable[int]) -> int:
    return reduce(lambda a, b: a * b // gcd(a, b), values, 1)


def build_scenario(config: SimulationConfig) -> Scenario:
    rng = random.Random(config.seed)
    links = tuple(range(1, config.link_count + 1))
    hyperperiod_ms = lcm(config.periods_ms)

    flows = _build_flows(config, links, rng)
    rb_capacity = _build_rb_capacity(config, links, hyperperiod_ms)
    rb_bits = _build_rb_bits(config, flows, links, hyperperiod_ms, rng)

    return Scenario(
        flows=tuple(flows),
        links=links,
        rb_capacity=rb_capacity,
        rb_bits=rb_bits,
        hyperperiod_ms=hyperperiod_ms,
        slot_ms=config.slot_ms,
    )


def _build_flows(
    config: SimulationConfig, links: tuple[Link, ...], rng: random.Random
) -> List[Flow]:
    flows: List[Flow] = []
    user_classes = ("edge", "middle", "close")

    for flow_id in range(config.flow_count):
        period = config.periods_ms[flow_id % len(config.periods_ms)]
        packet_size = int(
            config.packet_size_bits[flow_id % len(config.packet_size_bits)]
            * config.load_scale
        )
        deadline_ratio = rng.uniform(config.deadline_ratio_min, config.deadline_ratio_max)
        deadline = max(config.slot_ms, period * deadline_ratio)
        offset = float(rng.randrange(period))
        user_class = user_classes[flow_id % len(user_classes)]
        available_links = _available_links_for_user(user_class, links)

        flows.append(
            Flow(
                flow_id=flow_id,
                period_ms=period,
                packet_size_bits=packet_size,
                deadline_ms=deadline,
                available_links=available_links,
                offset_ms=offset,
                user_class=user_class,
            )
        )

    return flows


def _available_links_for_user(user_class: str, links: tuple[Link, ...]) -> tuple[Link, ...]:
    if user_class == "edge":
        return links[:1]
    if user_class == "middle":
        return links[: min(2, len(links))]
    return links


def _build_rb_capacity(
    config: SimulationConfig, links: tuple[Link, ...], hyperperiod_ms: int
) -> Dict[CellKey, int]:
    rb_capacity: Dict[CellKey, int] = {}
    for link in links:
        base = config.rb_base_by_link[min(link - 1, len(config.rb_base_by_link) - 1)]
        for slot in range(hyperperiod_ms):
            wave = int(round(config.rb_variation * math.sin((slot + link) / 4.0)))
            rb_capacity[(link, slot)] = max(1, base + wave)
    return rb_capacity


def _build_rb_bits(
    config: SimulationConfig,
    flows: List[Flow],
    links: tuple[Link, ...],
    hyperperiod_ms: int,
    rng: random.Random,
) -> Dict[tuple[int, Link, int], int]:
    rb_bits: Dict[tuple[int, Link, int], int] = {}
    class_factor = {"edge": 0.78, "middle": 0.95, "close": 1.15}
    link_base = {1: 360, 2: 560, 3: 820}
    link_fading = {1: 25, 2: 55, 3: 110}

    for flow in flows:
        for link in links:
            base = link_base.get(link, 360 + 180 * (link - 1))
            fading_amp = link_fading.get(link, 60 + 25 * link)
            for slot in range(hyperperiod_ms):
                slow_fading = fading_amp * math.sin((slot + 1) * (flow.flow_id + link + 1) / 6.0)
                random_jitter = rng.randrange(-18, 19)
                value = (base + slow_fading + random_jitter) * class_factor[flow.user_class]
                rb_bits[(flow.flow_id, link, slot)] = max(80, int(value))

    return rb_bits
