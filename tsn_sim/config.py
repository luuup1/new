"""Configuration objects and defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class HeuristicConfig:
    strategy: str = "urgency_lexicographic"
    peak_weight: float = 10.0
    slot_load_weight: float = 2.0
    link_avg_weight: float = 0.5
    delay_weight: float = 0.1
    efficiency_weight: float = 0.2


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 7
    slot_ms: float = 1.0
    link_count: int = 3
    flow_count: int = 50
    periods_ms: Tuple[int, ...] = (2, 4, 8, 16, 32)
    period_mode: str = "cyclic"
    order_mode: str = "edf"
    simple_periods: Tuple[int, ...] = (4, 8)
    packet_size_bits: Tuple[int, ...] = (800, 1200, 1800, 2400)
    load_scale: float = 1.0
    deadline_ratio_min: float = 0.5
    deadline_ratio_max: float = 1.0
    rb_base_by_link: Tuple[int, ...] = (40, 60, 80)
    rb_variation: int = 4
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)


def default_config(**overrides: Any) -> SimulationConfig:
    values = asdict(SimulationConfig())
    heuristic_values = values.pop("heuristic")
    if "heuristic" in overrides:
        heuristic_override = overrides.pop("heuristic")
        if isinstance(heuristic_override, HeuristicConfig):
            heuristic_values = asdict(heuristic_override)
        else:
            heuristic_values.update(heuristic_override)
    values.update(overrides)
    return SimulationConfig(heuristic=HeuristicConfig(**heuristic_values), **values)


def load_config(path: str | Path) -> SimulationConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _config_from_mapping(data)


def save_config(config: SimulationConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def _config_from_mapping(data: Mapping[str, Any]) -> SimulationConfig:
    values = dict(data)
    heuristic_data = values.pop("heuristic", {})
    tuple_fields = ("periods_ms", "packet_size_bits", "rb_base_by_link")
    for field_name in tuple_fields:
        if field_name in values:
            values[field_name] = tuple(values[field_name])
    return SimulationConfig(
        heuristic=HeuristicConfig(**heuristic_data),
        **values,
    )
