"""Configuration objects, defaults, and the single source of truth for all
environment settings.

All environment configuration lives here. Prefer :class:`SimulationConfig` /
:func:`default_config` over ad-hoc literals scattered across scripts.

Fields of :class:`SimulationConfig` (``name : type = default``):

    seed                : int   = 7
    slot_ms             : float = 1.0
    link_count          : int   = 3
    flow_count          : int   = 50
    periods_ms          : tuple = (2, 4, 8, 16, 32)
    period_mode         : str   = "cyclic"   # cyclic | simple | random
    order_mode          : str   = "edf"      # edf | random
    simple_periods      : tuple = (4, 8)     # used when period_mode == "simple"
    packet_size_bits    : tuple = (800, 1200, 1800, 2400)
    load_scale          : float = 1.0
    deadline_ratio_min  : float = 0.5
    deadline_ratio_max  : float = 1.0
    rb_base_by_link     : tuple = (40, 60, 80)
    rb_variation        : int   = 4
    heuristic.strategy  : str   = "random_feasible"

JSON round-tripping is supported via :func:`load_config` / :func:`save_config`.
Use :func:`write_example_config` to (re)generate an example JSON file; there is
no hand-maintained ``config.example.json`` anymore, so the example can never
drift out of sync with the defaults.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class HeuristicConfig:
    strategy: str = "random_feasible"


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


def write_example_config(path: str | Path = "config.example.json") -> SimulationConfig:
    """Write an example JSON config reflecting the current defaults.

    This replaces the old hand-maintained ``config.example.json``: the example
    is generated straight from :class:`SimulationConfig`, so it can never drift
    out of sync with the defaults. Returns the config that was written.
    """
    config = default_config()
    save_config(config, path)
    return config


def _config_from_mapping(data: Mapping[str, Any]) -> SimulationConfig:
    values = dict(data)
    heuristic_data = values.pop("heuristic", {})
    # Ignore legacy weight fields (removed) so stale config files still load.
    heuristic_data = {
        key: value
        for key, value in heuristic_data.items()
        if key in HeuristicConfig.__dataclass_fields__
    }
    # Normalise removed strategy names to the default (they no longer exist).
    _LEGACY_STRATEGIES = {"edf_min_load", "edf_min_peak", "urgency_lexicographic"}
    if heuristic_data.get("strategy") in _LEGACY_STRATEGIES:
        heuristic_data["strategy"] = "random_feasible"
    tuple_fields = ("periods_ms", "simple_periods", "packet_size_bits", "rb_base_by_link")
    for field_name in tuple_fields:
        if field_name in values:
            values[field_name] = tuple(values[field_name])
    return SimulationConfig(
        heuristic=HeuristicConfig(**heuristic_data),
        **values,
    )
