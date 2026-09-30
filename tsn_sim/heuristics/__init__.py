"""Heuristic scheduling algorithms.

Each algorithm is implemented in its own module and exposes a uniform entry
point::

    schedule(scenario, seed=0, **params) -> SimulationResult

The modules are:

=====================  ======================================================
Module                 Strategy
=====================  ======================================================
``random_feasible``    Random order + random feasible placement (baseline).
``greedy``             EDF order + min-load placement.
``proportional_fair``  EDF order + proportional-fair (log-utility) placement.
``genetic_algorithm``  Population-based whole-schedule search (GA).
=====================  ======================================================

Use :func:`schedule_with_heuristic` for a strategy-name driven dispatcher that
keeps the historical calling convention (``schedule_with_heuristic(scenario,
heuristic=..., seed=...)``) working for existing call sites.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..models import Scenario, SimulationResult
from . import genetic_algorithm, greedy, proportional_fair, random_feasible

#: Strategy name -> scheduler function.  All share the ``(scenario, seed,
#: **params)`` signature.
STRATEGIES: Mapping[str, Callable[..., SimulationResult]] = {
    "random_feasible": random_feasible.schedule,
    "greedy": greedy.schedule,
    "proportional_fair": proportional_fair.schedule,
    "genetic_algorithm": genetic_algorithm.schedule,
}

#: Human-readable alias for the proportional-fair algorithm.
PF_ALIASES = ("pf", "proportional_fair")
#: Human-readable alias for the genetic algorithm.
GA_ALIASES = ("ga", "genetic_algorithm")


def resolve_strategy(strategy: str) -> str:
    """Normalise a strategy name to a canonical key in :data:`STRATEGIES`."""
    key = strategy.strip().lower()
    if key in PF_ALIASES:
        return "proportional_fair"
    if key in GA_ALIASES:
        return "genetic_algorithm"
    if key not in STRATEGIES:
        raise ValueError(
            f"Unknown heuristic strategy {strategy!r}; "
            f"expected one of {sorted(STRATEGIES)}"
        )
    return key


def schedule_with_heuristic(
    scenario: Scenario,
    heuristic: Any = None,
    seed: int = 0,
    **params: Any,
) -> SimulationResult:
    """Dispatch to the heuristic named by ``heuristic``.

    ``heuristic`` may be:

    * a :class:`~tsn_sim.config.HeuristicConfig` (its ``strategy`` is used);
    * a string strategy name (``"random_feasible"``, ``"greedy"``,
      ``"proportional_fair"``/``"pf"``, ``"genetic_algorithm"``/``"ga"``);
    * ``None``, in which case ``strategy`` may be given as a keyword, or the
      default ``random_feasible`` is used.

    Extra keyword arguments are forwarded to the scheduler (e.g. GA
    ``population_size``/``generations``).
    """
    if heuristic is None:
        strategy = params.pop("strategy", "random_feasible")
    elif hasattr(heuristic, "strategy"):
        strategy = heuristic.strategy
    else:
        strategy = str(heuristic)

    return STRATEGIES[resolve_strategy(strategy)](scenario, seed=seed, **params)


__all__ = [
    "STRATEGIES",
    "PF_ALIASES",
    "GA_ALIASES",
    "resolve_strategy",
    "schedule_with_heuristic",
    "random_feasible",
    "greedy",
    "proportional_fair",
    "genetic_algorithm",
]
