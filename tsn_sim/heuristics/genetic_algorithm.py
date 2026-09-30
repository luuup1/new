"""Genetic-algorithm (GA) heuristic.

A population-based metaheuristic that searches over *whole schedules* rather
than placing packets one-by-one.

Encoding
--------
Each chromosome is a list of integers, one gene per packet.  Gene ``i`` stores
the index of the candidate (link, slot) chosen for packet ``i``; ``-1`` marks a
packet as dropped.  A decoder walks the packet order and places each packet
onto its chosen candidate only if enough RBs remain, otherwise the packet is
dropped.  This keeps every decoded chromosome capacity-feasible by
construction and lets the objective (``effective_peak_load``) fold both peak
load and drops into a single fitness.

Operators
---------
* Initialisation — a fraction of the population is seeded with the greedy
  solution (warm start); the rest is random.
* Selection — tournament.
* Crossover — uniform crossover over genes.
* Mutation — with probability ``mutation_prob`` each gene is re-picked at
  random; with probability ``drop_prob`` it is set to ``-1`` (drop).
* Elitism — the best ``elite_count`` chromosomes survive unchanged.

Parameters are tunable via ``schedule(..., population_size=..., ...)`` and
default to modest values that converge quickly on small/medium instances.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..candidate import build_candidates, instantiate_packets
from ..metrics import compute_metrics
from ..models import Candidate, PacketKey, Scenario, ScheduleEntry, SimulationResult
from .base import init_state, place, build_result
from .greedy import GreedyScheduler


@dataclass
class GaConfig:
    population_size: int = 40
    generations: int = 60
    crossover_prob: float = 0.8
    mutation_prob: float = 0.1
    drop_prob: float = 0.02
    elite_count: int = 4
    tournament_size: int = 3


class _Chromosome:
    """One candidate solution: a gene per packet selecting a candidate index."""

    __slots__ = ("genes", "fitness")

    def __init__(self, genes: List[int], fitness: float = float("inf")) -> None:
        self.genes = genes
        self.fitness = fitness


def _ordered_packets(packets) -> List[PacketKey]:
    return sorted(packets)


def _random_genes(
    packet_keys: Sequence[PacketKey],
    candidates_by_packet,
    rng: random.Random,
) -> List[int]:
    genes: List[int] = []
    for key in packet_keys:
        choices = candidates_by_packet.get(key, [])
        if not choices:
            genes.append(-1)
            continue
        # Occasionally drop a schedulable packet to allow load shedding.
        if rng.random() < 0.01:
            genes.append(-1)
        else:
            genes.append(rng.randrange(len(choices)))
    return genes


def _decode(
    genes: Sequence[int],
    packet_keys: Sequence[PacketKey],
    scenario: Scenario,
    packets,
    candidates_by_packet,
) -> Tuple[List[ScheduleEntry], List[PacketKey]]:
    """Turn a chromosome into a capacity-feasible schedule (drop on conflict)."""
    state = init_state(scenario)
    schedule: List[ScheduleEntry] = []
    infeasible: List[PacketKey] = []

    for key, gene in zip(packet_keys, genes):
        choices = candidates_by_packet.get(key, [])
        if gene < 0 or gene >= len(choices):
            infeasible.append(key)
            continue
        candidate = choices[gene]
        cell = (candidate.link, candidate.slot)
        if state.remaining[cell] < candidate.required_rb:
            infeasible.append(key)
            continue
        place(state, candidate)
        packet = packets[key]
        schedule.append(
            ScheduleEntry(
                flow_id=packet.flow_id,
                packet_index=packet.packet_index,
                arrival_ms=packet.arrival_ms,
                deadline_ms=packet.deadline_ms,
                link=candidate.link,
                slot=candidate.slot,
                required_rb=candidate.required_rb,
                bits_per_rb=candidate.bits_per_rb,
                delay_ms=candidate.delay_ms,
            )
        )

    return schedule, infeasible


def _evaluate(
    genes: Sequence[int],
    packet_keys: Sequence[PacketKey],
    scenario: Scenario,
    packets,
    candidates_by_packet,
) -> float:
    schedule, infeasible = _decode(genes, packet_keys, scenario, packets, candidates_by_packet)
    metrics = compute_metrics(scenario, tuple(schedule), packets, tuple(infeasible))
    return float(metrics["effective_peak_load"])


def _tournament(population: List[_Chromosome], size: int, rng: random.Random) -> _Chromosome:
    contestants = rng.sample(population, min(size, len(population)))
    return min(contestants, key=lambda c: c.fitness)


def _crossover(
    parent_a: _Chromosome, parent_b: _Chromosome, rng: random.Random
) -> List[int]:
    return [
        a if rng.random() < 0.5 else b
        for a, b in zip(parent_a.genes, parent_b.genes)
    ]


def _mutate(
    genes: List[int],
    packet_keys: Sequence[PacketKey],
    candidates_by_packet,
    rng: random.Random,
    config: GaConfig,
) -> List[int]:
    mutated = list(genes)
    for index, key in enumerate(packet_keys):
        choices = candidates_by_packet.get(key, [])
        if rng.random() >= config.mutation_prob:
            continue
        if not choices or rng.random() < config.drop_prob:
            mutated[index] = -1
        else:
            mutated[index] = rng.randrange(len(choices))
    return mutated


def _greedy_genes(
    packet_keys: Sequence[PacketKey],
    scenario: Scenario,
    packets,
    candidates_by_packet,
) -> List[int]:
    """Encode the greedy solution as a chromosome (warm start)."""
    greedy = GreedyScheduler(scenario).run()
    placed = {(e.flow_id, e.packet_index): (e.link, e.slot) for e in greedy.schedule}
    genes: List[int] = []
    for key in packet_keys:
        cell = placed.get(key)
        if cell is None:
            genes.append(-1)
            continue
        choices = candidates_by_packet.get(key, [])
        for index, candidate in enumerate(choices):
            if (candidate.link, candidate.slot) == cell:
                genes.append(index)
                break
        else:
            genes.append(-1)
    return genes


def schedule(
    scenario: Scenario,
    seed: int = 0,
    *,
    population_size: int | None = None,
    generations: int | None = None,
    crossover_prob: float | None = None,
    mutation_prob: float | None = None,
    drop_prob: float | None = None,
    elite_count: int | None = None,
    tournament_size: int | None = None,
) -> SimulationResult:
    """Run the genetic algorithm on ``scenario``."""
    config = GaConfig()
    if population_size is not None:
        config.population_size = population_size
    if generations is not None:
        config.generations = generations
    if crossover_prob is not None:
        config.crossover_prob = crossover_prob
    if mutation_prob is not None:
        config.mutation_prob = mutation_prob
    if drop_prob is not None:
        config.drop_prob = drop_prob
    if elite_count is not None:
        config.elite_count = elite_count
    if tournament_size is not None:
        config.tournament_size = tournament_size

    rng = random.Random(seed)
    packets = instantiate_packets(scenario)
    candidates_by_packet, initially_infeasible = build_candidates(scenario, packets)
    packet_keys = _ordered_packets(packets)

    def evaluate(genes: Sequence[int]) -> float:
        return _evaluate(genes, packet_keys, scenario, packets, candidates_by_packet)

    # -- initial population (greedy warm start + random) --------------------
    population: List[_Chromosome] = []
    population.append(_Chromosome(_greedy_genes(packet_keys, scenario, packets, candidates_by_packet)))
    for _ in range(config.population_size - 1):
        population.append(_Chromosome(_random_genes(packet_keys, candidates_by_packet, rng)))

    for chromosome in population:
        chromosome.fitness = evaluate(chromosome.genes)

    # -- evolution ---------------------------------------------------------
    for _ in range(config.generations):
        population.sort(key=lambda c: c.fitness)
        next_population = population[: config.elite_count]

        while len(next_population) < config.population_size:
            parent_a = _tournament(population, config.tournament_size, rng)
            parent_b = _tournament(population, config.tournament_size, rng)
            if rng.random() < config.crossover_prob:
                child_genes = _crossover(parent_a, parent_b, rng)
            else:
                child_genes = list(parent_a.genes)
            child_genes = _mutate(child_genes, packet_keys, candidates_by_packet, rng, config)
            next_population.append(_Chromosome(child_genes))

        population = next_population
        for chromosome in population:
            chromosome.fitness = evaluate(chromosome.genes)

    best = min(population, key=lambda c: c.fitness)
    schedule_rows, infeasible = _decode(
        best.genes, packet_keys, scenario, packets, candidates_by_packet
    )
    # Mark initially-infeasible packets explicitly (they can never be placed).
    infeasible = list(dict.fromkeys(list(infeasible) + list(initially_infeasible)))
    return build_result(scenario, packets, schedule_rows, infeasible)
