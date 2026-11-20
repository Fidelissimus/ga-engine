"""Emigrant selection and immigrant integration for the island model."""

from __future__ import annotations

from typing import Callable, List, Sequence

from .models import Individual
from .multiobjective import assign_rank_and_crowding, nsga2_order

ScoreFunction = Callable[[Individual], float]


def select_emigrants(
    population: Sequence[Individual],
    count: int,
    *,
    multi_objective: bool,
    score: ScoreFunction,
) -> List[Individual]:
    """Choose the individuals that leave a population.

    Single-objective: the ``count`` best by ``score``. Multi-objective: members
    of the first front with the largest crowding distance, so migrants are
    both good and diverse.

    Returns:
        Clones, so the source population is not aliased.
    """
    if not population or count <= 0:
        return []
    if multi_objective:
        first_front = assign_rank_and_crowding(population)[0]
        ordered = sorted(first_front, key=lambda i: -population[i].crowding_distance)
        return [population[i].clone() for i in ordered[:count]]
    return [ind.clone() for ind in sorted(population, key=score)[:count]]


def merge_immigrants(
    population: Sequence[Individual],
    immigrants: Sequence[Individual],
    size: int,
    *,
    multi_objective: bool,
    score: ScoreFunction,
) -> List[Individual]:
    """Merge immigrants into a population and keep the best ``size`` individuals.

    Ages are left unchanged (migration is not a generation).

    Returns:
        A new list of cloned individuals.
    """
    combined = list(population) + [m.clone() for m in immigrants]
    if multi_objective:
        order = nsga2_order(combined)
    else:
        order = sorted(range(len(combined)), key=lambda i: score(combined[i]))
    return [combined[i].clone() for i in order[:size]]
