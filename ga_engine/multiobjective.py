"""NSGA-II machinery: constrained dominance, fronts and crowding distance.

The dominance relation is computed as one vectorised ``(n, n)`` boolean matrix,
so sorting a population costs ``O(n^2 * m)`` NumPy work instead of ``O(n^2)``
Python-level function calls. Memory is ``O(n^2 * m)`` booleans; for populations
beyond a few thousand individuals consider splitting the problem across
islands.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from .models import Individual


def dominance_matrix(fitness: np.ndarray, violation: np.ndarray) -> np.ndarray:
    """Compute constrained Pareto dominance between every pair of solutions.

    Args:
        fitness: Array of shape ``(n, m)`` with the objective values.
        violation: Array of shape ``(n,)`` with constraint violations.

    Returns:
        Boolean ``(n, n)`` matrix ``D`` where ``D[i, j]`` means "``i`` dominates ``j``":
        a feasible solution dominates any infeasible one; between two infeasible
        solutions the smaller violation dominates; between two feasible ones
        the usual Pareto rule applies.
    """
    feasible = violation <= 0.0
    less_equal = np.all(fitness[:, None, :] <= fitness[None, :, :], axis=2)
    strictly_less = np.any(fitness[:, None, :] < fitness[None, :, :], axis=2)
    both_feasible = feasible[:, None] & feasible[None, :]
    both_infeasible = ~feasible[:, None] & ~feasible[None, :]
    return (
        (both_feasible & less_equal & strictly_less)
        | (feasible[:, None] & ~feasible[None, :])
        | (both_infeasible & (violation[:, None] < violation[None, :]))
    )


def _arrays(population: Sequence[Individual]) -> "tuple[np.ndarray, np.ndarray]":
    """Stack fitness and violation values of a population into arrays."""
    fitness = np.asarray([ind.fitness for ind in population], dtype=float)
    violation = np.asarray([ind.constraint_violation for ind in population], dtype=float)
    return fitness, violation


def non_dominated_fronts(population: Sequence[Individual]) -> List[List[int]]:
    """Partition a population into successive non-dominated fronts.

    Args:
        population: Evaluated individuals with equally sized fitness tuples.

    Returns:
        Fronts as lists of indices into ``population``; front 0 is the Pareto set.

    Raises:
        RuntimeError: If the dominance relation contains a cycle (impossible for
            valid inputs; guards against corrupted data).
    """
    count = len(population)
    if count == 0:
        return []
    fitness, violation = _arrays(population)
    dominates = dominance_matrix(fitness, violation)
    dominated_by = dominates.sum(axis=0)
    remaining = np.ones(count, dtype=bool)
    fronts: List[List[int]] = []
    while remaining.any():
        current = np.flatnonzero(remaining & (dominated_by == 0))
        if current.size == 0:
            raise RuntimeError("Dominance relation contains a cycle")
        fronts.append(current.tolist())
        remaining[current] = False
        dominated_by = dominated_by - dominates[current].sum(axis=0)
    return fronts


def assign_crowding_distance(front: Sequence[Individual]) -> None:
    """Set ``crowding_distance`` on every member of one front (in place).

    Boundary solutions of each objective receive an infinite distance. Fronts of
    one or two solutions are entirely boundary solutions.
    """
    size = len(front)
    if size == 0:
        return
    if size <= 2:
        for individual in front:
            individual.crowding_distance = math.inf
        return
    values = np.asarray([ind.fitness for ind in front], dtype=float)
    distance = np.zeros(size)
    for objective in range(values.shape[1]):
        column = values[:, objective]
        order = np.argsort(column, kind="stable")
        distance[order[0]] = distance[order[-1]] = math.inf
        span = column[order[-1]] - column[order[0]]
        if span == 0.0 or not np.isfinite(span):
            continue
        distance[order[1:-1]] += (column[order[2:]] - column[order[:-2]]) / span
    for individual, value in zip(front, distance):
        individual.crowding_distance = float(value)


def assign_rank_and_crowding(population: Sequence[Individual]) -> List[List[int]]:
    """Compute fronts, then set ``rank`` and ``crowding_distance`` in place.

    Returns:
        The fronts as index lists (see :func:`non_dominated_fronts`).
    """
    fronts = non_dominated_fronts(population)
    for rank, indices in enumerate(fronts):
        members = [population[i] for i in indices]
        assign_crowding_distance(members)
        for individual in members:
            individual.rank = rank
    return fronts


def nsga2_order(population: Sequence[Individual]) -> List[int]:
    """Order a population from best to worst by NSGA-II's crowded comparison.

    Individuals are ranked by front first and, within a front, by descending
    crowding distance. Ranks and crowding distances are updated in place.

    Returns:
        Indices into ``population``, best first.
    """
    order: List[int] = []
    for indices in assign_rank_and_crowding(population):
        order.extend(sorted(indices, key=lambda i: -population[i].crowding_distance))
    return order
