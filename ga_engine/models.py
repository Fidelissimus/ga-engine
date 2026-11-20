"""Plain data containers shared across the package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .aliases import Fitness


@dataclass(eq=False)
class Individual:
    """A candidate solution together with its evaluated metadata.

    The class is deliberately mutable: ``age``, ``rank`` and
    ``crowding_distance`` are evolutionary bookkeeping rather than value-object
    fields. Equality is identity-based (``eq=False``) because comparing NumPy
    arrays with ``==`` is ambiguous.

    Attributes:
        genes: One-dimensional genome array.
        fitness: Scalar, or tuple of objective values (all minimised).
        age: Generations the individual has survived (0 for a newborn).
        crowding_distance: NSGA-II crowding distance (multi-objective only).
        constraint_violation: Aggregate violation; ``0.0`` means feasible.
        rank: NSGA-II non-domination rank, 0 being the best front.
    """

    genes: np.ndarray
    fitness: Fitness
    age: int = 0
    crowding_distance: float = 0.0
    constraint_violation: float = 0.0
    rank: int = 0

    @property
    def is_feasible(self) -> bool:
        """Whether the individual satisfies every constraint."""
        return self.constraint_violation <= 0.0

    def clone(self) -> "Individual":
        """Return an independent copy (genes are copied, fitness is immutable)."""
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            age=int(self.age),
            crowding_distance=float(self.crowding_distance),
            constraint_violation=float(self.constraint_violation),
            rank=int(self.rank),
        )


@dataclass
class RunResult:
    """Structured result returned by ``run()``.

    Attributes:
        best_individual: Best solution found (a representative of the Pareto
            front in multi-objective mode).
        best_fitness: Fitness of ``best_individual``.
        generation: Generation counter when the run stopped.
        elapsed_seconds: Wall-clock duration of the ``run()`` call.
        converged: ``True`` if a target or stagnation criterion stopped the run.
        stop_reason: ``"max_generations"``, ``"target_fitness"`` or ``"stagnation"``.
        stats: Per-generation statistics (see :mod:`ga_engine.statistics`).
        pareto_front: Non-dominated set (multi-objective only, else ``None``).
        evaluations: Cumulative number of fitness evaluations performed.
        evaluation_failures: How many of those evaluations failed.
        island_stats: Per-island statistics (island model only, else ``None``).
    """

    best_individual: Individual
    best_fitness: Fitness
    generation: int
    elapsed_seconds: float
    converged: bool
    stop_reason: str
    stats: Dict[str, List[Any]]
    pareto_front: Optional[List[Individual]] = None
    evaluations: int = 0
    evaluation_failures: int = 0
    island_stats: Optional[List[Dict[str, List[Any]]]] = None
