"""Per-generation statistics collection."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

import numpy as np

from .models import Individual
from .representations.base import Representation

#: Keys of the statistics dictionary; every list has one entry per recorded generation.
STAT_KEYS = (
    "generation",
    "best_fitness",
    "worst_fitness",
    "mean_fitness",
    "median_fitness",
    "std_fitness",
    "pareto_front_size",
    "feasible_count",
    "mean_constraint_violation",
    "diversity",
    "mutation_prob",
    "gene_mutation_prob",
    "mutation_success_ratio",
    "evaluations",
    "evaluation_failures",
)

_FITNESS_KEYS = (
    "best_fitness",
    "worst_fitness",
    "mean_fitness",
    "median_fitness",
    "std_fitness",
)


class GenerationStatistics:
    """Accumulates one row of population statistics per generation.

    Fitness statistics describe the *current population* using raw fitness
    (without constraint penalties): scalars for single-objective runs and
    per-objective tuples for multi-objective runs. The best-so-far trajectory
    is tracked separately by the engine as ``best_fitness_history``.
    """

    def __init__(self) -> None:
        self.data: Dict[str, List[Any]] = {key: [] for key in STAT_KEYS}

    def reset(self) -> None:
        """Discard all recorded rows."""
        self.data = {key: [] for key in STAT_KEYS}

    def copy(self) -> Dict[str, List[Any]]:
        """Return a deep copy of the recorded statistics."""
        return copy.deepcopy(self.data)

    def load(self, recorded: Mapping[str, Sequence[Any]]) -> None:
        """Replace the contents with previously recorded rows.

        Keys missing from ``recorded`` (e.g. in checkpoints written by an older
        version) are padded with ``None`` so all lists stay the same length.
        Lists produced by JSON round-trips are converted back to tuples.
        """
        rows = len(recorded.get("generation", []))
        loaded: Dict[str, List[Any]] = {}
        for key in STAT_KEYS:
            values = list(recorded.get(key, [None] * rows))
            if key in _FITNESS_KEYS:
                values = [tuple(v) if isinstance(v, list) else v for v in values]
            loaded[key] = values
        self.data = loaded

    def record(
        self,
        *,
        generation: int,
        population: Sequence[Individual],
        representation: Representation,
        multi_objective: bool,
        pareto_front_size: int,
        mutation_prob: float,
        gene_mutation_prob: float,
        success_ratio: Optional[float],
        evaluations: int,
        evaluation_failures: int,
    ) -> None:
        """Append one row describing the current population."""
        if not population:
            return
        # Failed evaluations carry ``inf`` which legitimately yields NaN spreads.
        summary: Dict[str, Any]
        with np.errstate(invalid="ignore"):
            if multi_objective:
                values = np.asarray([ind.fitness for ind in population], dtype=np.float64)
                summary = {
                    "best_fitness": tuple(float(x) for x in np.min(values, axis=0)),
                    "worst_fitness": tuple(float(x) for x in np.max(values, axis=0)),
                    "mean_fitness": tuple(float(x) for x in np.mean(values, axis=0)),
                    "median_fitness": tuple(float(x) for x in np.median(values, axis=0)),
                    "std_fitness": tuple(float(x) for x in np.std(values, axis=0)),
                }
            else:
                raw = np.asarray(
                    [cast(float, ind.fitness) for ind in population], dtype=np.float64
                )
                summary = {
                    "best_fitness": float(np.min(raw)),
                    "worst_fitness": float(np.max(raw)),
                    "mean_fitness": float(np.mean(raw)),
                    "median_fitness": float(np.median(raw)),
                    "std_fitness": float(np.std(raw)),
                }
            mean_violation = float(np.mean([ind.constraint_violation for ind in population]))

        genes = np.asarray([ind.genes for ind in population])
        row: Dict[str, Any] = {
            "generation": generation,
            **summary,
            "pareto_front_size": pareto_front_size,
            "feasible_count": int(sum(ind.is_feasible for ind in population)),
            "mean_constraint_violation": mean_violation,
            "diversity": representation.diversity(genes),
            "mutation_prob": float(mutation_prob),
            "gene_mutation_prob": float(gene_mutation_prob),
            "mutation_success_ratio": success_ratio,
            "evaluations": evaluations,
            "evaluation_failures": evaluation_failures,
        }
        for key in STAT_KEYS:
            self.data[key].append(row[key])
