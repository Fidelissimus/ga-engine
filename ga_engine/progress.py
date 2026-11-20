"""Best-so-far, stagnation and restart bookkeeping."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .aliases import Fitness
from .fitness import FitnessComparator
from .models import Individual


class ProgressTracker:
    """Tracks whether the search is still improving.

    * **Single-objective:** the best-so-far individual (by penalised fitness);
      an improvement resets the stagnation counter if it exceeds
      ``tolerance``.
    * **Multi-objective:** a representative of the current Pareto front
      (lexicographically smallest objective vector); the stagnation counter
      resets whenever the front changes.

    It also counts generations since the last restart, which is separate from
    the stagnation counter so restarts never hide stagnation.

    Args:
        comparator: Defines "better" for single-objective comparison.
        multi_objective: Selects the tracking mode.
        tolerance: Smallest improvement that resets stagnation (single-objective).

    Attributes:
        best_individual: Best (or representative) solution so far.
        stagnation_count: Generations without (significant) improvement.
        generations_since_restart: Generations since the last restart.
        history: Best fitness after every recorded generation.
    """

    def __init__(
        self, comparator: FitnessComparator, multi_objective: bool, tolerance: float = 0.0
    ) -> None:
        self.comparator = comparator
        self.multi_objective = multi_objective
        self.tolerance = tolerance
        self.best_individual: Optional[Individual] = None
        self.stagnation_count = 0
        self.generations_since_restart = 0
        self.history: List[Fitness] = []
        self.pareto_signature: Optional[Tuple[Any, ...]] = None

    def reset(self) -> None:
        """Forget everything (start of a new run)."""
        self.best_individual = None
        self.stagnation_count = 0
        self.generations_since_restart = 0
        self.history = []
        self.pareto_signature = None

    def update(
        self,
        population: Sequence[Individual],
        first_front: Optional[Sequence[int]],
        *,
        count_stagnation: bool,
    ) -> None:
        """Refresh the best-so-far record after the population changed.

        Args:
            population: The current population.
            first_front: Indices of the Pareto front (multi-objective only).
            count_stagnation: Whether to advance/reset the stagnation counter.
                ``False`` after initialisation and after migration or restarts.

        Raises:
            RuntimeError: In multi-objective mode without ``first_front``.
        """
        if self.multi_objective:
            self._update_pareto(population, first_front, count_stagnation)
        else:
            self._update_scalar(population, count_stagnation)

    def _update_pareto(
        self,
        population: Sequence[Individual],
        first_front: Optional[Sequence[int]],
        count_stagnation: bool,
    ) -> None:
        if first_front is None:
            raise RuntimeError("The Pareto front is required in multi-objective mode")
        front = [population[i] for i in first_front]
        signature = tuple(
            sorted(
                (tuple(float(v) for v in ind.fitness), float(ind.constraint_violation))  # type: ignore[union-attr]
                for ind in front
            )
        )
        changed = signature != self.pareto_signature
        self.pareto_signature = signature
        self.best_individual = min(front, key=lambda ind: tuple(ind.fitness)).clone()  # type: ignore[arg-type]
        if count_stagnation:
            self.stagnation_count = 0 if changed else self.stagnation_count + 1

    def _update_scalar(self, population: Sequence[Individual], count_stagnation: bool) -> None:
        candidate = min(population, key=self.comparator.effective)
        if self.best_individual is None:
            self.best_individual = candidate.clone()
            return
        previous = self.comparator.effective(self.best_individual)
        current = self.comparator.effective(candidate)
        if current < previous:
            self.best_individual = candidate.clone()
            if count_stagnation:
                significant = (previous - current) > self.tolerance
                self.stagnation_count = 0 if significant else self.stagnation_count + 1
        elif count_stagnation:
            self.stagnation_count += 1

    def record(self) -> None:
        """Append the current best fitness to :attr:`history`.

        Raises:
            RuntimeError: If there is no best individual yet.
        """
        if self.best_individual is None:
            raise RuntimeError("Cannot record progress without a best individual")
        self.history.append(self.best_individual.fitness)

    def restart_due(self, threshold: Optional[int]) -> bool:
        """Whether stagnation has lasted ``threshold`` generations since the last restart."""
        return (
            threshold is not None
            and self.stagnation_count >= threshold
            and self.generations_since_restart >= threshold
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return the counters and signature needed to resume (individuals excluded)."""
        return {
            "stagnation": self.stagnation_count,
            "generations_since_restart": self.generations_since_restart,
            "pareto_signature": self.pareto_signature,
            "best_fitness_history": list(self.history),
        }

    def load_state_dict(self, state: Dict[str, Any], best: Optional[Individual]) -> None:
        """Restore the state from :meth:`state_dict` plus the best individual."""
        self.stagnation_count = int(state["stagnation"])
        self.generations_since_restart = int(state["generations_since_restart"])
        self.pareto_signature = state["pareto_signature"]
        self.history = list(state["best_fitness_history"])
        self.best_individual = best
