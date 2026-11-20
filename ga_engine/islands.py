"""Island-model coordination of several :class:`GeneticAlgorithm` populations.

Each island evolves independently; every ``migration_interval`` generations the
best individuals of each island migrate to the next island in a ring. Islands
are driven sequentially in one process (parallelism, if wanted, comes from each
island's own ``n_workers``). This module touches islands only through their
public API (``step``, ``select_migrants``, ``accept_migrants`` ...).
"""

from __future__ import annotations

import logging
import time
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .engine import GeneticAlgorithm
from .errors import ConfigurationError
from .models import Individual, RunResult
from .multiobjective import assign_rank_and_crowding
from .termination import StopCriteria, StopReason

#: Signature of the ``on_generation`` hook: return a truthy value to stop the run.
IslandCallback = Callable[["IslandModel"], Optional[bool]]

LOGGER = logging.getLogger(__name__)


class IslandModel:
    """Ring-topology island model.

    The islands must share the same genome description (representation type,
    dimensions, bounds) and objective mode so individuals can migrate between
    them. They may differ in everything else (operators, rates, population
    size, seeds), which is the usual way to add diversity.

    Args:
        islands: At least two engines.
        migration_interval: Generations between migrations (>= 1).
        migration_size: Individuals each island sends per migration; must not
            exceed the smallest island population.

    Raises:
        ConfigurationError: If the islands are incompatible or the migration
            settings are invalid.
    """

    def __init__(
        self,
        islands: Sequence[GeneticAlgorithm],
        migration_interval: int = 20,
        migration_size: int = 2,
    ) -> None:
        if len(islands) < 2:
            raise ConfigurationError("IslandModel requires at least two islands")
        if not all(isinstance(island, GeneticAlgorithm) for island in islands):
            raise ConfigurationError("every island must be a GeneticAlgorithm")
        reference = islands[0]
        for index, island in enumerate(islands[1:], start=1):
            if island.multi_objective != reference.multi_objective:
                raise ConfigurationError(
                    f"island {index} differs from island 0 in multi_objective mode"
                )
            if island.representation.describe() != reference.representation.describe():
                raise ConfigurationError(
                    f"island {index} uses a different genome description than island 0"
                )
        if migration_interval < 1:
            raise ConfigurationError("migration_interval must be >= 1")
        smallest = min(island.population_size for island in islands)
        if not 1 <= migration_size <= smallest:
            raise ConfigurationError(
                f"migration_size must be between 1 and the smallest island population ({smallest})"
            )
        self.islands: List[GeneticAlgorithm] = list(islands)
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.multi_objective = reference.multi_objective
        self._stack: Optional[ExitStack] = None

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Release every island's worker pool."""
        for island in self.islands:
            island.close()

    def __enter__(self) -> "IslandModel":
        with ExitStack() as stack:
            for island in self.islands:
                stack.enter_context(island)
            self._stack = stack.pop_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.__exit__(exc_type, exc, traceback)

    # -- migration -----------------------------------------------------------

    def _migrate(self) -> None:
        """Send each island's best individuals to the next island in the ring."""
        emigrants = [island.select_migrants(self.migration_size) for island in self.islands]
        count = len(self.islands)
        for index, island in enumerate(self.islands):
            island.accept_migrants(emigrants[(index - 1) % count])
        LOGGER.debug("Migrated %d individuals per island", self.migration_size)

    # -- run -----------------------------------------------------------------

    def run(
        self,
        max_generations: int = 500,
        stagnation_generations: Optional[int] = None,
        target_fitness: Optional[float] = None,
        resume: bool = False,
        on_generation: Optional[IslandCallback] = None,
    ) -> RunResult:
        """Evolve all islands in lock-step with periodic migration.

        Args:
            max_generations: Generation budget per island (``>= 0``).
            stagnation_generations: Stop when *every* island has stagnated for
                this many generations.
            target_fitness: Stop when *any* island reaches the target
                (single-objective only).
            resume: Continue the islands' current populations (they must all
                be initialised and at the same generation).
            on_generation: Optional hook called with this model after
                initialisation and after every generation (and its migration).
                Return a truthy value to stop (``stop_reason == "callback"``).

        Returns:
            A :class:`~ga_engine.models.RunResult`. Single-objective: the best
            solution over all islands. Multi-objective: the non-dominated set
            of the union of all island fronts. ``island_stats`` holds each
            island's statistics.

        Raises:
            ConfigurationError: On invalid arguments.
            RuntimeError: If ``resume`` is requested but the islands are not
                initialised or are out of sync.
        """
        criteria = StopCriteria(max_generations, stagnation_generations, target_fitness)
        criteria.require_single_objective(self.multi_objective)

        started = time.perf_counter()
        reason: Optional[StopReason] = None
        with ExitStack() as stack:
            for island in self.islands:
                stack.enter_context(island.evaluation_pool())
            if resume:
                self._require_resumable()
            else:
                for island in self.islands:
                    island.initialize_population()
                reason = self._callback_stop(on_generation)
            while reason is None:
                reason = criteria.check(
                    generation=self.generation,
                    stagnation_count=min(i.stagnation_count for i in self.islands),
                    target_reached=target_fitness is not None
                    and any(i.target_reached(target_fitness) for i in self.islands),
                )
                if reason is not None:
                    break
                for island in self.islands:
                    island.step()
                if self.generation % self.migration_interval == 0:
                    self._migrate()
                reason = self._callback_stop(on_generation)
        return self._build_result(reason, time.perf_counter() - started)

    @property
    def generation(self) -> int:
        """Generation counter of the islands (they advance in lock-step)."""
        return self.islands[0].generation

    def _callback_stop(self, hook: Optional[IslandCallback]) -> Optional[StopReason]:
        """Run the ``on_generation`` hook; returns ``CALLBACK`` if it asks to stop."""
        if hook is not None and hook(self):
            return StopReason.CALLBACK
        return None

    def _require_resumable(self) -> None:
        """Check that all islands hold a population and agree on the generation."""
        if any(not island.population for island in self.islands):
            raise RuntimeError("resume=True requires every island to be initialized or loaded")
        if len({island.generation for island in self.islands}) != 1:
            raise RuntimeError("resume=True requires all islands to be at the same generation")

    # -- results -------------------------------------------------------------

    def _build_result(self, reason: StopReason, elapsed: float) -> RunResult:
        """Combine the islands' states into one :class:`RunResult`."""
        island_stats = [island.get_stats() for island in self.islands]
        evaluations = sum(island.evaluations for island in self.islands)
        failures = sum(island.evaluation_failures for island in self.islands)
        pareto_front: Optional[List[Individual]] = None

        if self.multi_objective:
            pareto_front, owner = self._merged_pareto_front()
            best = min(pareto_front, key=lambda ind: tuple(ind.fitness))  # type: ignore[arg-type]
            stats = island_stats[owner[id(best)]]
        else:
            owner_index, best_island = min(
                enumerate(self.islands),
                key=lambda pair: pair[1].effective_fitness(pair[1].best_individual),  # type: ignore[arg-type]
            )
            if best_island.best_individual is None:
                raise RuntimeError("Island has no best individual; initialise it first")
            best = best_island.best_individual.clone()
            stats = island_stats[owner_index]

        return RunResult(
            best_individual=best,
            best_fitness=best.fitness,
            generation=self.generation,
            elapsed_seconds=elapsed,
            converged=reason in (StopReason.TARGET_FITNESS, StopReason.STAGNATION),
            stop_reason=reason.value,
            stats=stats,
            pareto_front=pareto_front,
            evaluations=evaluations,
            evaluation_failures=failures,
            island_stats=island_stats,
        )

    def _merged_pareto_front(self) -> Tuple[List[Individual], Dict[int, int]]:
        """Return the global non-dominated set and the island each member came from."""
        members: List[Individual] = []
        origin: List[int] = []
        for index, island in enumerate(self.islands):
            for individual in island.get_pareto_front():
                members.append(individual)
                origin.append(index)
        fronts = assign_rank_and_crowding(members)
        front = [members[i] for i in fronts[0]]
        owner = {id(members[i]): origin[i] for i in fronts[0]}
        return front, owner
