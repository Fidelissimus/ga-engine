"""The :class:`GeneticAlgorithm` orchestrator.

The engine owns the *population* and sequences one generation:

    produce offspring -> evaluate -> survive -> adapt -> track progress -> record

Every decision is delegated to a component that is exposed as an attribute:

=================  =========================================================
``reproducer``     parent selection, recombination, mutation (``reproduction``)
``evaluator``      running user code, serially or in a pool (``evaluation``)
``survival``       building the next population (``replacement``)
``adaptation``     mutation intensity (``adaptation``)
``progress``       best-so-far, stagnation, restarts (``progress``)
``statistics``     per-generation metrics (``statistics``)
``comparator``     what "better" means (``fitness``)
=================  =========================================================

Checkpoint mapping lives in :mod:`ga_engine.snapshots`, stopping rules in
:mod:`ga_engine.termination`, island migration helpers in
:mod:`ga_engine.migration`. All objectives are minimised.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .adaptation import AdaptiveMutationController
from .aliases import ConstraintFunction, Fitness, ObjectiveFunction
from .checkpoint import CHECKPOINT_VERSION, PathLike, read_snapshot, write_snapshot
from .config import GAConfig, build_config
from .enums import (
    CrossoverMethod,
    EvaluationErrorPolicy,
    MutationMethod,
    ReplacementMethod,
    SelectionMethod,
)
from .errors import ConfigurationError
from .evaluation import Evaluator
from .fitness import FitnessComparator
from .logging_utils import enable_console_logging
from .migration import merge_immigrants, select_emigrants
from .models import Individual, RunResult
from .multiobjective import assign_rank_and_crowding, nsga2_order
from .progress import ProgressTracker
from .replacement import SurvivalPolicy
from .reproduction import Reproducer
from .representations.base import Representation
from .selection import ParentSelector
from .snapshots import capture_snapshot, install_snapshot, validate_snapshot
from .statistics import GenerationStatistics
from .termination import StopCriteria, StopReason

LOGGER = logging.getLogger(__name__)

#: Signature of the ``on_generation`` hook: return a truthy value to stop the run.
GenerationCallback = Callable[["GeneticAlgorithm"], Optional[bool]]


class GeneticAlgorithm:
    """Single-population genetic algorithm with optional NSGA-II mode.

    The engine can be used as a context manager; leaving the ``with`` block
    shuts down any worker pool (aborting it if an exception is propagating)::

        with GeneticAlgorithm(100, RealRepresentation(10, (-5, 5)), sphere) as ga:
            result = ga.run(max_generations=200)

    The engine is not thread-safe.

    Args:
        population_size: Number of individuals (>= 2).
        representation: Genome encoding and operators.
        fitness_function: ``f(genes)`` -> float, or a sequence of floats when
            ``multi_objective`` is true. Minimised.
        selection_method: Parent-selection strategy.
        crossover_method: Recombination operator; ``None`` uses the
            representation's default.
        mutation_method: Mutation operator; ``None`` uses the representation's
            default.
        replacement_method: ``GENERATIONAL`` replaces the whole population;
            ``STEADY_STATE`` creates only a few offspring per generation that
            compete with the population (see ``steady_state_offspring``).
        crossover_prob: Probability a mating pair is recombined.
        mutation_prob: Probability an offspring is mutated.
        elitism_count: Elites kept by generational replacement.
        tournament_size: Contestants per tournament.
        rng_seed: Seed for the engine's private random generator.
        multi_objective: Enable NSGA-II (``fitness_function`` returns a sequence).
        n_workers: Processes for parallel evaluation; ``<= 1`` is serial and
            ``-1`` uses all CPUs. Functions must be picklable (module level).
        adaptive_mutation: Adapt mutation intensity with the 1/5 success rule.
        restart_on_stagnation: After this many stagnating generations, replace
            the worst ~10 % of the population with random individuals.
        verbose: Attach console logging at DEBUG level.
        gene_mutation_prob: Per-gene mutation probability (default ``1/dimensions``).
        mutation_sigma: Gaussian step size as a fraction of the gene range.
        constraint_function: ``g(genes)`` -> violation (positive = violated) or
            a sequence of violations.
        constraint_penalty: Weight of violations in single-objective comparison.
        crossover_params: Extra crossover keyword arguments, e.g. ``{"eta": 15}``.
        mutation_params: Extra mutation keyword arguments, e.g. ``{"eta_m": 30}``.
        steady_state_offspring: Offspring per generation in ``STEADY_STATE``
            mode (default ``max(2, population_size // 10)``; pass
            ``population_size`` for a classic (mu + lambda) strategy).
        adaptation_window: Minimum mutation trials before adapting.
        stagnation_tolerance: Improvements smaller than this do not reset the
            stagnation counter (single-objective only).
        evaluation_error_policy: ``PENALIZE`` (default) or ``RAISE`` when user
            code fails.

    Attributes:
        config: The resolved, immutable :class:`~ga_engine.config.GAConfig`.
        population: Current individuals (empty until initialised).
        generation: Number of completed generations.
        rng: The engine's ``numpy.random.Generator``.

    Raises:
        ConfigurationError: If any argument is invalid or the representation
            does not support the requested operators.
    """

    CHECKPOINT_VERSION = CHECKPOINT_VERSION

    def __init__(
        self,
        population_size: int,
        representation: Representation,
        fitness_function: ObjectiveFunction,
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        crossover_method: Optional[CrossoverMethod] = None,
        mutation_method: Optional[MutationMethod] = None,
        replacement_method: ReplacementMethod = ReplacementMethod.GENERATIONAL,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        elitism_count: int = 1,
        tournament_size: int = 3,
        rng_seed: Optional[int] = None,
        multi_objective: bool = False,
        n_workers: int = 0,
        adaptive_mutation: bool = False,
        restart_on_stagnation: Optional[int] = None,
        verbose: bool = False,
        gene_mutation_prob: Optional[float] = None,
        mutation_sigma: Optional[float] = None,
        constraint_function: Optional[ConstraintFunction] = None,
        constraint_penalty: float = 1.0,
        *,
        crossover_params: Optional[Mapping[str, Any]] = None,
        mutation_params: Optional[Mapping[str, Any]] = None,
        steady_state_offspring: Optional[int] = None,
        adaptation_window: int = 10,
        stagnation_tolerance: float = 0.0,
        evaluation_error_policy: EvaluationErrorPolicy = EvaluationErrorPolicy.PENALIZE,
    ) -> None:
        if not isinstance(representation, Representation):
            raise ConfigurationError("representation must be a Representation instance")
        if not callable(fitness_function):
            raise ConfigurationError("fitness_function must be callable")
        if constraint_function is not None and not callable(constraint_function):
            raise ConfigurationError("constraint_function must be callable or None")

        config = build_config(
            population_size=population_size,
            representation=representation,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            replacement_method=replacement_method,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            elitism_count=elitism_count,
            tournament_size=tournament_size,
            multi_objective=multi_objective,
            n_workers=n_workers,
            adaptive_mutation=adaptive_mutation,
            restart_on_stagnation=restart_on_stagnation,
            verbose=verbose,
            gene_mutation_prob=gene_mutation_prob,
            mutation_sigma=mutation_sigma,
            constraint_penalty=constraint_penalty,
            crossover_params=crossover_params,
            mutation_params=mutation_params,
            steady_state_offspring=steady_state_offspring,
            adaptation_window=adaptation_window,
            stagnation_tolerance=stagnation_tolerance,
            evaluation_error_policy=evaluation_error_policy,
        )
        if config.verbose:
            enable_console_logging(logging.DEBUG)

        self.config: GAConfig = config
        self.representation = representation
        self.fitness_function = fitness_function
        self.constraint_function = constraint_function
        self.population_size = config.population_size
        self.multi_objective = config.multi_objective
        self.n_workers = config.n_workers
        self.rng = np.random.default_rng(rng_seed)

        self.comparator = FitnessComparator(config.multi_objective, config.constraint_penalty)
        self.evaluator = Evaluator(
            fitness_function,
            constraint_function,
            multi_objective=config.multi_objective,
            n_workers=config.n_workers,
            error_policy=config.evaluation_error_policy,
        )
        self.adaptation = AdaptiveMutationController(
            enabled=config.adaptive_mutation,
            mutation_prob=config.mutation_prob,
            gene_mutation_prob=config.gene_mutation_prob,
            sigma=config.mutation_sigma,
            dimensions=representation.dimensions,
            adapt_sigma=config.mutation_method == MutationMethod.GAUSSIAN,
            window=config.adaptation_window,
        )
        self.reproducer = Reproducer(
            representation,
            config,
            ParentSelector(config.selection_method, config.tournament_size),
            self.comparator,
            self.adaptation,
            self.rng,
        )
        self.survival = SurvivalPolicy(
            size=config.population_size,
            multi_objective=config.multi_objective,
            generational=config.replacement_method == ReplacementMethod.GENERATIONAL,
            elitism_count=config.elitism_count,
            score=self.comparator.effective,
        )
        self.progress = ProgressTracker(
            self.comparator, config.multi_objective, config.stagnation_tolerance
        )
        self.statistics = GenerationStatistics()

        self.population: List[Individual] = []
        self.generation = 0
        self._front_size = 0

    # ------------------------------------------------------------------
    # Read-only views and lifecycle
    # ------------------------------------------------------------------

    @property
    def best_individual(self) -> Optional[Individual]:
        """Best (single-objective) or representative (multi-objective) solution so far."""
        return self.progress.best_individual

    @property
    def stagnation_count(self) -> int:
        """Generations since the last (significant) improvement."""
        return self.progress.stagnation_count

    @property
    def best_fitness_history(self) -> List[Fitness]:
        """Best-so-far fitness after every generation (index 0 = initial population)."""
        return self.progress.history

    @property
    def mutation_prob(self) -> float:
        """Current probability that an offspring is mutated (may adapt)."""
        return self.adaptation.mutation_prob

    @mutation_prob.setter
    def mutation_prob(self, value: float) -> None:
        self.adaptation.mutation_prob = float(value)

    @property
    def gene_mutation_prob(self) -> float:
        """Current per-gene mutation probability (may adapt)."""
        return self.adaptation.gene_mutation_prob

    @gene_mutation_prob.setter
    def gene_mutation_prob(self, value: float) -> None:
        self.adaptation.gene_mutation_prob = float(value)

    @property
    def stats(self) -> Dict[str, List[Any]]:
        """Live statistics dictionary; use :meth:`get_stats` for a safe copy."""
        return self.statistics.data

    @property
    def evaluations(self) -> int:
        """Fitness evaluations performed since the population was initialised."""
        return self.evaluator.evaluations

    @property
    def evaluation_failures(self) -> int:
        """How many of those evaluations failed."""
        return self.evaluator.failures

    @contextmanager
    def evaluation_pool(self) -> Iterator[None]:
        """Keep the worker pool alive for the duration of a ``with`` block.

        The pool is started if parallel evaluation is configured and not yet
        running, and shut down on exit only if this call started it. On error
        the pool is terminated instead of drained. ``run()`` uses this
        internally; call it yourself to share one pool across several steps.
        """
        started = self.evaluator.start()
        try:
            yield
        except BaseException:
            if started:
                self.evaluator.terminate()
            raise
        if started:
            self.evaluator.close()

    def close(self) -> None:
        """Release the worker pool, if any. Safe to call repeatedly."""
        self.evaluator.close()

    def __enter__(self) -> "GeneticAlgorithm":
        self.evaluator.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.evaluator.close()
        else:
            self.evaluator.terminate()

    def effective_fitness(self, individual: Individual) -> float:
        """Penalised scalar fitness used for single-objective comparison."""
        return self.comparator.effective(individual)

    # ------------------------------------------------------------------
    # Evaluation and population construction
    # ------------------------------------------------------------------

    def _evaluate(
        self, genes_batch: Sequence[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Fitness], List[float]]:
        """Make genomes valid, then evaluate them; returns the valid genomes too."""
        prepared = [self.representation.make_valid(g) for g in genes_batch]
        fitnesses, violations = self.evaluator.evaluate(prepared)
        return prepared, fitnesses, violations

    def _newborns(self, genes_batch: Sequence[np.ndarray]) -> List[Individual]:
        """Evaluate genomes and wrap them as newborn individuals."""
        prepared, fitnesses, violations = self._evaluate(genes_batch)
        return [
            Individual(g, f, age=0, constraint_violation=v)
            for g, f, v in zip(prepared, fitnesses, violations)
        ]

    def _reset_run_state(self) -> None:
        """Forget everything about any previous run."""
        self.generation = 0
        self._front_size = 0
        self.progress.reset()
        self.adaptation.reset()
        self.statistics.reset()
        self.evaluator.evaluations = 0
        self.evaluator.failures = 0

    def initialize_population(
        self, initial_genomes: Optional[Sequence[np.ndarray]] = None
    ) -> None:
        """Create and evaluate a population, starting a new run.

        Args:
            initial_genomes: Optional starting genomes (for example known good
                solutions). At most ``population_size`` are used; the remaining
                slots are filled randomly. Each is repaired into validity.

        Raises:
            ConfigurationError: If more genomes than ``population_size`` are
                given, or a genome has the wrong length.
        """
        seeds = [self.representation.make_valid(g) for g in (initial_genomes or [])]
        if len(seeds) > self.population_size:
            raise ConfigurationError(
                f"got {len(seeds)} initial genomes for a population of {self.population_size}"
            )
        self._reset_run_state()
        genes = seeds + self.reproducer.random_genomes(self.population_size - len(seeds))
        self.population = self._newborns(genes)
        self._finalize(record_statistics=True, count_stagnation=False)

    # ------------------------------------------------------------------
    # Variation and survival (thin wrappers over the components)
    # ------------------------------------------------------------------

    def select_parents(self) -> Tuple[Individual, Individual]:
        """Select two distinct parents and return clones of them.

        Raises:
            RuntimeError: If the population has not been initialised.
        """
        if not self.population:
            raise RuntimeError("Population is not initialized")
        if self.multi_objective:
            self.refresh_ranking()
        first, second = self.reproducer.select_parent_indices(self.population, 2, distinct=True)
        return self.population[first].clone(), self.population[second].clone()

    def create_offspring(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """Create, evaluate and return two children of the given parents.

        This is the low-level building block of :meth:`step`; it also feeds the
        adaptive-mutation statistics. It does not touch the population.
        """
        children = self.reproducer.make_children(parent1, parent2)
        offspring = self._newborns([genes for genes, _ in children])
        for child, (_, trial) in zip(offspring, children):
            self.reproducer.record_trial(child, trial)
        return offspring

    def replacement(self, offspring: Sequence[Individual]) -> None:
        """Form the next population from the current one and ``offspring``.

        See :class:`~ga_engine.replacement.SurvivalPolicy` for the rules.

        Raises:
            ConfigurationError: If the offspring count does not suit the scheme.
        """
        self.population = self.survival.select(self.population, offspring)

    # ------------------------------------------------------------------
    # Per-generation bookkeeping
    # ------------------------------------------------------------------

    def refresh_ranking(self) -> List[List[int]]:
        """Recompute NSGA-II ranks and crowding distances; returns the fronts."""
        fronts = assign_rank_and_crowding(self.population)
        self._front_size = len(fronts[0]) if fronts else 0
        return fronts

    def _finalize(self, record_statistics: bool, count_stagnation: bool) -> None:
        """Refresh ranking and progress after the population changed."""
        fronts = self.refresh_ranking() if self.multi_objective else None
        self.progress.update(
            self.population, fronts[0] if fronts else None, count_stagnation=count_stagnation
        )
        if record_statistics:
            self._record_statistics()

    def _record_statistics(self) -> None:
        """Append this generation's best-so-far fitness and statistics row."""
        self.progress.record()
        self.statistics.record(
            generation=self.generation,
            population=self.population,
            representation=self.representation,
            multi_objective=self.multi_objective,
            pareto_front_size=self._front_size,
            mutation_prob=self.adaptation.mutation_prob,
            gene_mutation_prob=self.adaptation.gene_mutation_prob,
            success_ratio=self.adaptation.last_success_ratio,
            evaluations=self.evaluator.evaluations,
            evaluation_failures=self.evaluator.failures,
        )

    # ------------------------------------------------------------------
    # Restart on stagnation
    # ------------------------------------------------------------------

    def _worst_indices(self, count: int) -> List[int]:
        """Indices of the ``count`` worst individuals."""
        if self.multi_objective:
            return nsga2_order(self.population)[-count:]
        ranked = sorted(
            range(len(self.population)),
            key=lambda i: self.comparator.effective(self.population[i]),
            reverse=True,
        )
        return ranked[:count]

    def _inject_random_individuals(self, count: int) -> None:
        """Replace the worst individuals with fresh random ones.

        The best individual always survives. Newcomers *replace* the worst
        members instead of competing with them: random genomes are almost
        always worse than a converged population, so a competition would
        discard them immediately.
        """
        count = min(count, self.population_size - 1)
        if count <= 0:
            return
        newcomers = self._newborns(self.reproducer.random_genomes(count))
        for slot, newcomer in zip(self._worst_indices(count), newcomers):
            self.population[slot] = newcomer

    def _maybe_restart(self) -> None:
        """Inject random individuals when stagnation has lasted long enough."""
        if not self.progress.restart_due(self.config.restart_on_stagnation):
            return
        count = max(1, self.population_size // 10)
        LOGGER.info(
            "Generation %d: stagnation for %d generations, injecting %d random individuals",
            self.generation,
            self.progress.stagnation_count,
            count,
        )
        self._inject_random_individuals(count)
        self.progress.generations_since_restart = 0
        self._finalize(record_statistics=False, count_stagnation=False)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the search by one generation (initialising it if necessary).

        In ``STEADY_STATE`` mode a "generation" is one step that creates only
        ``steady_state_offspring`` children, so many more generations are needed
        for the same number of evaluations.
        """
        if not self.population:
            self.initialize_population()

        genes, trials = self.reproducer.produce(
            self.population, self.config.offspring_per_generation
        )
        offspring = self._newborns(genes)
        for child, trial in zip(offspring, trials):
            self.reproducer.record_trial(child, trial)

        self.replacement(offspring)
        self.adaptation.update()
        self.generation += 1
        self.progress.generations_since_restart += 1
        self._finalize(record_statistics=False, count_stagnation=True)
        self._maybe_restart()
        self._record_statistics()
        LOGGER.debug(
            "generation=%d best=%s stagnation=%d",
            self.generation,
            self.best_individual.fitness if self.best_individual else None,
            self.stagnation_count,
        )

    def target_reached(self, target_fitness: float) -> bool:
        """Whether the best solution is feasible and at least as good as the target."""
        best = self.best_individual
        return (
            best is not None
            and best.constraint_violation <= 0.0
            and float(best.fitness) <= float(target_fitness)  # type: ignore[arg-type]
        )

    def run(
        self,
        max_generations: int = 500,
        stagnation_generations: Optional[int] = None,
        target_fitness: Optional[float] = None,
        resume: bool = False,
        on_generation: Optional[GenerationCallback] = None,
    ) -> RunResult:
        """Evolve until a stopping criterion is met.

        Args:
            max_generations: Generation budget (``>= 0``); the run stops when
                ``generation`` reaches it.
            stagnation_generations: Stop after this many generations without
                improvement (``>= 1``), or ``None`` to disable.
            target_fitness: Stop once a feasible solution at least this good is
                found (single-objective only).
            resume: Continue the current/loaded population instead of
                initialising a new one.
            on_generation: Optional hook called with the engine after the
                initial population is evaluated and after every generation.
                Return a truthy value to stop the run (``stop_reason`` is then
                ``"callback"``). Use it for progress output or custom criteria.

        Returns:
            A :class:`~ga_engine.models.RunResult`. ``stop_reason`` is
            ``"target_fitness"``, ``"stagnation"``, ``"callback"`` or
            ``"max_generations"``.

        Raises:
            ConfigurationError: On invalid arguments.
            RuntimeError: If ``resume`` is requested without a population.
            EvaluationError: Under the ``RAISE`` failure policy.
        """
        criteria = StopCriteria(max_generations, stagnation_generations, target_fitness)
        criteria.require_single_objective(self.multi_objective)

        started = time.perf_counter()
        reason: Optional[StopReason] = None
        with self.evaluation_pool():
            if resume:
                if not self.population:
                    raise RuntimeError("resume=True requires an initialized or loaded population")
            else:
                self.initialize_population()
                reason = self._callback_stop(on_generation)
            while reason is None:
                reason = criteria.check(
                    generation=self.generation,
                    stagnation_count=self.stagnation_count,
                    target_reached=target_fitness is not None
                    and self.target_reached(target_fitness),
                )
                if reason is not None:
                    break
                self.step()
                reason = self._callback_stop(on_generation)
        return self._build_result(reason, time.perf_counter() - started)

    def _callback_stop(self, hook: Optional[GenerationCallback]) -> Optional[StopReason]:
        """Run the ``on_generation`` hook; returns ``CALLBACK`` if it asks to stop."""
        if hook is not None and hook(self):
            return StopReason.CALLBACK
        return None

    def _build_result(self, reason: StopReason, elapsed: float) -> RunResult:
        """Package the engine state as a :class:`RunResult`."""
        best = self.best_individual
        if best is None:
            raise RuntimeError("Run finished without a best individual")
        LOGGER.info("Run finished at generation %d (%s)", self.generation, reason.value)
        return RunResult(
            best_individual=best.clone(),
            best_fitness=best.fitness,
            generation=self.generation,
            elapsed_seconds=elapsed,
            converged=reason in (StopReason.TARGET_FITNESS, StopReason.STAGNATION),
            stop_reason=reason.value,
            stats=self.get_stats(),
            pareto_front=self.get_pareto_front() if self.multi_objective else None,
            evaluations=self.evaluator.evaluations,
            evaluation_failures=self.evaluator.failures,
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_best(self) -> Tuple[np.ndarray, Fitness]:
        """Return ``(genes, fitness)`` of the best (or representative) solution.

        Raises:
            RuntimeError: If no population exists yet.
        """
        best = self.best_individual
        if best is None:
            raise RuntimeError(
                "No best individual yet - call run() or initialize_population() first"
            )
        return best.genes.copy(), best.fitness

    def get_stats(self) -> Dict[str, List[Any]]:
        """Return a deep copy of the accumulated generation statistics."""
        return self.statistics.copy()

    def get_pareto_front(self) -> List[Individual]:
        """Return clones of the current non-dominated set.

        Raises:
            ConfigurationError: If the engine is not in multi-objective mode.
        """
        if not self.multi_objective:
            raise ConfigurationError("Pareto front only available in multi-objective mode")
        if not self.population:
            return []
        fronts = self.refresh_ranking()
        return [self.population[i].clone() for i in fronts[0]]

    # ------------------------------------------------------------------
    # Island-model support
    # ------------------------------------------------------------------

    def select_migrants(self, count: int) -> List[Individual]:
        """Return clones of the ``count`` best individuals for emigration."""
        return select_emigrants(
            self.population,
            count,
            multi_objective=self.multi_objective,
            score=self.comparator.effective,
        )

    def accept_migrants(self, migrants: Sequence[Individual]) -> None:
        """Merge immigrants in, keeping the best ``population_size`` individuals."""
        if not migrants:
            return
        self.population = merge_immigrants(
            self.population,
            migrants,
            self.population_size,
            multi_objective=self.multi_objective,
            score=self.comparator.effective,
        )
        self._finalize(record_statistics=False, count_stagnation=False)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: PathLike) -> None:
        """Persist the complete run state to a ``.json`` or ``.npz`` file.

        The write is atomic. The fitness function and the engine settings are
        *not* stored; see ``docs/checkpoints.md``.

        Raises:
            CheckpointError: If the suffix is unsupported or the population is empty.
        """
        write_snapshot(path, capture_snapshot(self))

    def load_checkpoint(self, path: PathLike) -> None:
        """Restore run state from a checkpoint written by :meth:`save_checkpoint`.

        The engine must be configured compatibly (population size,
        representation type, dimensions, bounds, objective mode). Loading is
        all-or-nothing: the file is parsed and validated completely before any
        engine state changes. Continue with ``run(..., resume=True)``.

        Raises:
            CheckpointError: If the file is corrupt or incompatible.
            FileNotFoundError: If the file does not exist.
        """
        snapshot = read_snapshot(path)
        validate_snapshot(self, snapshot)
        install_snapshot(self, snapshot)
