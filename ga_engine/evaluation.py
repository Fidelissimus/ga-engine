"""Running the user's fitness and constraint functions.

:class:`Evaluator` is the only component that executes user code. It hides the
difference between serial and process-parallel evaluation, applies the failure
policy, normalises results (see :mod:`ga_engine.fitness`) and counts
evaluations and failures.

Parallel evaluation uses a ``spawn`` pool, so it behaves identically on Windows,
macOS and Linux. The user's functions are shipped to each worker **once**
through the pool initialiser (not once per task); only genomes travel per
batch. Consequently the functions must be picklable, i.e. defined at module
level (or be instances of module-level classes).
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import pickle
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .aliases import ConstraintFunction, Fitness, ObjectiveFunction
from .enums import EvaluationErrorPolicy
from .errors import EvaluationError
from .fitness import normalize_fitness, normalize_violation

LOGGER = logging.getLogger(__name__)

RawEvaluation = Tuple[Any, Any, Optional[str]]

_WORKER_FITNESS: Optional[ObjectiveFunction] = None
_WORKER_CONSTRAINT: Optional[ConstraintFunction] = None


def _call_user_functions(
    fitness_fn: ObjectiveFunction,
    constraint_fn: Optional[ConstraintFunction],
    genes: np.ndarray,
) -> RawEvaluation:
    """Call user code for one genome, converting an exception into a message.

    This is the single, deliberate boundary where arbitrary user exceptions are
    caught. They are never discarded: the message travels back to the caller,
    which logs it, counts it and applies the configured failure policy.
    ``BaseException`` subclasses such as ``KeyboardInterrupt`` still propagate.
    """
    try:
        value = fitness_fn(genes)
        violation = constraint_fn(genes) if constraint_fn is not None else 0.0
    except Exception as exc:  # noqa: BLE001 - user-code boundary, reported upstream
        return None, 0.0, f"{type(exc).__name__}: {exc}"
    return value, violation, None


def _initialize_worker(
    fitness_fn: ObjectiveFunction, constraint_fn: Optional[ConstraintFunction]
) -> None:
    """Pool initialiser: store the user functions in the worker process."""
    global _WORKER_FITNESS, _WORKER_CONSTRAINT
    _WORKER_FITNESS = fitness_fn
    _WORKER_CONSTRAINT = constraint_fn


def _evaluate_in_worker(genes: np.ndarray) -> RawEvaluation:
    """Evaluate one genome inside a pool worker."""
    if _WORKER_FITNESS is None:
        raise RuntimeError("worker process was not initialised")
    return _call_user_functions(_WORKER_FITNESS, _WORKER_CONSTRAINT, genes)


class Evaluator:
    """Evaluates batches of genomes, serially or in a process pool.

    Args:
        fitness_function: ``f(genes)`` returning a float (or a sequence of
            floats in multi-objective mode). Minimised.
        constraint_function: Optional ``g(genes)`` returning a violation
            (positive = violated) or a sequence of them.
        multi_objective: Whether ``fitness_function`` returns several objectives.
        n_workers: ``<= 1`` evaluates serially in this process.
        error_policy: How failures of user code are handled.

    Attributes:
        objective_count: Number of objectives (inferred on first use when
            multi-objective, otherwise ``None``).
        evaluations: Cumulative number of genomes evaluated.
        failures: Cumulative number of failed evaluations.
    """

    def __init__(
        self,
        fitness_function: ObjectiveFunction,
        constraint_function: Optional[ConstraintFunction] = None,
        *,
        multi_objective: bool = False,
        n_workers: int = 0,
        error_policy: EvaluationErrorPolicy = EvaluationErrorPolicy.PENALIZE,
    ) -> None:
        self.fitness_function = fitness_function
        self.constraint_function = constraint_function
        self.multi_objective = multi_objective
        self.n_workers = n_workers
        self.error_policy = error_policy
        self.objective_count: Optional[int] = None
        self.evaluations = 0
        self.failures = 0
        self._pool: Optional[Any] = None

    # -- pool lifecycle -----------------------------------------------------

    @property
    def pool_active(self) -> bool:
        """Whether a worker pool is currently running."""
        return self._pool is not None

    def start(self) -> bool:
        """Start the worker pool if parallel evaluation is configured.

        Returns:
            ``True`` if this call created a new pool (so the caller owns its
            shutdown), ``False`` if no pool is needed or one already exists.

        Raises:
            TypeError: If the user functions cannot be pickled.
        """
        if self.n_workers <= 1 or self._pool is not None:
            return False
        try:
            pickle.dumps((self.fitness_function, self.constraint_function))
        except Exception as exc:  # noqa: BLE001 - pickle raises many unrelated types
            raise TypeError(
                "Parallel evaluation requires a picklable fitness_function and "
                "constraint_function. Use module-level functions when n_workers > 1."
            ) from exc
        context = mp.get_context("spawn")
        self._pool = context.Pool(
            processes=self.n_workers,
            initializer=_initialize_worker,
            initargs=(self.fitness_function, self.constraint_function),
        )
        return True

    def close(self) -> None:
        """Shut the pool down gracefully, waiting for pending work."""
        if self._pool is None:
            return
        pool, self._pool = self._pool, None
        pool.close()
        pool.join()

    def terminate(self) -> None:
        """Kill the pool immediately (used when a run is aborted by an error)."""
        if self._pool is None:
            return
        pool, self._pool = self._pool, None
        pool.terminate()
        pool.join()

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, genes_batch: Sequence[np.ndarray]) -> Tuple[List[Fitness], List[float]]:
        """Evaluate a batch of (already repaired) genomes.

        Args:
            genes_batch: Genomes to evaluate.

        Returns:
            ``(fitnesses, violations)``, aligned with ``genes_batch``. Failed
            evaluations get an infinitely bad fitness and violation.

        Raises:
            EvaluationError: Under the ``RAISE`` policy when any evaluation
                fails, if a multi-objective function returns inconsistent
                objective counts, or if the objective count cannot be inferred.
        """
        if len(genes_batch) == 0:
            return [], []
        raw_results = self._compute_raw(genes_batch)
        self._infer_objective_count(raw_results)

        fitnesses: List[Fitness] = []
        violations: List[float] = []
        errors: List[str] = []
        for raw_value, raw_violation, error in raw_results:
            fitness: Fitness = math.inf
            violation = math.inf
            if error is None:
                try:
                    fitness = normalize_fitness(raw_value, self.multi_objective)
                    violation = normalize_violation(raw_violation)
                except (TypeError, ValueError, OverflowError) as exc:
                    error = f"invalid result ({type(exc).__name__}: {exc})"
            if error is not None:
                errors.append(error)
                fitness = self._failure_fitness()
                violation = math.inf
            elif (
                self.multi_objective
                and isinstance(fitness, tuple)
                and len(fitness) != self.objective_count
            ):
                raise EvaluationError(
                    "All individuals must return the same number of objectives "
                    f"(expected {self.objective_count}, got {len(fitness)})"
                )
            fitnesses.append(fitness)
            violations.append(violation)

        self.evaluations += len(raw_results)
        self.failures += len(errors)
        if errors:
            self._handle_failures(errors, len(raw_results))
        return fitnesses, violations

    def _compute_raw(self, genes_batch: Sequence[np.ndarray]) -> List[RawEvaluation]:
        """Run user code serially or through the pool."""
        if self.n_workers <= 1:
            return [
                _call_user_functions(self.fitness_function, self.constraint_function, g)
                for g in genes_batch
            ]
        # Manual step()/initialize_population() calls remain usable without an
        # enclosing run(); that pool then lives until close() / context exit.
        self.start()
        pool = self._pool
        if pool is None:
            raise RuntimeError("worker pool failed to start")
        results: List[RawEvaluation] = pool.map(_evaluate_in_worker, list(genes_batch))
        return results

    def _infer_objective_count(self, raw_results: Sequence[RawEvaluation]) -> None:
        """Learn the number of objectives from the first usable result."""
        if not self.multi_objective or self.objective_count is not None:
            return
        for raw_value, _violation, error in raw_results:
            if error is not None:
                continue
            try:
                fitness = normalize_fitness(raw_value, True)
            except (TypeError, ValueError, OverflowError):
                continue
            if isinstance(fitness, tuple):
                self.objective_count = len(fitness)
                return
        raise EvaluationError(
            "Every multi-objective fitness evaluation failed, so the objective "
            "count cannot be inferred."
        )

    def _failure_fitness(self) -> Fitness:
        """Fitness assigned to a failed evaluation."""
        if self.multi_objective and self.objective_count:
            return tuple(math.inf for _ in range(self.objective_count))
        return math.inf

    def _handle_failures(self, errors: Sequence[str], total: int) -> None:
        """Apply the failure policy to a batch that contained failures."""
        message = f"{len(errors)} of {total} fitness evaluations failed; first error: {errors[0]}"
        if self.error_policy == EvaluationErrorPolicy.RAISE:
            raise EvaluationError(message)
        LOGGER.warning(message)


__all__ = ["Evaluator"]
