"""Fitness-value semantics: normalisation, penalties and comparison.

Everything here is a pure function (or a tiny stateless class) over fitness
values, which keeps the rules for "what counts as better" in one place that is
shared by selection, replacement, adaptation and the island model.

All objectives are minimised. Non-finite handling:

* ``NaN`` objective values are mapped to ``+inf`` (worst possible).
* ``-inf`` is rejected: it would make an individual unbeatable and break
  fitness-proportionate weights, so it is treated as an evaluation failure.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np

from .aliases import Fitness
from .models import Individual


def _to_python(raw: Any) -> Any:
    """Convert NumPy scalars/arrays to plain Python values."""
    if isinstance(raw, np.ndarray):
        return raw.item() if raw.ndim == 0 else raw.tolist()
    return raw


def _clean_objective(value: Any) -> float:
    """Convert one objective value to ``float``, mapping NaN to ``+inf``.

    Raises:
        ValueError: If the value is ``-inf`` or not convertible to float.
    """
    number = float(value)
    if math.isnan(number):
        return math.inf
    if number == -math.inf:
        raise ValueError("objective value is -inf")
    return number


def normalize_fitness(raw: Any, multi_objective: bool) -> Fitness:
    """Validate and normalise whatever the user's objective function returned.

    Args:
        raw: Return value of the objective function.
        multi_objective: Whether a tuple of >= 2 objectives is expected.

    Returns:
        A ``float`` (single-objective) or a ``tuple`` of floats.

    Raises:
        ValueError: If the shape does not match the objective mode or a value
            is ``-inf``.
        TypeError: If a value cannot be converted to ``float``.
    """
    value = _to_python(raw)
    if multi_objective:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Multi-objective fitness must be a sequence of at least two numbers")
        if len(value) < 2:
            raise ValueError("Multi-objective fitness must contain at least two objectives")
        return tuple(_clean_objective(v) for v in value)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Single-objective fitness must be a scalar or a one-element sequence")
        value = value[0]
    return _clean_objective(value)


def normalize_violation(raw: Any) -> float:
    """Collapse a constraint function's output to one non-negative violation.

    A scalar is clipped at zero; a sequence contributes the sum of its positive
    parts. ``NaN`` anywhere means "unknown", which is treated as ``+inf``
    (maximally infeasible).

    Raises:
        TypeError: If a value cannot be converted to ``float``.
        ValueError: If a value cannot be converted to ``float``.
    """
    value = _to_python(raw)
    parts = value if isinstance(value, (list, tuple)) else [value]
    total = 0.0
    for part in parts:
        number = float(part)
        if math.isnan(number):
            return math.inf
        total += max(0.0, number)
    return total


def penalized_fitness(fitness: float, violation: float, penalty: float) -> float:
    """Return ``fitness + penalty * violation`` without producing NaN.

    A zero penalty means constraints are ignored by the scalar comparison, even
    when the violation is infinite (``0 * inf`` would otherwise be NaN).
    """
    if violation == 0.0 or penalty == 0.0:
        return float(fitness)
    return float(fitness) + penalty * violation


def dominates(first: Tuple[float, ...], second: Tuple[float, ...]) -> bool:
    """Return whether ``first`` Pareto-dominates ``second`` (minimisation)."""
    return all(a <= b for a, b in zip(first, second)) and any(
        a < b for a, b in zip(first, second)
    )


class FitnessComparator:
    """Decides which of two evaluated solutions is better.

    Single-objective comparison uses the additive penalty
    ``fitness + penalty * violation``. Multi-objective comparison uses
    constrained Pareto dominance: feasible beats infeasible; two infeasible
    solutions compare by violation; two feasible ones by Pareto dominance.

    Args:
        multi_objective: Selects the comparison rule.
        penalty: Weight of the constraint violation (single-objective only).
    """

    def __init__(self, multi_objective: bool, penalty: float) -> None:
        self.multi_objective = multi_objective
        self.penalty = penalty

    def effective(self, individual: Individual) -> float:
        """Penalised scalar fitness (single-objective only; lower is better)."""
        return penalized_fitness(
            individual.fitness, individual.constraint_violation, self.penalty  # type: ignore[arg-type]
        )

    def is_better(
        self,
        first: Fitness,
        second: Fitness,
        first_violation: float = 0.0,
        second_violation: float = 0.0,
    ) -> bool:
        """Return whether ``first`` is strictly better than ``second``."""
        if self.multi_objective:
            first_ok = first_violation <= 0.0
            second_ok = second_violation <= 0.0
            if first_ok and not second_ok:
                return True
            if second_ok and not first_ok:
                return False
            if not first_ok and not second_ok:
                return first_violation < second_violation
            return dominates(tuple(first), tuple(second))  # type: ignore[arg-type]
        return penalized_fitness(first, first_violation, self.penalty) < penalized_fitness(  # type: ignore[arg-type]
            second, second_violation, self.penalty  # type: ignore[arg-type]
        )

    def better_of(self, first: Individual, second: Individual) -> Individual:
        """Return the strictly better individual, or ``second`` on a tie."""
        if self.is_better(
            first.fitness,
            second.fitness,
            first.constraint_violation,
            second.constraint_violation,
        ):
            return first
        return second
