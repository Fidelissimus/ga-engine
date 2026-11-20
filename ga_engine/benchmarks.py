"""Ready-made test problems for examples, tests and quick experiments.

Everything here is defined at module level, so the callables are picklable and
work with ``n_workers > 1`` on every platform (including Windows ``spawn``).
All problems are posed as *minimisation*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def sphere(x: np.ndarray) -> float:
    """Sphere function; minimum 0 at the origin."""
    return float(np.sum(np.square(x)))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function (highly multimodal); minimum 0 at the origin, usually in [-5.12, 5.12]."""
    return float(10.0 * len(x) + np.sum(np.square(x) - 10.0 * np.cos(2.0 * math.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock valley; minimum 0 at (1, ..., 1). Needs at least two dimensions."""
    return float(np.sum(100.0 * np.square(x[1:] - np.square(x[:-1])) + np.square(1.0 - x[:-1])))


def ackley(x: np.ndarray) -> float:
    """Ackley function; minimum 0 at the origin."""
    n = len(x)
    return float(
        -20.0 * math.exp(-0.2 * math.sqrt(np.sum(np.square(x)) / n))
        - math.exp(np.sum(np.cos(2.0 * math.pi * x)) / n)
        + 20.0
        + math.e
    )


def zdt1(x: np.ndarray) -> Tuple[float, float]:
    """ZDT1 two-objective problem on ``[0, 1]^n`` (n >= 2).

    The true Pareto front is ``f2 = 1 - sqrt(f1)`` for ``f1`` in ``[0, 1]``.
    """
    f1 = float(x[0])
    g = 1.0 + 9.0 * float(np.mean(x[1:]))
    return f1, g * (1.0 - math.sqrt(f1 / g))


def onemax(bits: np.ndarray) -> float:
    """Number of zero bits; minimum 0 for the all-ones string."""
    return float(len(bits) - int(np.sum(bits)))


@dataclass(eq=False)
class KnapsackProblem:
    """0/1 knapsack posed as constrained minimisation.

    ``objective`` is the negated total value; ``constraint`` is the amount by
    which the total weight exceeds the capacity (0 when feasible).
    """

    values: np.ndarray
    weights: np.ndarray
    capacity: float

    @classmethod
    def random(cls, n_items: int, seed: int = 0, capacity_ratio: float = 0.4) -> "KnapsackProblem":
        """Create a random instance whose capacity is a fraction of the total weight."""
        rng = np.random.default_rng(seed)
        weights = rng.integers(5, 50, n_items).astype(float)
        values = rng.integers(10, 100, n_items).astype(float)
        return cls(values, weights, float(capacity_ratio * weights.sum()))

    def objective(self, bits: np.ndarray) -> float:
        """Negative total value of the selected items."""
        return -float(np.dot(self.values, bits))

    def constraint(self, bits: np.ndarray) -> float:
        """Overweight of the selection (0 if it fits)."""
        return max(0.0, float(np.dot(self.weights, bits)) - self.capacity)


@dataclass(eq=False)
class TspProblem:
    """Euclidean travelling-salesman problem; a genome is a tour (permutation)."""

    coordinates: np.ndarray

    @classmethod
    def random(cls, n_cities: int, seed: int = 0) -> "TspProblem":
        """Create ``n_cities`` uniformly random points in the unit square."""
        return cls(np.random.default_rng(seed).random((n_cities, 2)))

    def __call__(self, tour: np.ndarray) -> float:
        """Length of the closed tour."""
        points = self.coordinates[tour]
        legs = points - np.roll(points, -1, axis=0)
        return float(np.sum(np.sqrt(np.sum(np.square(legs), axis=1))))
