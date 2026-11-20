"""Module-level objective functions shared by tests (picklable for worker pools)."""

from __future__ import annotations

import numpy as np


def sometimes_fails(x: np.ndarray) -> float:
    """Raise for genomes whose first gene is positive, else return the sphere value."""
    if x[0] > 0.0:
        raise ValueError("boom")
    return float(np.sum(x**2))


def always_fails(x: np.ndarray) -> float:
    """Always raise."""
    raise RuntimeError("nope")


def returns_nan(x: np.ndarray) -> float:
    """Return NaN."""
    return float("nan")


def inconsistent_objectives(x: np.ndarray):
    """Return two objectives for some genomes and three for others."""
    if x[0] < 0.0:
        return float(x[0]), float(x[1])
    return float(x[0]), float(x[1]), float(x[2])


def violation_of_first_gene(x: np.ndarray) -> float:
    """Constraint ``x[0] <= 0``."""
    return float(x[0])


def flat(x: np.ndarray) -> float:
    """A fitness landscape with no gradient at all."""
    return 1.0
