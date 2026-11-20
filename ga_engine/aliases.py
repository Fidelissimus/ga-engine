"""Shared type aliases."""

from __future__ import annotations

from typing import Any, Callable, Tuple, Union

import numpy as np

#: A scalar (single-objective) or a tuple of objective values (multi-objective).
Fitness = Union[float, Tuple[float, ...]]

#: ``f(genes) -> float | sequence of floats``. Every objective is *minimised*.
ObjectiveFunction = Callable[[np.ndarray], Any]

#: ``g(genes) -> float | sequence of floats``. Positive values mean violation.
ConstraintFunction = Callable[[np.ndarray], Any]
