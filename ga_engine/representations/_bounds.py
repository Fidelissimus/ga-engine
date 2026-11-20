"""Parsing of gene bounds shared by the numeric representations."""

from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import numpy as np

from ..errors import ConfigurationError

#: ``(low, high)`` shared by all genes, or one ``(low, high)`` pair per gene.
Bounds = Union[Tuple[float, float], Sequence[Tuple[float, float]]]


def parse_bounds(dimensions: int, bounds: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Turn the user's ``bounds`` into per-gene lower and upper arrays.

    Raises:
        ConfigurationError: On a wrong shape, non-finite values or ``low >= high``.
    """
    try:
        elements = list(bounds)
    except TypeError as exc:
        raise ConfigurationError("bounds must be a (low, high) pair or one pair per gene") from exc
    try:
        if len(elements) == 2 and all(np.ndim(e) == 0 for e in elements):
            lower = np.full(dimensions, float(elements[0]))
            upper = np.full(dimensions, float(elements[1]))
        else:
            if len(elements) != dimensions or any(np.shape(e) != (2,) for e in elements):
                raise ConfigurationError(
                    f"per-gene bounds need exactly {dimensions} (low, high) pairs"
                )
            lower = np.array([float(e[0]) for e in elements])
            upper = np.array([float(e[1]) for e in elements])
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"bounds contain non-numeric values: {exc}") from exc
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ConfigurationError("bounds must be finite")
    if not np.all(lower < upper):
        raise ConfigurationError("every gene needs lower < upper")
    return lower, upper
