"""Integer-valued representation (discrete parameters)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..enums import CrossoverMethod, MutationMethod
from ..errors import ConfigurationError
from ._bounds import Bounds, parse_bounds
from ._exchange import single_point_exchange, two_point_exchange, uniform_exchange
from .base import Representation, get_param


class IntegerRepresentation(Representation):
    """Vector of bounded integers (both bounds inclusive).

    Typical uses: tree depth, number of layers, discrete settings. Genes are
    independent; each may have its own range.

    Crossover: single-point, two-point and uniform (default uniform).
    Mutation: random reset (default) and creep.

    Operator parameters: ``probability`` for uniform crossover (default 0.5)
    and ``max_step`` for creep mutation (default 1; the step is drawn uniformly
    from ``1 .. max_step`` and its sign at random).

    Args:
        dimensions: Number of genes.
        bounds: One inclusive ``(low, high)`` pair of integers shared by every
            gene, e.g. ``(1, 10)``, or ``dimensions`` such pairs, e.g.
            ``[(1, 10), (0, 3), (50, 500)]``. Each needs ``low < high``.

    Attributes:
        lower_bounds: Per-gene inclusive lower bounds (``int64`` array).
        upper_bounds: Per-gene inclusive upper bounds.

    Raises:
        ConfigurationError: On invalid ``dimensions`` or ``bounds`` (including
            non-integer bounds).
    """

    GENOME_DTYPE = np.dtype(np.int64)
    SUPPORTED_CROSSOVERS = frozenset(
        {CrossoverMethod.SINGLE_POINT, CrossoverMethod.TWO_POINT, CrossoverMethod.UNIFORM}
    )
    SUPPORTED_MUTATIONS = frozenset({MutationMethod.RANDOM_RESET, MutationMethod.CREEP})
    DEFAULT_CROSSOVER = CrossoverMethod.UNIFORM
    DEFAULT_MUTATION = MutationMethod.RANDOM_RESET
    CROSSOVER_PARAMS = {
        CrossoverMethod.SINGLE_POINT: frozenset(),
        CrossoverMethod.TWO_POINT: frozenset(),
        CrossoverMethod.UNIFORM: frozenset({"probability"}),
    }
    MUTATION_PARAMS = {
        MutationMethod.RANDOM_RESET: frozenset(),
        MutationMethod.CREEP: frozenset({"max_step"}),
    }

    def __init__(self, dimensions: int, bounds: Bounds = (0, 1)):
        super().__init__(dimensions, None)
        lower, upper = parse_bounds(self.dimensions, bounds)
        if not (np.all(lower == np.floor(lower)) and np.all(upper == np.floor(upper))):
            raise ConfigurationError("integer bounds must be whole numbers")
        self.lower_bounds = lower.astype(np.int64)
        self.upper_bounds = upper.astype(np.int64)
        self.bounds = bounds

    def describe(self) -> Dict[str, Any]:
        """Describe type, dimensions and bounds (a pair, or one pair per gene)."""
        uniform = bool(
            np.all(self.lower_bounds == self.lower_bounds[0])
            and np.all(self.upper_bounds == self.upper_bounds[0])
        )
        if uniform:
            bounds: Any = [int(self.lower_bounds[0]), int(self.upper_bounds[0])]
        else:
            bounds = [[int(lo), int(hi)] for lo, hi in zip(self.lower_bounds, self.upper_bounds)]
        return {"type": type(self).__name__, "dimensions": self.dimensions, "bounds": bounds}

    def random_individual(self, rng: np.random.Generator) -> np.ndarray:
        """Draw each gene uniformly from its inclusive range."""
        return rng.integers(self.lower_bounds, self.upper_bounds, endpoint=True)

    def repair(self, individual: np.ndarray) -> np.ndarray:
        """Round to the nearest integer and clip into range; NaN becomes the lower bound."""
        arr = np.asarray(individual, dtype=float)
        arr = np.where(np.isnan(arr), self.lower_bounds, arr)
        arr = np.clip(np.rint(arr), self.lower_bounds, self.upper_bounds)
        return arr.astype(np.int64)

    def validate(self, individual: np.ndarray) -> bool:
        """Return whether the genome is a 1-D integer array of the right size within bounds."""
        arr = np.asarray(individual)
        return bool(
            arr.ndim == 1
            and len(arr) == self.dimensions
            and np.issubdtype(arr.dtype, np.integer)
            and np.all(arr >= self.lower_bounds)
            and np.all(arr <= self.upper_bounds)
        )

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        method: CrossoverMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recombine two parents by exchanging genes.

        Raises:
            ConfigurationError: For an unsupported method or invalid parameter.
        """
        if method == CrossoverMethod.SINGLE_POINT:
            return single_point_exchange(parent1, parent2, rng)
        if method == CrossoverMethod.TWO_POINT:
            return two_point_exchange(parent1, parent2, rng)
        if method == CrossoverMethod.UNIFORM:
            return uniform_exchange(parent1, parent2, rng, get_param(kwargs, "probability", 0.5))
        raise ConfigurationError(f"Unsupported crossover for IntegerRepresentation: {method}")

    def mutate(
        self,
        individual: np.ndarray,
        method: MutationMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> np.ndarray:
        """Mutate a copy of ``individual``.

        Keyword Args:
            mutation_prob: Per-gene mutation probability (default ``1/dimensions``).
            max_step: Largest creep step (default 1, must be >= 1).

        Raises:
            ConfigurationError: For an unsupported method or invalid parameter.
        """
        probability = get_param(kwargs, "mutation_prob", 1.0 / self.dimensions)
        mutated: np.ndarray = np.asarray(individual, dtype=np.int64).copy()
        mask = rng.random(self.dimensions) < probability
        count = int(np.sum(mask))
        if method == MutationMethod.RANDOM_RESET:
            if count:
                mutated[mask] = rng.integers(
                    self.lower_bounds[mask], self.upper_bounds[mask], endpoint=True
                )
            return mutated
        if method == MutationMethod.CREEP:
            max_step = int(get_param(kwargs, "max_step", 1.0))
            if max_step < 1:
                raise ConfigurationError("parameter 'max_step' must be >= 1")
            if count:
                steps = rng.integers(1, max_step, endpoint=True, size=count)
                signs = rng.choice(np.array([-1, 1], dtype=np.int64), size=count)
                mutated[mask] += steps * signs
            return np.asarray(np.clip(mutated, self.lower_bounds, self.upper_bounds))
        raise ConfigurationError(f"Unsupported mutation for IntegerRepresentation: {method}")
