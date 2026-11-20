"""Binary bit-vector representation."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..enums import CrossoverMethod, MutationMethod
from ..errors import ConfigurationError
from ._exchange import single_point_exchange, two_point_exchange, uniform_exchange
from .base import Representation, get_param


class BinaryRepresentation(Representation):
    """Fixed-length vector of bits stored as ``int8``.

    Crossover: single-point, two-point and uniform (default single-point).
    Mutation: bit-flip (default).

    Operator parameters: ``probability`` for uniform crossover (default 0.5).

    Args:
        dimensions: Number of bits.

    Raises:
        ConfigurationError: If ``dimensions`` is not a positive integer.
    """

    GENOME_DTYPE = np.dtype(np.int8)
    SUPPORTED_CROSSOVERS = frozenset(
        {CrossoverMethod.SINGLE_POINT, CrossoverMethod.TWO_POINT, CrossoverMethod.UNIFORM}
    )
    SUPPORTED_MUTATIONS = frozenset({MutationMethod.BIT_FLIP})
    DEFAULT_CROSSOVER = CrossoverMethod.SINGLE_POINT
    DEFAULT_MUTATION = MutationMethod.BIT_FLIP
    CROSSOVER_PARAMS = {
        CrossoverMethod.SINGLE_POINT: frozenset(),
        CrossoverMethod.TWO_POINT: frozenset(),
        CrossoverMethod.UNIFORM: frozenset({"probability"}),
    }
    MUTATION_PARAMS = {MutationMethod.BIT_FLIP: frozenset()}

    def __init__(self, dimensions: int):
        super().__init__(dimensions, (0, 1))

    def random_individual(self, rng: np.random.Generator) -> np.ndarray:
        """Draw each bit uniformly at random."""
        return rng.integers(0, 2, self.dimensions, dtype=np.int8)

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        method: CrossoverMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recombine two parents.

        Raises:
            ConfigurationError: For an unsupported method or invalid parameter.
        """
        if method == CrossoverMethod.SINGLE_POINT:
            return single_point_exchange(parent1, parent2, rng)
        if method == CrossoverMethod.TWO_POINT:
            return two_point_exchange(parent1, parent2, rng)
        if method == CrossoverMethod.UNIFORM:
            return uniform_exchange(parent1, parent2, rng, get_param(kwargs, "probability", 0.5))
        raise ConfigurationError(f"Unsupported crossover for BinaryRepresentation: {method}")

    def mutate(
        self,
        individual: np.ndarray,
        method: MutationMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> np.ndarray:
        """Flip each bit with probability ``mutation_prob`` (default ``1/dimensions``).

        Raises:
            ConfigurationError: For an unsupported method or invalid probability.
        """
        if method != MutationMethod.BIT_FLIP:
            raise ConfigurationError(f"Unsupported mutation for BinaryRepresentation: {method}")
        probability = get_param(kwargs, "mutation_prob", 1.0 / self.dimensions)
        mutated: np.ndarray = np.asarray(individual, dtype=np.int8).copy()
        mask = rng.random(self.dimensions) < probability
        mutated[mask] = 1 - mutated[mask]
        return mutated

    def repair(self, individual: np.ndarray) -> np.ndarray:
        """Round to the nearest bit; NaN becomes 0."""
        arr = np.nan_to_num(np.asarray(individual, dtype=float), nan=0.0)
        return np.asarray(np.clip(np.rint(arr), 0, 1).astype(np.int8))

    def validate(self, individual: np.ndarray) -> bool:
        """Return whether the genome is 1-D, right-sized and contains only 0/1."""
        arr = np.asarray(individual)
        return bool(
            arr.ndim == 1 and len(arr) == self.dimensions and np.all((arr == 0) | (arr == 1))
        )
