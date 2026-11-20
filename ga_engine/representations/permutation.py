"""Permutation representation for ordering problems such as the TSP."""

from __future__ import annotations

from typing import Any, Set, Tuple

import numpy as np

from ..enums import CrossoverMethod, MutationMethod
from ..errors import ConfigurationError
from .base import Representation


class PermutationRepresentation(Representation):
    """A permutation of ``0 .. dimensions-1`` stored as ``int64``.

    Crossover: order crossover (OX, default) and partially mapped crossover
    (PMX). Mutation: swap (default), scramble and inversion.

    Args:
        dimensions: Permutation length.

    Raises:
        ConfigurationError: If ``dimensions`` is not a positive integer.
    """

    GENOME_DTYPE = np.dtype(np.int64)
    SUPPORTED_CROSSOVERS = frozenset({CrossoverMethod.ORDER, CrossoverMethod.PMX})
    SUPPORTED_MUTATIONS = frozenset(
        {MutationMethod.SWAP, MutationMethod.SCRAMBLE, MutationMethod.INVERSION}
    )
    DEFAULT_CROSSOVER = CrossoverMethod.ORDER
    DEFAULT_MUTATION = MutationMethod.SWAP
    CROSSOVER_PARAMS = {CrossoverMethod.ORDER: frozenset(), CrossoverMethod.PMX: frozenset()}
    MUTATION_PARAMS = {
        MutationMethod.SWAP: frozenset(),
        MutationMethod.SCRAMBLE: frozenset(),
        MutationMethod.INVERSION: frozenset(),
    }

    def __init__(self, dimensions: int):
        super().__init__(dimensions)

    def random_individual(self, rng: np.random.Generator) -> np.ndarray:
        """Return a uniformly random permutation."""
        return rng.permutation(self.dimensions).astype(np.int64)

    # -- crossover ----------------------------------------------------------

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        method: CrossoverMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recombine two permutations; children are always valid permutations.

        Raises:
            ConfigurationError: For an unsupported method.
        """
        if method not in (CrossoverMethod.ORDER, CrossoverMethod.PMX):
            raise ConfigurationError(
                f"Unsupported crossover for PermutationRepresentation: {method}"
            )
        if self.dimensions < 2:
            return parent1.copy(), parent2.copy()
        start, end = self._choose_segment(rng)
        fill = self._ox_fill if method == CrossoverMethod.ORDER else self._pmx_fill
        return fill(parent1, parent2, start, end), fill(parent2, parent1, start, end)

    def _choose_segment(self, rng: np.random.Generator) -> Tuple[int, int]:
        """Pick two distinct positions and return them in ascending order."""
        start, end = np.sort(rng.choice(self.dimensions, 2, replace=False))
        return int(start), int(end)

    @staticmethod
    def _ox_fill(primary: np.ndarray, donor: np.ndarray, start: int, end: int) -> np.ndarray:
        """Textbook order crossover (OX1).

        The child keeps ``primary[start:end+1]`` in place. The remaining
        positions are filled cyclically, beginning just after the segment, with
        the genes missing from the segment in the order they appear in
        ``donor`` when it is read cyclically from the same point.
        """
        size = len(primary)
        child = np.full(size, -1, dtype=np.int64)
        child[start : end + 1] = primary[start : end + 1]
        in_segment = set(primary[start : end + 1].tolist())
        cyclic = [(end + 1 + offset) % size for offset in range(size)]
        donor_genes = [int(donor[i]) for i in cyclic if int(donor[i]) not in in_segment]
        free_positions = [i for i in cyclic if not start <= i <= end]
        for position, gene in zip(free_positions, donor_genes):
            child[position] = gene
        return child

    @staticmethod
    def _pmx_fill(p1: np.ndarray, p2: np.ndarray, start: int, end: int) -> np.ndarray:
        """Partially mapped crossover: keep ``p1``'s segment, resolve conflicts by mapping."""
        size = len(p1)
        child = np.full(size, -1, dtype=np.int64)
        child[start : end + 1] = p1[start : end + 1]
        mapping = {int(p1[i]): int(p2[i]) for i in range(start, end + 1)}
        segment_values = set(child[start : end + 1].tolist())
        for i in range(size):
            if start <= i <= end:
                continue
            gene = int(p2[i])
            while gene in segment_values:
                gene = mapping[gene]
            child[i] = gene
        return child

    # -- mutation -----------------------------------------------------------

    def mutate(
        self,
        individual: np.ndarray,
        method: MutationMethod,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply swap, scramble or inversion to a copy of ``individual``.

        The permutation operators act on the whole genome, so the per-gene
        ``mutation_prob`` keyword is accepted and ignored.

        Raises:
            ConfigurationError: For an unsupported method.
        """
        if method not in self.SUPPORTED_MUTATIONS:
            raise ConfigurationError(
                f"Unsupported mutation for PermutationRepresentation: {method}"
            )
        mutated: np.ndarray = np.asarray(individual, dtype=np.int64).copy()
        if len(mutated) < 2:
            return mutated
        if method == MutationMethod.SWAP:
            i, j = rng.choice(len(mutated), 2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
            return mutated
        start, end = self._choose_segment(rng)
        if method == MutationMethod.SCRAMBLE:
            segment = mutated[start : end + 1].copy()
            rng.shuffle(segment)
            mutated[start : end + 1] = segment
        else:  # inversion
            mutated[start : end + 1] = mutated[start : end + 1][::-1]
        return mutated

    # -- repair / validation ------------------------------------------------

    def repair(self, individual: np.ndarray) -> np.ndarray:
        """Replace duplicates and out-of-range values with the missing values.

        Raises:
            ConfigurationError: If the genome has the wrong length.
        """
        arr: np.ndarray = np.asarray(individual, dtype=np.int64).copy()
        if arr.ndim != 1 or len(arr) != self.dimensions:
            raise ConfigurationError("Unable to repair permutation: wrong shape")
        expected = set(range(self.dimensions))
        missing = sorted(expected - set(arr.tolist()))
        seen: Set[int] = set()
        next_missing = 0
        for i, value in enumerate(arr.tolist()):
            if value not in expected or value in seen:
                arr[i] = missing[next_missing]
                next_missing += 1
            seen.add(int(arr[i]))
        return arr

    def validate(self, individual: np.ndarray) -> bool:
        """Return whether the genome is an integer permutation of ``0..n-1``.

        Integer-valued floats are rejected here; :meth:`repair` normalises them.
        """
        arr = np.asarray(individual)
        if arr.ndim != 1 or len(arr) != self.dimensions:
            return False
        if not np.issubdtype(arr.dtype, np.integer):
            return False
        return set(arr.tolist()) == set(range(self.dimensions))
