"""Offspring production: parent selection, recombination and mutation.

:class:`Reproducer` turns a population into *unevaluated* child genomes. It
reads the current mutation intensity from the adaptation controller and
reports, for every child, what it has to beat to count as a mutation success.
Evaluation and survival are other components' jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .adaptation import AdaptiveMutationController
from .aliases import Fitness
from .config import GAConfig
from .enums import MutationMethod
from .fitness import FitnessComparator
from .models import Individual
from .representations.base import Representation
from .selection import ParentSelector


@dataclass(frozen=True)
class MutationTrial:
    """What a mutated child must beat to count as a mutation success.

    Attributes:
        mutated: Whether the child was mutated at all.
        reference_fitness: Fitness of the better of its two parents.
        reference_violation: Constraint violation of that parent.
    """

    mutated: bool
    reference_fitness: Fitness
    reference_violation: float


class Reproducer:
    """Creates child genomes from a population.

    Args:
        representation: Genome operators.
        config: Resolved engine settings (operators, rates, parameters).
        selector: Parent-selection strategy.
        comparator: Defines "better"; supplies selection scores.
        adaptation: Source of the current mutation intensity.
        rng: The engine's random generator.
    """

    def __init__(
        self,
        representation: Representation,
        config: GAConfig,
        selector: ParentSelector,
        comparator: FitnessComparator,
        adaptation: AdaptiveMutationController,
        rng: np.random.Generator,
    ) -> None:
        self.representation = representation
        self.config = config
        self.selector = selector
        self.comparator = comparator
        self.adaptation = adaptation
        self.rng = rng

    def random_genomes(self, count: int) -> List[np.ndarray]:
        """Draw ``count`` random genomes from the representation."""
        return [self.representation.random_individual(self.rng) for _ in range(count)]

    def select_parent_indices(
        self, population: Sequence[Individual], count: int, *, distinct: bool = False
    ) -> List[int]:
        """Select parent indices from ``population``.

        Multi-objective selection reads each individual's ``rank`` and
        ``crowding_distance``, which must be current (the engine refreshes them
        after every population change).
        """
        if self.config.multi_objective:
            ranks = np.asarray([ind.rank for ind in population], dtype=float)
            crowding = np.asarray([ind.crowding_distance for ind in population], dtype=float)
            return self.selector.select(
                self.rng, count, ranks=ranks, crowding=crowding, distinct=distinct
            )
        scores = np.asarray([self.comparator.effective(ind) for ind in population])
        return self.selector.select(self.rng, count, scores=scores, distinct=distinct)

    def _mutate(self, genes: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Possibly mutate one genome; returns it and whether mutation happened."""
        if self.rng.random() >= self.adaptation.mutation_prob:
            return genes, False
        kwargs: Dict[str, Any] = dict(self.config.mutation_params)
        kwargs["mutation_prob"] = self.adaptation.gene_mutation_prob
        if self.config.mutation_method == MutationMethod.GAUSSIAN:
            kwargs["sigma"] = self.representation.absolute_sigma(self.adaptation.sigma)
        mutated = self.representation.mutate(genes, self.config.mutation_method, self.rng, **kwargs)
        return self.representation.make_valid(mutated), True

    def make_children(
        self, parent1: Individual, parent2: Individual
    ) -> List[Tuple[np.ndarray, MutationTrial]]:
        """Recombine and mutate one pair of parents (two children, not evaluated)."""
        if self.rng.random() < self.config.crossover_prob:
            child1, child2 = self.representation.crossover(
                parent1.genes,
                parent2.genes,
                self.config.crossover_method,
                self.rng,
                **self.config.crossover_params,
            )
        else:
            child1, child2 = parent1.genes.copy(), parent2.genes.copy()

        reference = self.comparator.better_of(parent1, parent2)
        children = []
        for genes in (child1, child2):
            genes, mutated = self._mutate(self.representation.make_valid(genes))
            trial = MutationTrial(mutated, reference.fitness, reference.constraint_violation)
            children.append((genes, trial))
        return children

    def produce(
        self, population: Sequence[Individual], count: int
    ) -> Tuple[List[np.ndarray], List[MutationTrial]]:
        """Produce ``count`` child genomes from freshly selected parents."""
        parent_indices = self.select_parent_indices(population, count + count % 2)
        genes: List[np.ndarray] = []
        trials: List[MutationTrial] = []
        for position in range(0, len(parent_indices), 2):
            parent1 = population[parent_indices[position]]
            parent2 = population[parent_indices[position + 1]]
            for child_genes, trial in self.make_children(parent1, parent2):
                genes.append(child_genes)
                trials.append(trial)
        return genes[:count], trials[:count]

    def record_trial(self, child: Individual, trial: MutationTrial) -> None:
        """Report to the adaptation controller whether a mutated child beat its parents."""
        if not trial.mutated:
            return
        success = self.comparator.is_better(
            child.fitness,
            trial.reference_fitness,
            child.constraint_violation,
            trial.reference_violation,
        )
        self.adaptation.record_trial(success)
