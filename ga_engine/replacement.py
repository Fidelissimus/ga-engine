"""Survivor selection: building the next population from parents and offspring.

All functions are pure with respect to their inputs: they return *new*
:class:`~ga_engine.models.Individual` objects and never mutate the arguments.
Ages are maintained here: a survivor that came from the parent generation ages
by one, a survivor that came from the offspring starts at age zero.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

from .errors import ConfigurationError
from .models import Individual
from .multiobjective import nsga2_order

ScoreFunction = Callable[[Individual], float]


def assemble_survivors(
    parents: Sequence[Individual],
    offspring: Sequence[Individual],
    order: Sequence[int],
    size: int,
) -> List[Individual]:
    """Take the first ``size`` entries of a ranking over ``parents + offspring``.

    Args:
        parents: Current population.
        offspring: Newly evaluated individuals.
        order: Indices into ``list(parents) + list(offspring)``, best first.
        size: Number of survivors to keep.

    Returns:
        Cloned survivors with ages updated (parents +1, offspring 0).
    """
    boundary = len(parents)
    survivors: List[Individual] = []
    for index in order[:size]:
        if index < boundary:
            clone = parents[index].clone()
            clone.age += 1
        else:
            clone = offspring[index - boundary].clone()
            clone.age = 0
        survivors.append(clone)
    return survivors


def truncation_replacement(
    parents: Sequence[Individual],
    offspring: Sequence[Individual],
    size: int,
    score: ScoreFunction,
) -> List[Individual]:
    """(mu + lambda) survival: keep the best ``size`` of parents and offspring.

    Ties favour parents (they come first in the stable sort). With a small
    number of offspring this is classic steady-state replacement.
    """
    combined = list(parents) + list(offspring)
    order = sorted(range(len(combined)), key=lambda i: score(combined[i]))
    return assemble_survivors(parents, offspring, order, size)


def generational_replacement(
    parents: Sequence[Individual],
    offspring: Sequence[Individual],
    elitism_count: int,
    score: ScoreFunction,
) -> List[Individual]:
    """Replace the whole population by the offspring, preserving elites.

    The ``elitism_count`` best parents each displace the currently worst member
    of the new population, but only if they are strictly better than it.

    Returns:
        The next population, the same size as ``offspring``.
    """
    next_population = []
    for child in offspring:
        clone = child.clone()
        clone.age = 0
        next_population.append(clone)
    if elitism_count <= 0:
        return next_population

    scores = [score(ind) for ind in next_population]
    for elite in sorted(parents, key=score)[:elitism_count]:
        worst = max(range(len(scores)), key=scores.__getitem__)
        elite_score = score(elite)
        if elite_score < scores[worst]:
            kept = elite.clone()
            kept.age += 1
            next_population[worst] = kept
            scores[worst] = elite_score
    return next_population


class SurvivalPolicy:
    """Chooses the next population from the current one and the new offspring.

    * multi-objective: NSGA-II (mu + lambda) truncation by rank and crowding;
    * single-objective, ``generational``: offspring replace the population,
      elites displace the worst offspring when better;
    * single-objective, steady-state: (mu + lambda) truncation by ``score``.

    Args:
        size: Population size to maintain.
        multi_objective: Use NSGA-II survival.
        generational: Use generational replacement (single-objective).
        elitism_count: Elites preserved by generational replacement.
        score: Lower-is-better scalar score (single-objective).
    """

    def __init__(
        self,
        *,
        size: int,
        multi_objective: bool,
        generational: bool,
        elitism_count: int,
        score: ScoreFunction,
    ) -> None:
        self.size = size
        self.multi_objective = multi_objective
        self.generational = generational
        self.elitism_count = elitism_count
        self.score = score

    def select(
        self, population: Sequence[Individual], offspring: Sequence[Individual]
    ) -> List[Individual]:
        """Return the next population as new individuals.

        Raises:
            ConfigurationError: If there are no offspring, or the offspring
                count does not equal the population size under generational
                replacement.
        """
        if not offspring:
            raise ConfigurationError("replacement needs at least one offspring")
        if self.multi_objective:
            combined = list(population) + [child.clone() for child in offspring]
            order = nsga2_order(combined)
            return assemble_survivors(population, combined[len(population) :], order, self.size)
        if self.generational:
            if len(offspring) != self.size:
                raise ConfigurationError(
                    f"Generational replacement requires exactly {self.size} offspring, "
                    f"got {len(offspring)}"
                )
            return generational_replacement(population, offspring, self.elitism_count, self.score)
        return truncation_replacement(population, offspring, self.size, self.score)
