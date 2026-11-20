"""Tune discrete settings with an integer genome and per-gene ranges.

The "model" below is a stand-in for something expensive (training a model,
running a simulation): its score depends on three integer settings.
"""

from __future__ import annotations

import numpy as np

from ga_engine import GeneticAlgorithm, IntegerRepresentation, MutationMethod

#: Setting name -> inclusive (low, high) range; one gene per setting.
SEARCH_SPACE = {
    "tree_depth": (2, 20),
    "n_estimators": (10, 500),
    "min_samples_leaf": (1, 30),
}
BEST_SETTINGS = np.array([9, 340, 4])


def validation_error(settings: np.ndarray) -> float:
    """A smooth fake validation error with its minimum at ``BEST_SETTINGS``."""
    span = np.array([high - low for low, high in SEARCH_SPACE.values()])
    return float(np.sum(((settings - BEST_SETTINGS) / span) ** 2))


def main() -> None:
    """Search the space and print the best settings found."""
    ga = GeneticAlgorithm(
        population_size=40,
        representation=IntegerRepresentation(len(SEARCH_SPACE), list(SEARCH_SPACE.values())),
        fitness_function=validation_error,
        mutation_method=MutationMethod.CREEP,       # ordinal settings: nudge, don't jump
        mutation_params={"max_step": 10},
        mutation_prob=0.5,
        rng_seed=4,
    )
    result = ga.run(max_generations=120, target_fitness=1e-6)

    print(f"stopped: {result.stop_reason} after {result.evaluations} evaluations")
    for name, value in zip(SEARCH_SPACE, result.best_individual.genes):
        print(f"  {name:<17}= {int(value)}")
    print(f"validation error: {result.best_fitness:.6f}")


if __name__ == "__main__":
    main()
