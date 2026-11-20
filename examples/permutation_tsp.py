"""Find a short tour through random cities with a permutation genome."""

from __future__ import annotations

from ga_engine import (
    CrossoverMethod,
    GeneticAlgorithm,
    MutationMethod,
    PermutationRepresentation,
)
from ga_engine.benchmarks import TspProblem


def main() -> None:
    """Compare the first and the final tour length."""
    problem = TspProblem.random(n_cities=25, seed=3)
    ga = GeneticAlgorithm(
        population_size=120,
        representation=PermutationRepresentation(dimensions=25),
        fitness_function=problem,
        crossover_method=CrossoverMethod.ORDER,
        mutation_method=MutationMethod.INVERSION,
        mutation_prob=0.3,
        rng_seed=3,
    )
    result = ga.run(max_generations=300)

    first = result.stats["best_fitness"][0]
    print(f"initial best tour: {first:.3f}")
    print(f"final best tour  : {result.best_fitness:.3f}")
    print(f"tour             : {result.best_individual.genes.tolist()}")


if __name__ == "__main__":
    main()
