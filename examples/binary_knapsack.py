"""Solve a constrained 0/1 knapsack with a binary genome and a penalty."""

from __future__ import annotations

import numpy as np

from ga_engine import BinaryRepresentation, GeneticAlgorithm
from ga_engine.benchmarks import KnapsackProblem


def main() -> None:
    """Pack as much value as possible without exceeding the capacity."""
    problem = KnapsackProblem.random(n_items=40, seed=7, capacity_ratio=0.4)
    ga = GeneticAlgorithm(
        population_size=80,
        representation=BinaryRepresentation(dimensions=40),
        fitness_function=problem.objective,
        constraint_function=problem.constraint,
        constraint_penalty=1000.0,
        rng_seed=7,
    )
    result = ga.run(max_generations=200, stagnation_generations=60)

    chosen = result.best_individual.genes.astype(bool)
    print(f"stop reason : {result.stop_reason} at generation {result.generation}")
    print(f"items packed: {int(chosen.sum())} of {len(chosen)}")
    print(f"total value : {-result.best_fitness:.0f}")
    print(f"weight      : {np.dot(problem.weights, chosen):.0f} / capacity {problem.capacity:.0f}")
    print(f"feasible    : {result.best_individual.is_feasible}")


if __name__ == "__main__":
    main()
