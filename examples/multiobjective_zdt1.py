"""Approximate a Pareto front with NSGA-II on the ZDT1 benchmark."""

from __future__ import annotations

import math

from ga_engine import GeneticAlgorithm, RealRepresentation
from ga_engine.benchmarks import zdt1


def main() -> None:
    """Report how close the final front is to the analytic ZDT1 front."""
    ga = GeneticAlgorithm(
        population_size=100,
        representation=RealRepresentation(dimensions=10, bounds=(0.0, 1.0)),
        fitness_function=zdt1,
        multi_objective=True,
        rng_seed=5,
    )
    result = ga.run(max_generations=150)

    front = sorted(result.pareto_front or [], key=lambda ind: ind.fitness[0])
    gaps = [ind.fitness[1] - (1.0 - math.sqrt(ind.fitness[0])) for ind in front]
    print(f"front size          : {len(front)}")
    print(f"mean gap to true front: {sum(gaps) / len(gaps):.4f}")
    for individual in front[:: max(1, len(front) // 6)]:
        f1, f2 = individual.fitness
        print(f"  f1={f1:.3f}  f2={f2:.3f}")


if __name__ == "__main__":
    main()
