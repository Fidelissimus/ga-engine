import unittest

import numpy as np

from ga_engine import (
    BinaryRepresentation,
    ConfigurationError,
    GeneticAlgorithm,
    IslandModel,
    RealRepresentation,
)
from ga_engine.benchmarks import onemax, rastrigin, sphere, zdt1


def island(seed, dims=4, population=24, **kwargs):
    return GeneticAlgorithm(population, RealRepresentation(dims, (-5.12, 5.12)), rastrigin, rng_seed=seed, **kwargs)


class IslandValidationTests(unittest.TestCase):
    def test_requires_two_compatible_islands(self):
        with self.assertRaises(ConfigurationError):
            IslandModel([island(1)])
        with self.assertRaises(ConfigurationError):
            IslandModel([island(1), island(2, dims=5)])
        with self.assertRaises(ConfigurationError):
            IslandModel([island(1), GeneticAlgorithm(10, BinaryRepresentation(4), onemax)])
        with self.assertRaises(ConfigurationError):
            IslandModel([island(1), GeneticAlgorithm(24, RealRepresentation(4, (0, 1)), zdt1, multi_objective=True)])
        with self.assertRaises(ConfigurationError):
            IslandModel([island(1), "not an engine"])

    def test_migration_settings(self):
        with self.assertRaises(ConfigurationError):
            IslandModel([island(1), island(2)], migration_interval=0)
        with self.assertRaises(ConfigurationError):
            IslandModel([island(1), island(2)], migration_size=0)
        with self.assertRaises(ConfigurationError):
            IslandModel([island(1), island(2)], migration_size=25)


class IslandRunTests(unittest.TestCase):
    def test_run_returns_global_best_and_per_island_stats(self):
        model = IslandModel([island(1), island(2), island(3)], migration_interval=5, migration_size=2)
        result = model.run(max_generations=30)
        self.assertEqual(result.generation, 30)
        self.assertEqual(len(result.island_stats), 3)
        island_bests = [i.best_individual.fitness for i in model.islands]
        self.assertEqual(result.best_fitness, min(island_bests))
        self.assertEqual(result.evaluations, sum(i.evaluations for i in model.islands))

    def test_run_is_reproducible(self):
        def build():
            return IslandModel([island(1), island(2)], migration_interval=4)

        self.assertEqual(build().run(max_generations=20).best_fitness, build().run(max_generations=20).best_fitness)

    def test_migration_moves_the_best_individual_to_the_next_island(self):
        model = IslandModel([island(1), island(2)], migration_interval=1000, migration_size=1)
        for member in model.islands:
            member.initialize_population()
        champion = model.islands[0].select_migrants(1)[0]
        model._migrate()
        fitnesses = [i.fitness for i in model.islands[1].population]
        self.assertIn(champion.fitness, fitnesses)
        self.assertEqual(len(model.islands[1].population), 24)

    def test_target_and_stagnation_stop(self):
        model = IslandModel(
            [GeneticAlgorithm(30, RealRepresentation(3, (-1, 1)), sphere, rng_seed=s) for s in (1, 2)],
            migration_interval=5,
        )
        result = model.run(max_generations=500, target_fitness=1e-3)
        self.assertEqual(result.stop_reason, "target_fitness")
        result = IslandModel([island(1), island(2)]).run(max_generations=500, stagnation_generations=15)
        self.assertEqual(result.stop_reason, "stagnation")

    def test_resume_requires_initialised_synchronised_islands(self):
        model = IslandModel([island(1), island(2)])
        with self.assertRaises(RuntimeError):
            model.run(resume=True)
        model.run(max_generations=5)
        result = model.run(max_generations=10, resume=True)
        self.assertEqual(result.generation, 10)
        model.islands[0].step()
        with self.assertRaises(RuntimeError):
            model.run(max_generations=12, resume=True)

    def test_multi_objective_returns_merged_non_dominated_front(self):
        def build(seed):
            return GeneticAlgorithm(30, RealRepresentation(5, (0, 1)), zdt1, multi_objective=True, rng_seed=seed)

        model = IslandModel([build(1), build(2)], migration_interval=5)
        result = model.run(max_generations=25)
        front = [f.fitness for f in result.pareto_front]
        for a in front:
            for b in front:
                self.assertFalse(all(x <= y for x, y in zip(a, b)) and a != b)
        union = [i.fitness for m in model.islands for i in m.get_pareto_front()]
        for point in union:  # nothing from any island may dominate a member of the merged front
            for member in front:
                self.assertFalse(all(x <= y for x, y in zip(point, member)) and point != member)
        self.assertIn(result.best_fitness, front)

    def test_callback_is_called_each_generation_and_can_stop_the_run(self):
        model = IslandModel([island(1), island(2)], migration_interval=3)
        seen = []
        result = model.run(max_generations=50, on_generation=lambda m: seen.append(m.generation) or m.generation >= 4)
        self.assertEqual(seen, [0, 1, 2, 3, 4])
        self.assertEqual((result.stop_reason, result.generation, result.converged), ("callback", 4, False))

    def test_generation_property_tracks_the_islands(self):
        model = IslandModel([island(1), island(2)])
        self.assertEqual(model.generation, 0)
        model.run(max_generations=3)
        self.assertEqual(model.generation, 3)

    def test_invalid_run_arguments(self):
        model = IslandModel([island(1), island(2)])
        with self.assertRaises(ConfigurationError):
            model.run(max_generations=-1)
        with self.assertRaises(ConfigurationError):
            model.run(stagnation_generations=0)

    def test_parallel_islands_release_their_pools(self):
        model = IslandModel(
            [
                GeneticAlgorithm(12, RealRepresentation(3, (-1, 1)), sphere, rng_seed=s, n_workers=2)
                for s in (1, 2)
            ],
            migration_interval=2,
            migration_size=1,
        )
        model.run(max_generations=4)
        self.assertTrue(all(not i.evaluator.pool_active for i in model.islands))
        with model:
            self.assertTrue(all(i.evaluator.pool_active for i in model.islands))
        self.assertTrue(all(not i.evaluator.pool_active for i in model.islands))


if __name__ == "__main__":
    unittest.main()
