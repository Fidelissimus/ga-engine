import tempfile
import unittest
from pathlib import Path

import numpy as np

from ga_engine import (
    CheckpointError,
    ConfigurationError,
    CrossoverMethod,
    GeneticAlgorithm,
    IntegerRepresentation,
    MutationMethod,
)

PAIRS = [(1, 10), (0, 3), (50, 500), (-5, 5)]


class ConstructionTests(unittest.TestCase):
    def test_shared_and_per_gene_bounds(self):
        shared = IntegerRepresentation(3, (1, 10))
        np.testing.assert_array_equal(shared.lower_bounds, [1, 1, 1])
        self.assertEqual(shared.describe()["bounds"], [1, 10])
        per_gene = IntegerRepresentation(4, PAIRS)
        np.testing.assert_array_equal(per_gene.upper_bounds, [10, 3, 500, 5])
        self.assertEqual(per_gene.describe()["bounds"][2], [50, 500])
        self.assertEqual(per_gene.lower_bounds.dtype, np.int64)

    def test_whole_number_floats_are_accepted(self):
        rep = IntegerRepresentation(2, (1.0, 5.0))
        self.assertEqual(rep.describe()["bounds"], [1, 5])

    def test_invalid_bounds(self):
        for dimensions, bounds in ((2, (0.5, 3)), (2, (3, 3)), (2, (5, 1)), (3, [(0, 1)]), (2, [(0, 1), (0, 2.5)])):
            with self.subTest(bounds=bounds), self.assertRaises(ConfigurationError):
                IntegerRepresentation(dimensions, bounds)


class GenomeTests(unittest.TestCase):
    def setUp(self):
        self.rep = IntegerRepresentation(4, PAIRS)
        self.rng = np.random.default_rng(0)

    def test_random_individuals_are_valid_integers_within_each_range(self):
        for _ in range(200):
            genes = self.rep.random_individual(self.rng)
            self.assertEqual(genes.dtype, np.int64)
            self.assertTrue(self.rep.validate(genes))

    def test_every_gene_value_in_range_is_reachable(self):
        rep = IntegerRepresentation(1, (0, 3))
        seen = {int(rep.random_individual(self.rng)[0]) for _ in range(200)}
        self.assertEqual(seen, {0, 1, 2, 3})  # upper bound is inclusive

    def test_validate_rejects_floats_and_out_of_range(self):
        self.assertFalse(self.rep.validate(np.array([1.0, 0.0, 50.0, 0.0])))
        self.assertFalse(self.rep.validate(np.array([11, 0, 50, 0])))
        self.assertFalse(self.rep.validate(np.array([1, 0, 50])))

    def test_repair_rounds_clips_and_handles_nan(self):
        repaired = self.rep.repair(np.array([3.6, 99.0, np.nan, -17.2]))
        np.testing.assert_array_equal(repaired, [4, 3, 50, -5])
        self.assertTrue(self.rep.validate(repaired))

    def test_operators_preserve_validity(self):
        a, b = self.rep.random_individual(self.rng), self.rep.random_individual(self.rng)
        for method in self.rep.SUPPORTED_CROSSOVERS:
            for child in self.rep.crossover(a, b, method, self.rng):
                self.assertTrue(self.rep.validate(child), method)
        for method in self.rep.SUPPORTED_MUTATIONS:
            for _ in range(100):
                self.assertTrue(self.rep.validate(self.rep.mutate(a, method, self.rng, mutation_prob=1.0)), method)

    def test_random_reset_changes_genes_and_never_mutates_the_input(self):
        genes = self.rep.random_individual(self.rng)
        before = genes.copy()
        changed = 0
        for _ in range(100):
            changed += int(np.any(self.rep.mutate(genes, MutationMethod.RANDOM_RESET, self.rng, mutation_prob=1.0) != genes))
        np.testing.assert_array_equal(genes, before)
        self.assertGreater(changed, 90)

    def test_creep_moves_by_at_most_max_step_and_stays_in_range(self):
        rep = IntegerRepresentation(5, (0, 100))
        start = np.full(5, 50, dtype=np.int64)
        for max_step in (1, 3):
            for _ in range(100):
                moved = rep.mutate(start, MutationMethod.CREEP, self.rng, mutation_prob=1.0, max_step=max_step)
                steps = np.abs(moved - start)
                self.assertTrue(np.all((steps >= 1) & (steps <= max_step)))
        edge = rep.mutate(np.zeros(5, dtype=np.int64), MutationMethod.CREEP, self.rng, mutation_prob=1.0, max_step=2)
        self.assertTrue(rep.validate(edge))

    def test_uniform_crossover_probability_extremes(self):
        a, b = np.array([1, 0, 50, -5]), np.array([10, 3, 500, 5])
        c1, c2 = self.rep.crossover(a, b, CrossoverMethod.UNIFORM, self.rng, probability=1.0)
        np.testing.assert_array_equal(c1, a)
        np.testing.assert_array_equal(c2, b)


class EngineIntegrationTests(unittest.TestCase):
    def test_defaults_resolve_to_uniform_crossover_and_random_reset(self):
        ga = GeneticAlgorithm(10, IntegerRepresentation(3, (0, 5)), lambda x: float(np.sum(x)))
        self.assertEqual(ga.config.crossover_method, CrossoverMethod.UNIFORM)
        self.assertEqual(ga.config.mutation_method, MutationMethod.RANDOM_RESET)

    def test_unsupported_choices_and_parameters_fail_at_construction(self):
        rep = IntegerRepresentation(3, (0, 5))
        for kwargs in (
            dict(crossover_method=CrossoverMethod.SBX),
            dict(mutation_method=MutationMethod.GAUSSIAN),
            dict(mutation_method=MutationMethod.CREEP, mutation_params={"max_step": 0}),
            dict(mutation_method=MutationMethod.CREEP, mutation_params={"step": 2}),
        ):
            with self.subTest(kwargs), self.assertRaises(ConfigurationError):
                GeneticAlgorithm(10, rep, lambda x: 0.0, **kwargs)

    def test_finds_the_optimum_of_a_discrete_problem(self):
        target = np.array([7, 2, 321, -3])
        objective = lambda x: float(np.sum(np.abs(x - target) / (np.array([9, 3, 450, 10]))))
        for mutation, params in ((MutationMethod.RANDOM_RESET, None), (MutationMethod.CREEP, {"max_step": 5})):
            with self.subTest(mutation):
                ga = GeneticAlgorithm(
                    40, IntegerRepresentation(4, PAIRS), objective, mutation_method=mutation,
                    mutation_params=params, mutation_prob=0.5, rng_seed=1,
                )
                result = ga.run(max_generations=150, target_fitness=0.0)
                self.assertLess(result.best_fitness, 0.05)
                self.assertTrue(ga.representation.validate(result.best_individual.genes))

    def test_checkpoint_roundtrip_keeps_dtype_and_detects_bounds_mismatch(self):
        def build(bounds, seed=5):
            return GeneticAlgorithm(16, IntegerRepresentation(4, bounds), lambda x: float(np.sum(x)), rng_seed=seed)

        ga = build(PAIRS)
        ga.run(max_generations=3)
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".json", ".npz"):
                path = Path(directory) / f"ints{suffix}"
                ga.save_checkpoint(path)
                other = build(PAIRS, seed=9)
                other.load_checkpoint(path)
                self.assertEqual(other.population[0].genes.dtype, np.int64)
                np.testing.assert_array_equal(other.population[0].genes, ga.population[0].genes)
                with self.assertRaises(CheckpointError):
                    build((0, 10)).load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
