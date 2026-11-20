import unittest

import numpy as np

from ga_engine import (
    BinaryRepresentation,
    ConfigurationError,
    CrossoverMethod,
    MutationMethod,
    PermutationRepresentation,
    RealRepresentation,
)


class RealRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.rep = RealRepresentation(6, (-5.0, 5.0))

    def test_random_individual_is_valid(self):
        for _ in range(20):
            self.assertTrue(self.rep.validate(self.rep.random_individual(self.rng)))

    def test_exchange_crossovers_recombine_instead_of_blending(self):
        zeros, ones = np.zeros(6), np.ones(6)
        for method in (CrossoverMethod.SINGLE_POINT, CrossoverMethod.TWO_POINT):
            c1, c2 = self.rep.crossover(zeros, ones, method, self.rng)
            self.assertTrue(set(np.unique(c1)) <= {0.0, 1.0})  # genes are exchanged, never averaged
            np.testing.assert_array_equal(c1 + c2, ones)  # complementary children
            self.assertFalse(np.array_equal(c1, c2))

    def test_two_point_with_two_dimensions_returns_copies(self):
        rep = RealRepresentation(2, (-1, 1))
        a, b = np.array([0.1, 0.2]), np.array([0.3, 0.4])
        c1, c2 = rep.crossover(a, b, CrossoverMethod.TWO_POINT, self.rng)
        np.testing.assert_array_equal(c1, a)
        np.testing.assert_array_equal(c2, b)

    def test_all_operators_stay_within_bounds(self):
        a, b = self.rep.random_individual(self.rng), self.rep.random_individual(self.rng)
        for method in self.rep.SUPPORTED_CROSSOVERS:
            for child in self.rep.crossover(a, b, method, self.rng):
                self.assertTrue(self.rep.validate(child), method)
        for method in self.rep.SUPPORTED_MUTATIONS:
            mutated = self.rep.mutate(a, method, self.rng, mutation_prob=1.0)
            self.assertTrue(self.rep.validate(mutated), method)

    def test_mutation_does_not_modify_input(self):
        genes = self.rep.random_individual(self.rng)
        before = genes.copy()
        self.rep.mutate(genes, MutationMethod.GAUSSIAN, self.rng, mutation_prob=1.0)
        np.testing.assert_array_equal(genes, before)

    def test_repair_handles_nan_inf_and_out_of_range(self):
        repaired = self.rep.repair(np.array([np.nan, np.inf, -np.inf, 99.0, -99.0, 0.5]))
        self.assertTrue(self.rep.validate(repaired))
        self.assertEqual(repaired[0], 0.0)
        self.assertEqual(repaired[1], 5.0)
        self.assertEqual(repaired[2], -5.0)

    def test_parameter_validation(self):
        a, b = self.rep.random_individual(self.rng), self.rep.random_individual(self.rng)
        with self.assertRaises(ConfigurationError):
            self.rep.crossover(a, b, CrossoverMethod.SBX, self.rng, eta=0.0)
        with self.assertRaises(ConfigurationError):
            self.rep.mutate(a, MutationMethod.GAUSSIAN, self.rng, mutation_prob=1.5)

    def test_invalid_construction(self):
        for bounds in ((1.0, 1.0), (2.0, 1.0), (float("-inf"), 1.0)):
            with self.assertRaises(ConfigurationError):
                RealRepresentation(3, bounds)
        with self.assertRaises(ConfigurationError):
            RealRepresentation(0)

    def test_check_operator_config(self):
        self.rep.check_operator_config(CrossoverMethod.SBX, {"eta": 10}, MutationMethod.POLYNOMIAL, {"eta_m": 5})
        with self.assertRaises(ConfigurationError):
            self.rep.check_operator_config(CrossoverMethod.SBX, {"alpha": 1}, MutationMethod.GAUSSIAN, {})
        with self.assertRaises(ConfigurationError):
            self.rep.check_operator_config(CrossoverMethod.ORDER, {}, MutationMethod.GAUSSIAN, {})
        with self.assertRaises(ConfigurationError):
            self.rep.check_operator_config(CrossoverMethod.SBX, {"eta": -1}, MutationMethod.GAUSSIAN, {})

    def test_diversity_range(self):
        same = np.tile(self.rep.random_individual(self.rng), (5, 1))
        self.assertEqual(self.rep.diversity(same), 0.0)
        spread = np.stack([self.rep.random_individual(self.rng) for _ in range(30)])
        self.assertGreater(self.rep.diversity(spread), 0.0)


class BinaryRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.rep = BinaryRepresentation(12)

    def test_operators_preserve_validity(self):
        a, b = self.rep.random_individual(self.rng), self.rep.random_individual(self.rng)
        for method in self.rep.SUPPORTED_CROSSOVERS:
            for child in self.rep.crossover(a, b, method, self.rng):
                self.assertTrue(self.rep.validate(child), method)
        self.assertTrue(self.rep.validate(self.rep.mutate(a, MutationMethod.BIT_FLIP, self.rng, mutation_prob=0.5)))

    def test_bit_flip_probability_one_inverts(self):
        genes = self.rep.random_individual(self.rng)
        np.testing.assert_array_equal(
            self.rep.mutate(genes, MutationMethod.BIT_FLIP, self.rng, mutation_prob=1.0), 1 - genes
        )

    def test_repair_rounds_and_handles_nan(self):
        repaired = self.rep.repair(np.array([0.2, 0.7, np.nan, 5, -3, 1, 0, 0.5, 0.49, 0.51, 1, 0]))
        self.assertEqual(repaired.dtype, np.int8)
        self.assertTrue(self.rep.validate(repaired))
        self.assertEqual(repaired[1], 1)
        self.assertEqual(repaired[2], 0)

    def test_rejects_unsupported_operators(self):
        with self.assertRaises(ConfigurationError):
            self.rep.check_operator_config(CrossoverMethod.SBX, {}, MutationMethod.BIT_FLIP, {})
        with self.assertRaises(ConfigurationError):
            self.rep.check_operator_config(CrossoverMethod.UNIFORM, {}, MutationMethod.GAUSSIAN, {})


class PermutationRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.rep = PermutationRepresentation(10)

    def test_crossovers_always_yield_permutations(self):
        for method in (CrossoverMethod.ORDER, CrossoverMethod.PMX):
            for _ in range(300):
                a, b = self.rep.random_individual(self.rng), self.rep.random_individual(self.rng)
                for child in self.rep.crossover(a, b, method, self.rng):
                    self.assertTrue(self.rep.validate(child), method)

    def test_order_crossover_matches_textbook_example(self):
        p1 = np.arange(1, 10)
        p2 = np.array([9, 3, 7, 8, 2, 6, 5, 1, 4])
        child = PermutationRepresentation._ox_fill(p1, p2, 3, 6)
        np.testing.assert_array_equal(child, [3, 8, 2, 4, 5, 6, 7, 1, 9])

    def test_mutations_yield_permutations(self):
        genes = self.rep.random_individual(self.rng)
        for method in self.rep.SUPPORTED_MUTATIONS:
            for _ in range(100):
                self.assertTrue(self.rep.validate(self.rep.mutate(genes, method, self.rng)), method)

    def test_swap_changes_exactly_two_positions(self):
        genes = self.rep.random_individual(self.rng)
        mutated = self.rep.mutate(genes, MutationMethod.SWAP, self.rng)
        self.assertEqual(int(np.sum(mutated != genes)), 2)

    def test_repair(self):
        repaired = self.rep.repair(np.array([0, 0, 1, 99, 3, 3, 5, 6, 7, 8]))
        self.assertTrue(self.rep.validate(repaired))
        with self.assertRaises(ConfigurationError):
            self.rep.repair(np.arange(5))

    def test_validate_rejects_float_dtype_and_duplicates(self):
        self.assertFalse(self.rep.validate(np.arange(10, dtype=float)))
        self.assertFalse(self.rep.validate(np.zeros(10, dtype=int)))

    def test_rejects_unsupported_operators(self):
        with self.assertRaises(ConfigurationError):
            self.rep.check_operator_config(CrossoverMethod.ORDER, {}, MutationMethod.GAUSSIAN, {})
        with self.assertRaises(ConfigurationError):
            self.rep.check_operator_config(CrossoverMethod.UNIFORM, {}, MutationMethod.SWAP, {})


if __name__ == "__main__":
    unittest.main()
