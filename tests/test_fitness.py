import math
import unittest

import numpy as np

from ga_engine.fitness import (
    FitnessComparator,
    dominates,
    normalize_fitness,
    normalize_violation,
    penalized_fitness,
)
from ga_engine.models import Individual


class NormalizeFitnessTests(unittest.TestCase):
    def test_scalar_forms(self):
        self.assertEqual(normalize_fitness(3, False), 3.0)
        self.assertEqual(normalize_fitness(np.float32(2.5), False), 2.5)
        self.assertEqual(normalize_fitness(np.array(4.0), False), 4.0)
        self.assertEqual(normalize_fitness([7.0], False), 7.0)

    def test_nan_becomes_inf_and_negative_inf_is_rejected(self):
        self.assertEqual(normalize_fitness(float("nan"), False), math.inf)
        with self.assertRaises(ValueError):
            normalize_fitness(-math.inf, False)

    def test_multi_objective_shapes(self):
        self.assertEqual(normalize_fitness([1, 2], True), (1.0, 2.0))
        self.assertEqual(normalize_fitness(np.array([1.0, np.nan]), True), (1.0, math.inf))
        with self.assertRaises(ValueError):
            normalize_fitness(1.0, True)
        with self.assertRaises(ValueError):
            normalize_fitness([1.0], True)
        with self.assertRaises(ValueError):
            normalize_fitness([1.0, 2.0], False)

    def test_non_numeric_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            normalize_fitness("abc", False)


class ViolationTests(unittest.TestCase):
    def test_scalar_and_sequence(self):
        self.assertEqual(normalize_violation(-5), 0.0)
        self.assertEqual(normalize_violation(2.5), 2.5)
        self.assertEqual(normalize_violation([1.0, -3.0, 2.0]), 3.0)

    def test_nan_means_maximally_infeasible_even_after_zero(self):
        self.assertEqual(normalize_violation(float("nan")), math.inf)
        self.assertEqual(normalize_violation([0.0, float("nan")]), math.inf)


class PenaltyTests(unittest.TestCase):
    def test_penalized_fitness(self):
        self.assertEqual(penalized_fitness(1.0, 2.0, 3.0), 7.0)
        self.assertEqual(penalized_fitness(1.0, 0.0, 3.0), 1.0)

    def test_zero_penalty_never_yields_nan(self):
        self.assertEqual(penalized_fitness(2.0, math.inf, 0.0), 2.0)
        self.assertEqual(penalized_fitness(math.inf, math.inf, 1.0), math.inf)


class ComparatorTests(unittest.TestCase):
    def test_dominates(self):
        self.assertTrue(dominates((1, 1), (2, 2)))
        self.assertTrue(dominates((1, 2), (1, 3)))
        self.assertFalse(dominates((1, 3), (2, 2)))
        self.assertFalse(dominates((1, 1), (1, 1)))

    def test_single_objective_uses_penalty(self):
        cmp = FitnessComparator(False, 10.0)
        self.assertTrue(cmp.is_better(5.0, 1.0, 0.0, 1.0))  # 5 < 1 + 10
        self.assertFalse(cmp.is_better(1.0, 5.0, 1.0, 0.0))
        individual = Individual(np.zeros(1), 2.0, constraint_violation=0.5)
        self.assertEqual(cmp.effective(individual), 7.0)

    def test_constrained_dominance(self):
        cmp = FitnessComparator(True, 1.0)
        self.assertTrue(cmp.is_better((9, 9), (0, 0), 0.0, 1.0))  # feasible beats infeasible
        self.assertFalse(cmp.is_better((0, 0), (9, 9), 1.0, 0.0))
        self.assertTrue(cmp.is_better((9, 9), (0, 0), 1.0, 2.0))  # smaller violation wins
        self.assertTrue(cmp.is_better((1, 1), (2, 2)))

    def test_better_of_returns_second_on_tie(self):
        cmp = FitnessComparator(False, 1.0)
        a, b = Individual(np.zeros(1), 1.0), Individual(np.ones(1), 1.0)
        self.assertIs(cmp.better_of(a, b), b)
        winner = Individual(np.zeros(1), 0.5)
        self.assertIs(cmp.better_of(winner, b), winner)


if __name__ == "__main__":
    unittest.main()
