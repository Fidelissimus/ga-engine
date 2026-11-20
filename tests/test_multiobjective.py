import math
import unittest

import numpy as np

from ga_engine.fitness import FitnessComparator
from ga_engine.models import Individual
from ga_engine.multiobjective import (
    assign_crowding_distance,
    assign_rank_and_crowding,
    dominance_matrix,
    non_dominated_fronts,
    nsga2_order,
)


def make(fitness, violation=0.0):
    return Individual(np.zeros(1), tuple(map(float, fitness)), constraint_violation=violation)


class DominanceMatrixTests(unittest.TestCase):
    def test_matches_scalar_comparator_on_random_data(self):
        """Cross-check the vectorised matrix against the pairwise comparator."""
        rng = np.random.default_rng(0)
        fitness = np.round(rng.random((40, 3)), 1)  # rounding creates ties
        violation = np.where(rng.random(40) < 0.4, np.round(rng.random(40) * 3, 1), 0.0)
        matrix = dominance_matrix(fitness, violation)
        comparator = FitnessComparator(True, 1.0)
        for i in range(40):
            for j in range(40):
                expected = comparator.is_better(
                    tuple(fitness[i]), tuple(fitness[j]), float(violation[i]), float(violation[j])
                )
                self.assertEqual(bool(matrix[i, j]), expected, (i, j))

    def test_irreflexive(self):
        fitness = np.array([[1.0, 2.0], [1.0, 2.0]])
        self.assertFalse(dominance_matrix(fitness, np.zeros(2)).any())


class FrontTests(unittest.TestCase):
    def test_known_fronts(self):
        pop = [make(f) for f in [(1, 5), (2, 3), (4, 1), (3, 4), (5, 2), (6, 6)]]
        fronts = non_dominated_fronts(pop)
        self.assertEqual([sorted(f) for f in fronts], [[0, 1, 2], [3, 4], [5]])

    def test_every_index_appears_exactly_once(self):
        rng = np.random.default_rng(1)
        pop = [make(rng.random(2)) for _ in range(60)]
        flat = sorted(i for f in non_dominated_fronts(pop) for i in f)
        self.assertEqual(flat, list(range(60)))

    def test_infeasible_ranked_behind_feasible(self):
        pop = [make((9, 9), 0.0), make((0, 0), 1.0), make((0, 0), 2.0)]
        self.assertEqual(non_dominated_fronts(pop), [[0], [1], [2]])

    def test_empty(self):
        self.assertEqual(non_dominated_fronts([]), [])


class CrowdingTests(unittest.TestCase):
    def test_boundaries_are_infinite_and_interior_finite(self):
        front = [make(f) for f in [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]]
        assign_crowding_distance(front)
        self.assertTrue(math.isinf(front[0].crowding_distance))
        self.assertTrue(math.isinf(front[-1].crowding_distance))
        for individual in front[1:-1]:
            self.assertTrue(math.isfinite(individual.crowding_distance))
            # per objective: neighbour gap 2 / span 4 = 0.5; two objectives -> 1.0
            self.assertAlmostEqual(individual.crowding_distance, 1.0)

    def test_small_fronts_are_all_boundary(self):
        front = [make((0, 1)), make((1, 0))]
        assign_crowding_distance(front)
        self.assertTrue(all(math.isinf(i.crowding_distance) for i in front))

    def test_degenerate_span_is_ignored(self):
        front = [make((1, i)) for i in range(4)]
        assign_crowding_distance(front)
        self.assertTrue(all(i.crowding_distance >= 0 for i in front))


class OrderTests(unittest.TestCase):
    def test_order_sets_rank_and_prefers_front_then_crowding(self):
        pop = [make(f) for f in [(0, 4), (1, 3), (1.05, 2.95), (4, 0), (5, 5)]]
        order = nsga2_order(pop)
        self.assertEqual(order[-1], 4)  # dominated point is last
        self.assertEqual(pop[4].rank, 1)
        self.assertEqual({pop[i].rank for i in order[:4]}, {0})
        self.assertEqual(set(order[:2]), {0, 3})  # boundary points have infinite crowding

    def test_assign_rank_and_crowding_returns_fronts(self):
        pop = [make(f) for f in [(1, 2), (2, 1), (3, 3)]]
        fronts = assign_rank_and_crowding(pop)
        self.assertEqual(len(fronts), 2)
        self.assertEqual([i.rank for i in pop], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
