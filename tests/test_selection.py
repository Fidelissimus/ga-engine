import unittest

import numpy as np

from ga_engine import ConfigurationError, SelectionMethod
from ga_engine.selection import ParentSelector


class ScalarSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.scores = np.arange(50, dtype=float)  # index 0 is best (minimisation)

    def test_every_method_exerts_selection_pressure(self):
        """The average selected score must clearly beat the population average."""
        for method in SelectionMethod:
            selector = ParentSelector(method, 3)
            means = [
                np.mean(self.scores[selector.select(self.rng, 50, scores=self.scores)])
                for _ in range(200)
            ]
            self.assertLess(np.mean(means), 0.8 * np.mean(self.scores), method)

    def test_weighted_methods_sample_with_replacement(self):
        for method in (SelectionMethod.ROULETTE, SelectionMethod.RANK, SelectionMethod.SUS):
            chosen = ParentSelector(method).select(self.rng, 50, scores=self.scores)
            self.assertLess(len(set(chosen)), 50, method)  # a permutation would have 50 uniques

    def test_sus_output_is_not_sorted(self):
        chosen = ParentSelector(SelectionMethod.SUS).select(self.rng, 50, scores=self.scores)
        self.assertNotEqual(chosen, sorted(chosen))

    def test_distinct_returns_unique_indices(self):
        for method in SelectionMethod:
            chosen = ParentSelector(method, 3).select(self.rng, 2, scores=self.scores, distinct=True)
            self.assertEqual(len(set(chosen)), 2, method)

    def test_distinct_more_than_population_is_an_error(self):
        with self.assertRaises(ConfigurationError):
            ParentSelector(SelectionMethod.TOURNAMENT).select(
                self.rng, 60, scores=self.scores, distinct=True
            )

    def test_flat_and_infinite_scores_do_not_break_weights(self):
        for scores in (np.ones(10), np.full(10, np.inf), np.array([1.0, np.inf, 2.0, np.inf])):
            for method in SelectionMethod:
                chosen = ParentSelector(method, 2).select(self.rng, 4, scores=scores)
                self.assertEqual(len(chosen), 4)
                self.assertTrue(all(0 <= i < len(scores) for i in chosen))

    def test_tournament_size_capped_at_population(self):
        chosen = ParentSelector(SelectionMethod.TOURNAMENT, 99).select(
            self.rng, 3, scores=np.array([3.0, 1.0, 2.0])
        )
        self.assertEqual(chosen, [1, 1, 1])  # the full-population tournament always picks the best

    def test_invalid_use(self):
        with self.assertRaises(ConfigurationError):
            ParentSelector(SelectionMethod.TOURNAMENT, 0)
        with self.assertRaises(ConfigurationError):
            ParentSelector(SelectionMethod.TOURNAMENT).select(self.rng, 2)
        self.assertEqual(ParentSelector(SelectionMethod.RANK).select(self.rng, 0, scores=self.scores), [])


class NsgaSelectionTests(unittest.TestCase):
    def test_prefers_low_rank_for_every_method(self):
        rng = np.random.default_rng(4)
        ranks = np.array([0.0] * 5 + [3.0] * 45)
        crowding = np.ones(50)
        for method in SelectionMethod:
            chosen = ParentSelector(method, 3).select(rng, 400, ranks=ranks, crowding=crowding)
            self.assertGreater(np.mean(np.asarray(chosen) < 5), 0.1, method)  # 10% of population, but over-selected

    def test_infinite_crowding_handled(self):
        rng = np.random.default_rng(5)
        ranks = np.zeros(6)
        crowding = np.array([np.inf, 1.0, 2.0, np.inf, 0.5, 0.1])
        for method in SelectionMethod:
            self.assertEqual(len(ParentSelector(method).select(rng, 6, ranks=ranks, crowding=crowding)), 6)


if __name__ == "__main__":
    unittest.main()
