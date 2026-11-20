import unittest

import numpy as np

from ga_engine.adaptation import AdaptiveMutationController
from ga_engine.models import Individual
from ga_engine.replacement import (
    assemble_survivors,
    generational_replacement,
    truncation_replacement,
)


def ind(value, age=0, tag=0.0):
    return Individual(np.array([tag]), float(value), age=age)


def score(individual):
    return float(individual.fitness)


class ReplacementTests(unittest.TestCase):
    def test_truncation_keeps_best_and_updates_ages(self):
        parents = [ind(5, age=2, tag=1), ind(1, age=3, tag=2)]
        offspring = [ind(3, tag=3), ind(0.5, tag=4)]
        survivors = truncation_replacement(parents, offspring, 2, score)
        self.assertEqual([s.fitness for s in survivors], [0.5, 1.0])
        self.assertEqual([s.age for s in survivors], [0, 4])  # offspring reset, parent aged

    def test_truncation_prefers_parents_on_ties(self):
        parents = [ind(1, tag=1)]
        offspring = [ind(1, tag=2)]
        survivors = truncation_replacement(parents, offspring, 1, score)
        self.assertEqual(survivors[0].genes[0], 1)

    def test_inputs_are_not_mutated(self):
        parents = [ind(2, age=1)]
        offspring = [ind(1)]
        truncation_replacement(parents, offspring, 1, score)
        self.assertEqual(parents[0].age, 1)

    def test_generational_without_elitism_is_a_full_replacement(self):
        parents = [ind(0, tag=9), ind(0, tag=9)]
        offspring = [ind(5, tag=1), ind(6, tag=2)]
        nxt = generational_replacement(parents, offspring, 0, score)
        self.assertEqual(sorted(s.fitness for s in nxt), [5.0, 6.0])

    def test_generational_elite_displaces_worst_only_if_better(self):
        parents = [ind(1, age=4, tag=9), ind(100, tag=8)]
        offspring = [ind(5, tag=1), ind(6, tag=2)]
        nxt = generational_replacement(parents, offspring, 1, score)
        self.assertEqual(sorted(s.fitness for s in nxt), [1.0, 5.0])
        elite = next(s for s in nxt if s.fitness == 1.0)
        self.assertEqual(elite.age, 5)
        # An elite worse than every offspring must not be inserted.
        nxt = generational_replacement([ind(50)], [ind(5), ind(6)], 1, score)
        self.assertEqual(sorted(s.fitness for s in nxt), [5.0, 6.0])

    def test_assemble_survivors_ages_by_origin(self):
        parents = [ind(1, age=0, tag=1)]
        offspring = [ind(2, age=7, tag=2)]
        survivors = assemble_survivors(parents, offspring, [1, 0], 2)
        self.assertEqual([s.age for s in survivors], [0, 1])


class AdaptationTests(unittest.TestCase):
    def make(self, **overrides):
        settings = dict(
            enabled=True, mutation_prob=0.1, gene_mutation_prob=0.2, sigma=0.1,
            dimensions=5, adapt_sigma=True, window=10,
        )
        settings.update(overrides)
        return AdaptiveMutationController(**settings)

    def test_waits_for_window_and_accumulates_across_calls(self):
        controller = self.make()
        for _ in range(9):
            controller.record_trial(True)
        controller.update()
        self.assertEqual(controller.mutation_prob, 0.1)
        controller.record_trial(True)
        controller.update()
        self.assertGreater(controller.mutation_prob, 0.1)
        self.assertEqual(controller.trial_count, 0)

    def test_low_success_shrinks_and_high_success_grows(self):
        low = self.make()
        for _ in range(10):
            low.record_trial(False)
        low.update()
        self.assertLess(low.sigma, 0.1)
        self.assertEqual(low.last_success_ratio, 0.0)
        high = self.make()
        for _ in range(10):
            high.record_trial(True)
        high.update()
        self.assertGreater(high.sigma, 0.1)

    def test_exact_one_fifth_leaves_values_unchanged(self):
        controller = self.make()
        for outcome in [True] * 2 + [False] * 8:
            controller.record_trial(outcome)
        controller.update()
        self.assertEqual(controller.mutation_prob, 0.1)

    def test_bounds_never_exclude_starting_value(self):
        controller = self.make(mutation_prob=0.0, window=1)
        controller.record_trial(False)
        controller.update()
        self.assertEqual(controller.mutation_prob, 0.0)  # a zero probability is not silently raised
        high = self.make(mutation_prob=1.0, window=1)
        high.record_trial(True)
        high.update()
        self.assertEqual(high.mutation_prob, 1.0)

    def test_disabled_controller_is_inert(self):
        controller = self.make(enabled=False, window=1)
        controller.record_trial(True)
        controller.update()
        self.assertEqual((controller.mutation_prob, controller.trial_count), (0.1, 0))

    def test_sigma_untouched_when_not_adapted(self):
        controller = self.make(adapt_sigma=False, window=1)
        controller.record_trial(True)
        controller.update()
        self.assertEqual(controller.sigma, 0.1)

    def test_state_roundtrip_and_reset(self):
        controller = self.make(window=1)
        controller.record_trial(True)
        controller.update()
        state = controller.state_dict()
        other = self.make()
        other.load_state_dict(state)
        self.assertEqual(other.state_dict(), state)
        other.reset()
        self.assertEqual((other.mutation_prob, other.sigma, other.last_success_ratio), (0.1, 0.1, None))


if __name__ == "__main__":
    unittest.main()
