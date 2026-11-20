"""Unit tests for the focused components the engine is assembled from."""

import math
import unittest

import numpy as np

from ga_engine import (
    ConfigurationError,
    GeneticAlgorithm,
    Individual,
    MutationMethod,
    RealRepresentation,
    StopReason,
)
from ga_engine.benchmarks import sphere
from ga_engine.fitness import FitnessComparator
from ga_engine.migration import merge_immigrants, select_emigrants
from ga_engine.progress import ProgressTracker
from ga_engine.replacement import SurvivalPolicy
from ga_engine.reproduction import MutationTrial
from ga_engine.termination import StopCriteria


def ind(value, tag=0.0, age=0, violation=0.0):
    return Individual(np.array([tag]), float(value), age=age, constraint_violation=violation)


def mo(f1, f2):
    return Individual(np.zeros(1), (float(f1), float(f2)))


class StopCriteriaTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({"max_generations": -1}, {"max_generations": True}, {"stagnation_generations": 0}):
            with self.subTest(kwargs), self.assertRaises(ConfigurationError):
                StopCriteria(**kwargs)

    def test_target_in_multi_objective_mode_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            StopCriteria(target_fitness=1.0).require_single_objective(True)
        StopCriteria().require_single_objective(True)  # no target: fine

    def test_check_order_is_target_then_stagnation_then_budget(self):
        criteria = StopCriteria(10, 3, 0.5)
        everything = dict(generation=10, stagnation_count=3, target_reached=True)
        self.assertEqual(criteria.check(**everything), StopReason.TARGET_FITNESS)
        self.assertEqual(criteria.check(**{**everything, "target_reached": False}), StopReason.STAGNATION)
        self.assertEqual(
            criteria.check(**{**everything, "target_reached": False, "stagnation_count": 0}),
            StopReason.MAX_GENERATIONS,
        )
        self.assertIsNone(criteria.check(generation=1, stagnation_count=0, target_reached=False))

    def test_target_hit_is_ignored_when_no_target_is_configured(self):
        self.assertIsNone(StopCriteria(10).check(generation=0, stagnation_count=0, target_reached=True))

    def test_reasons_are_plain_strings(self):
        self.assertEqual(StopReason.STAGNATION, "stagnation")
        self.assertEqual(StopReason.CALLBACK.value, "callback")


class ProgressTrackerTests(unittest.TestCase):
    def scalar(self, tolerance=0.0):
        return ProgressTracker(FitnessComparator(False, 1.0), False, tolerance)

    def test_improvement_resets_and_no_improvement_counts(self):
        tracker = self.scalar()
        tracker.update([ind(5)], None, count_stagnation=False)
        tracker.update([ind(5)], None, count_stagnation=True)
        tracker.update([ind(5)], None, count_stagnation=True)
        self.assertEqual(tracker.stagnation_count, 2)
        tracker.update([ind(4)], None, count_stagnation=True)
        self.assertEqual((tracker.stagnation_count, tracker.best_individual.fitness), (0, 4.0))

    def test_tolerance_treats_tiny_gains_as_stagnation_but_still_records_them(self):
        tracker = self.scalar(tolerance=0.1)
        tracker.update([ind(5)], None, count_stagnation=False)
        tracker.update([ind(4.99)], None, count_stagnation=True)
        self.assertEqual(tracker.stagnation_count, 1)
        self.assertEqual(tracker.best_individual.fitness, 4.99)

    def test_counter_untouched_when_not_counting(self):
        tracker = self.scalar()
        tracker.update([ind(5)], None, count_stagnation=False)
        tracker.update([ind(6)], None, count_stagnation=False)
        self.assertEqual(tracker.stagnation_count, 0)

    def test_pareto_mode_resets_only_when_the_front_changes(self):
        tracker = ProgressTracker(FitnessComparator(True, 1.0), True)
        population = [mo(0, 1), mo(1, 0)]
        tracker.update(population, [0, 1], count_stagnation=False)
        tracker.update(population, [0, 1], count_stagnation=True)
        self.assertEqual(tracker.stagnation_count, 1)
        tracker.update([mo(0, 1), mo(0.5, 0.2)], [0, 1], count_stagnation=True)
        self.assertEqual(tracker.stagnation_count, 0)
        with self.assertRaises(RuntimeError):
            tracker.update(population, None, count_stagnation=True)

    def test_restart_due_needs_both_counters(self):
        tracker = self.scalar()
        tracker.stagnation_count, tracker.generations_since_restart = 5, 3
        self.assertFalse(tracker.restart_due(5))
        tracker.generations_since_restart = 5
        self.assertTrue(tracker.restart_due(5))
        self.assertFalse(tracker.restart_due(None))

    def test_record_requires_a_best_individual(self):
        with self.assertRaises(RuntimeError):
            self.scalar().record()

    def test_state_roundtrip(self):
        tracker = self.scalar()
        tracker.update([ind(3)], None, count_stagnation=False)
        tracker.record()
        tracker.stagnation_count = 4
        other = self.scalar()
        other.load_state_dict(tracker.state_dict(), tracker.best_individual)
        self.assertEqual(other.state_dict(), tracker.state_dict())
        self.assertIs(other.best_individual, tracker.best_individual)


class MigrationTests(unittest.TestCase):
    def test_scalar_emigrants_are_the_best_clones(self):
        population = [ind(5, 1), ind(1, 2), ind(3, 3)]
        migrants = select_emigrants(population, 2, multi_objective=False, score=lambda i: i.fitness)
        self.assertEqual([m.fitness for m in migrants], [1.0, 3.0])
        self.assertTrue(all(m is not p for m in migrants for p in population))

    def test_multi_objective_emigrants_come_from_the_first_front(self):
        population = [mo(0, 4), mo(1, 3), mo(2, 2), mo(3, 1), mo(4, 0), mo(5, 5)]
        migrants = select_emigrants(population, 3, multi_objective=True, score=lambda i: 0.0)
        self.assertEqual(len(migrants), 3)
        self.assertNotIn((5.0, 5.0), [m.fitness for m in migrants])
        self.assertTrue(all(math.isinf(m.crowding_distance) or m.rank == 0 for m in migrants))

    def test_empty_inputs(self):
        self.assertEqual(select_emigrants([], 3, multi_objective=False, score=lambda i: 0.0), [])
        self.assertEqual(select_emigrants([ind(1)], 0, multi_objective=False, score=lambda i: 0.0), [])

    def test_merge_keeps_best_and_leaves_ages_alone(self):
        population = [ind(5, 1, age=7), ind(4, 2, age=3)]
        merged = merge_immigrants(
            population, [ind(1, 3, age=9)], 2, multi_objective=False, score=lambda i: i.fitness
        )
        self.assertEqual([(m.fitness, m.age) for m in merged], [(1.0, 9), (4.0, 3)])

    def test_merge_multi_objective_returns_requested_size(self):
        population = [mo(0, 4), mo(2, 2), mo(4, 0), mo(5, 5)]
        merged = merge_immigrants(population, [mo(1, 1)], 4, multi_objective=True, score=lambda i: 0.0)
        self.assertEqual(len(merged), 4)
        self.assertIn((1.0, 1.0), [m.fitness for m in merged])


class SurvivalPolicyTests(unittest.TestCase):
    def policy(self, **overrides):
        settings = dict(size=3, multi_objective=False, generational=True, elitism_count=1, score=lambda i: i.fitness)
        settings.update(overrides)
        return SurvivalPolicy(**settings)

    def test_generational_requires_full_offspring_set(self):
        parents = [ind(1), ind(2), ind(3)]
        with self.assertRaises(ConfigurationError):
            self.policy().select(parents, [ind(1)])
        with self.assertRaises(ConfigurationError):
            self.policy().select(parents, [])

    def test_steady_state_accepts_any_offspring_count(self):
        parents = [ind(5), ind(6), ind(7)]
        survivors = self.policy(generational=False).select(parents, [ind(1)])
        self.assertEqual(sorted(s.fitness for s in survivors), [1.0, 5.0, 6.0])

    def test_multi_objective_keeps_the_population_size(self):
        parents = [mo(0, 4), mo(2, 2), mo(4, 0)]
        survivors = self.policy(multi_objective=True).select(parents, [mo(1, 1), mo(5, 5)])
        self.assertEqual(len(survivors), 3)
        self.assertNotIn((5.0, 5.0), [s.fitness for s in survivors])


class ReproducerTests(unittest.TestCase):
    def setUp(self):
        self.ga = GeneticAlgorithm(
            20, RealRepresentation(4, (-2, 2)), sphere, rng_seed=1, crossover_prob=1.0, mutation_prob=1.0
        )
        self.ga.initialize_population()
        self.reproducer = self.ga.reproducer

    def test_produce_returns_exactly_the_requested_number_of_valid_children(self):
        for count in (1, 2, 5, 20):
            genes, trials = self.reproducer.produce(self.ga.population, count)
            self.assertEqual((len(genes), len(trials)), (count, count))
            self.assertTrue(all(self.ga.representation.validate(g) for g in genes))

    def test_trials_flag_mutation_and_reference_the_better_parent(self):
        parent_a, parent_b = self.ga.population[0], self.ga.population[1]
        for genes, trial in self.reproducer.make_children(parent_a, parent_b):
            self.assertTrue(trial.mutated)
            better = min(parent_a.fitness, parent_b.fitness)
            self.assertEqual(trial.reference_fitness, better)

    def test_no_mutation_when_probability_is_zero(self):
        self.ga.adaptation.mutation_prob = 0.0
        children = self.reproducer.make_children(*self.ga.population[:2])
        self.assertFalse(any(trial.mutated for _, trial in children))

    def test_record_trial_feeds_the_adaptation_controller(self):
        ga = GeneticAlgorithm(20, RealRepresentation(4, (-2, 2)), sphere, rng_seed=1, adaptive_mutation=True)
        ga.initialize_population()
        worse_than_everything = Individual(np.zeros(4), 1e9)
        ga.reproducer.record_trial(worse_than_everything, MutationTrial(True, 0.0, 0.0))
        self.assertEqual(ga.adaptation.trial_count, 1)
        self.assertEqual(ga.adaptation.success_count, 0)
        better_than_reference = Individual(np.zeros(4), -1.0)
        ga.reproducer.record_trial(better_than_reference, MutationTrial(True, 0.0, 0.0))
        self.assertEqual((ga.adaptation.trial_count, ga.adaptation.success_count), (2, 1))
        ga.reproducer.record_trial(worse_than_everything, MutationTrial(False, 0.0, 0.0))
        self.assertEqual(ga.adaptation.trial_count, 2)  # unmutated children are not trials

    def test_distinct_selection(self):
        indices = self.reproducer.select_parent_indices(self.ga.population, 5, distinct=True)
        self.assertEqual(len(set(indices)), 5)


class MakeValidTests(unittest.TestCase):
    def test_rejects_wrong_shapes(self):
        rep = RealRepresentation(3, (-1, 1))
        for bad in (np.zeros(2), np.zeros((3, 1)), np.zeros(4)):
            with self.subTest(shape=bad.shape), self.assertRaises(ConfigurationError):
                rep.make_valid(bad)

    def test_repairs_and_returns_an_owned_copy_of_canonical_dtype(self):
        rep = RealRepresentation(3, (-1, 1))
        source = np.array([5, -5, 0], dtype=np.int32)
        fixed = rep.make_valid(source)
        self.assertEqual(fixed.dtype, np.float64)
        np.testing.assert_array_equal(fixed, [1.0, -1.0, 0.0])
        fixed[0] = 0.0
        self.assertEqual(source[0], 5)  # the input is untouched

    def test_faulty_repair_is_reported(self):
        class Broken(RealRepresentation):
            def repair(self, individual):
                return np.full(self.dimensions, 99.0)

        with self.assertRaises(ConfigurationError):
            Broken(2, (-1, 1)).make_valid(np.array([5.0, 5.0]))


class PerGeneBoundsTests(unittest.TestCase):
    PAIRS = [(0.0, 1.0), (-10.0, 10.0), (100.0, 500.0)]

    def test_accepts_both_forms(self):
        shared = RealRepresentation(3, (-1.0, 1.0))
        self.assertEqual(shared.describe()["bounds"], [-1.0, 1.0])
        per_gene = RealRepresentation(3, self.PAIRS)
        np.testing.assert_array_equal(per_gene.lower_bounds, [0, -10, 100])
        np.testing.assert_array_equal(per_gene.upper_bounds, [1, 10, 500])
        self.assertEqual(per_gene.describe()["bounds"], [[0.0, 1.0], [-10.0, 10.0], [100.0, 500.0]])

    def test_two_dimensions_are_not_confused_with_a_shared_pair(self):
        rep = RealRepresentation(2, [(0, 1), (5, 6)])
        np.testing.assert_array_equal(rep.lower_bounds, [0, 5])
        np.testing.assert_array_equal(RealRepresentation(2, (0, 1)).lower_bounds, [0, 0])

    def test_equal_pairs_describe_like_the_shared_form(self):
        self.assertEqual(
            RealRepresentation(2, [(0, 1), (0, 1)]).describe(), RealRepresentation(2, (0, 1)).describe()
        )

    def test_invalid_bounds(self):
        for dimensions, bounds in (
            (3, [(0, 1), (0, 1)]),  # wrong number of pairs
            (2, [(0, 1), (2, 1)]),  # lower >= upper
            (2, [(0, 1), (0, float("inf"))]),
            (2, [(0, 1, 2), (0, 1)]),
            (2, [("a", 1), (0, 1)]),
            (2, 5),
        ):
            with self.subTest(bounds=bounds), self.assertRaises(ConfigurationError):
                RealRepresentation(dimensions, bounds)

    def test_every_operator_respects_each_genes_own_range(self):
        rep = RealRepresentation(3, self.PAIRS)
        rng = np.random.default_rng(0)
        a, b = rep.random_individual(rng), rep.random_individual(rng)
        self.assertTrue(rep.validate(a))
        for method in rep.SUPPORTED_CROSSOVERS:
            for child in rep.crossover(a, b, method, rng):
                self.assertTrue(rep.validate(child), method)
        for method in rep.SUPPORTED_MUTATIONS:
            for _ in range(50):
                self.assertTrue(rep.validate(rep.mutate(a, method, rng, mutation_prob=1.0)), method)

    def test_repair_clips_each_gene_to_its_own_bounds(self):
        rep = RealRepresentation(3, self.PAIRS)
        repaired = rep.repair(np.array([7.0, -50.0, np.nan]))
        np.testing.assert_array_equal(repaired, [1.0, -10.0, 300.0])

    def test_gaussian_step_scales_with_gene_range(self):
        rep = RealRepresentation(2, [(0, 1), (0, 1000)])
        rng = np.random.default_rng(3)
        start = np.array([0.5, 500.0])
        moves = np.array([rep.mutate(start, MutationMethod.GAUSSIAN, rng, mutation_prob=1.0) - start for _ in range(400)])
        ratio = np.std(moves[:, 1]) / np.std(moves[:, 0])
        self.assertGreater(ratio, 500)  # the wide gene moves ~1000x more than the narrow one

    def test_engine_optimises_over_heterogeneous_ranges(self):
        target = np.array([0.25, -3.0, 321.0])
        span = np.array([1.0, 20.0, 400.0])
        objective = lambda x: float(np.sum(((x - target) / span) ** 2))
        ga = GeneticAlgorithm(50, RealRepresentation(3, self.PAIRS), objective, rng_seed=2)
        result = ga.run(max_generations=120)
        self.assertLess(result.best_fitness, 1e-3)
        self.assertTrue(np.all(np.abs(result.best_individual.genes - target) <= 0.1 * span))

    def test_sigma_validation(self):
        rep = RealRepresentation(2, (-1, 1))
        with self.assertRaises(ConfigurationError):
            rep.mutate(np.zeros(2), MutationMethod.GAUSSIAN, np.random.default_rng(0), sigma=-1.0)
        with self.assertRaises(ConfigurationError):
            rep.mutate(np.zeros(2), MutationMethod.GAUSSIAN, np.random.default_rng(0), sigma=float("nan"))


class WarmStartTests(unittest.TestCase):
    def make(self):
        return GeneticAlgorithm(20, RealRepresentation(3, (-5, 5)), sphere, rng_seed=1)

    def test_seeded_genomes_enter_the_population(self):
        ga = self.make()
        ga.initialize_population([np.zeros(3), np.array([1.0, 1.0, 1.0])])
        self.assertEqual(len(ga.population), 20)
        self.assertEqual(ga.best_individual.fitness, 0.0)
        np.testing.assert_array_equal(ga.population[1].genes, [1.0, 1.0, 1.0])
        self.assertEqual(ga.evaluations, 20)

    def test_seeds_are_repaired(self):
        ga = self.make()
        ga.initialize_population([np.array([99.0, 0.0, 0.0])])
        np.testing.assert_array_equal(ga.population[0].genes, [5.0, 0.0, 0.0])

    def test_invalid_seeds(self):
        ga = self.make()
        with self.assertRaises(ConfigurationError):
            ga.initialize_population([np.zeros(3)] * 21)
        with self.assertRaises(ConfigurationError):
            ga.initialize_population([np.zeros(2)])
        self.assertEqual(ga.population, [])  # nothing was initialised

    def test_warm_start_then_resume_keeps_the_seed_as_best(self):
        ga = self.make()
        ga.initialize_population([np.zeros(3)])
        result = ga.run(max_generations=5, resume=True)
        self.assertEqual(result.best_fitness, 0.0)


class CallbackTests(unittest.TestCase):
    def make(self, **kwargs):
        return GeneticAlgorithm(20, RealRepresentation(3, (-5, 5)), sphere, rng_seed=1, **kwargs)

    def test_called_after_initialisation_and_every_generation(self):
        seen = []
        self.make().run(max_generations=4, on_generation=lambda ga: seen.append(ga.generation))
        self.assertEqual(seen, [0, 1, 2, 3, 4])

    def test_truthy_return_stops_the_run(self):
        result = self.make().run(max_generations=100, on_generation=lambda ga: ga.generation >= 3)
        self.assertEqual((result.stop_reason, result.generation, result.converged), ("callback", 3, False))

    def test_stopping_at_generation_zero(self):
        result = self.make().run(max_generations=100, on_generation=lambda ga: True)
        self.assertEqual((result.stop_reason, result.generation), ("callback", 0))

    def test_not_called_before_the_first_step_when_resuming(self):
        ga = self.make()
        ga.run(max_generations=2)
        seen = []
        ga.run(max_generations=4, resume=True, on_generation=lambda g: seen.append(g.generation))
        self.assertEqual(seen, [3, 4])

    def test_callback_can_read_engine_state(self):
        bests = []
        self.make().run(max_generations=5, on_generation=lambda ga: bests.append(ga.best_individual.fitness))
        self.assertTrue(all(b2 <= b1 for b1, b2 in zip(bests, bests[1:])))


class EngineWiringTests(unittest.TestCase):
    def test_components_are_exposed_and_share_state(self):
        ga = GeneticAlgorithm(10, RealRepresentation(2), sphere, rng_seed=1, adaptive_mutation=True)
        self.assertIs(ga.reproducer.adaptation, ga.adaptation)
        self.assertIs(ga.reproducer.rng, ga.rng)
        self.assertIs(ga.reproducer.comparator, ga.comparator)
        ga.run(max_generations=3)
        self.assertEqual(ga.best_fitness_history, ga.progress.history)
        self.assertEqual(ga.stats, ga.statistics.data)
        self.assertEqual(len(ga.best_fitness_history), 4)

    def test_mutation_setters_write_through_to_the_controller(self):
        ga = GeneticAlgorithm(10, RealRepresentation(2), sphere)
        ga.mutation_prob = 0.5
        ga.gene_mutation_prob = 0.25
        self.assertEqual((ga.adaptation.mutation_prob, ga.adaptation.gene_mutation_prob), (0.5, 0.25))


if __name__ == "__main__":
    unittest.main()
