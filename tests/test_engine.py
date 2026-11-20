import math
import unittest

import numpy as np

from ga_engine import (
    BinaryRepresentation,
    ConfigurationError,
    CrossoverMethod,
    EvaluationError,
    EvaluationErrorPolicy,
    GeneticAlgorithm,
    MutationMethod,
    PermutationRepresentation,
    RealRepresentation,
    Representation,
    ReplacementMethod,
    SelectionMethod,
)
from ga_engine.benchmarks import KnapsackProblem, TspProblem, onemax, sphere, zdt1

from . import helpers


def real_ga(**kwargs):
    settings = dict(population_size=40, representation=RealRepresentation(5, (-5, 5)), fitness_function=sphere, rng_seed=1)
    settings.update(kwargs)
    return GeneticAlgorithm(**settings)


class ConfigurationTests(unittest.TestCase):
    def test_invalid_arguments_raise_configuration_error(self):
        bad = [
            dict(population_size=1),
            dict(population_size=True),
            dict(population_size=10.5),
            dict(crossover_prob=1.2),
            dict(mutation_prob=-0.1),
            dict(gene_mutation_prob=2.0),
            dict(elitism_count=41),
            dict(elitism_count=-1),
            dict(tournament_size=0),
            dict(restart_on_stagnation=0),
            dict(n_workers=-5),
            dict(mutation_sigma=-1.0),
            dict(constraint_penalty=float("nan")),
            dict(adaptation_window=0),
            dict(stagnation_tolerance=-1.0),
            dict(steady_state_offspring=0, replacement_method=ReplacementMethod.STEADY_STATE),
            dict(steady_state_offspring=999, replacement_method=ReplacementMethod.STEADY_STATE),
            dict(selection_method="tournament"),
            dict(mutation_params={"sigma": 0.5}),
            dict(mutation_params={"nonsense": 1}),
            dict(crossover_params={"eta": -3}),
            dict(crossover_params={"alpha": 1.0}),  # alpha does not belong to SBX
            dict(fitness_function="not callable"),
            dict(constraint_function=5),
        ]
        for overrides in bad:
            with self.subTest(overrides=overrides), self.assertRaises(ConfigurationError):
                real_ga(**overrides)

    def test_configuration_error_is_a_value_error(self):
        with self.assertRaises(ValueError):
            real_ga(population_size=1)

    def test_representation_must_be_a_representation(self):
        with self.assertRaises(ConfigurationError):
            real_ga(representation="real")

    def test_unsupported_operators_fail_at_construction(self):
        with self.assertRaises(ConfigurationError):
            GeneticAlgorithm(10, BinaryRepresentation(8), onemax, crossover_method=CrossoverMethod.SBX)
        with self.assertRaises(ConfigurationError):
            GeneticAlgorithm(10, PermutationRepresentation(8), TspProblem.random(8), mutation_method=MutationMethod.GAUSSIAN)

    def test_default_operators_work_for_every_builtin_representation(self):
        problems = [
            (RealRepresentation(4, (-1, 1)), sphere),
            (BinaryRepresentation(10), onemax),
            (PermutationRepresentation(8), TspProblem.random(8)),
        ]
        for representation, objective in problems:
            with self.subTest(type(representation).__name__):
                result = GeneticAlgorithm(16, representation, objective, rng_seed=2).run(max_generations=5)
                self.assertTrue(representation.validate(result.best_individual.genes))

    def test_all_selection_methods_run(self):
        for method in SelectionMethod:
            with self.subTest(method):
                result = real_ga(selection_method=method).run(max_generations=15)
                self.assertTrue(math.isfinite(result.best_fitness))

    def test_n_workers_minus_one_means_all_cpus(self):
        self.assertGreaterEqual(real_ga(n_workers=-1).n_workers, 1)

    def test_custom_representation_without_metadata_keeps_working(self):
        class Minimal(Representation):
            def random_individual(self, rng):
                return rng.uniform(-1, 1, self.dimensions)

            def crossover(self, parent1, parent2, method, rng, **kwargs):
                mask = rng.random(self.dimensions) < 0.5
                return np.where(mask, parent1, parent2), np.where(mask, parent2, parent1)

            def mutate(self, individual, method, rng, **kwargs):
                return np.clip(individual + rng.normal(0, 0.1, self.dimensions), -1, 1)

            def repair(self, individual):
                return np.clip(np.asarray(individual, dtype=float), -1, 1)

        result = GeneticAlgorithm(20, Minimal(3), sphere, rng_seed=1).run(max_generations=10)
        self.assertTrue(math.isfinite(result.best_fitness))

    def test_operator_parameters_reach_the_operator(self):
        seen = {}

        class Spy(RealRepresentation):
            def crossover(self, p1, p2, method, rng, **kwargs):
                seen["crossover"] = kwargs
                return super().crossover(p1, p2, method, rng, **kwargs)

            def mutate(self, ind, method, rng, **kwargs):
                seen["mutation"] = kwargs
                return super().mutate(ind, method, rng, **kwargs)

        GeneticAlgorithm(
            20, Spy(3, (-1, 1)), sphere, rng_seed=1, crossover_prob=1.0, mutation_prob=1.0,
            mutation_method=MutationMethod.POLYNOMIAL,
            crossover_params={"eta": 7.0}, mutation_params={"eta_m": 9.0},
        ).run(max_generations=2)
        self.assertEqual(seen["crossover"], {"eta": 7.0})
        self.assertEqual(seen["mutation"]["eta_m"], 9.0)
        self.assertIn("mutation_prob", seen["mutation"])


class ConvergenceTests(unittest.TestCase):
    def test_sphere_converges(self):
        result = real_ga(population_size=60).run(max_generations=150)
        self.assertLess(result.best_fitness, 1e-2)
        self.assertEqual(result.stop_reason, "max_generations")
        self.assertFalse(result.converged)

    def test_onemax_stops_on_target(self):
        ga = GeneticAlgorithm(40, BinaryRepresentation(24), onemax, rng_seed=3)
        result = ga.run(max_generations=300, target_fitness=0.0)
        self.assertEqual(result.stop_reason, "target_fitness")
        self.assertTrue(result.converged)
        self.assertEqual(result.best_fitness, 0.0)

    def test_tsp_improves_and_stays_a_valid_tour(self):
        problem = TspProblem.random(15, seed=4)
        representation = PermutationRepresentation(15)
        ga = GeneticAlgorithm(60, representation, problem, rng_seed=4, mutation_prob=0.3)
        result = ga.run(max_generations=120)
        initial_best = result.stats["best_fitness"][0]
        self.assertLess(result.best_fitness, initial_best)
        self.assertTrue(representation.validate(result.best_individual.genes))

    def test_constrained_knapsack_returns_feasible_solution(self):
        problem = KnapsackProblem.random(30, seed=5)
        ga = GeneticAlgorithm(
            60, BinaryRepresentation(30), problem.objective, constraint_function=problem.constraint,
            constraint_penalty=1000.0, rng_seed=5,
        )
        result = ga.run(max_generations=120)
        self.assertEqual(result.best_individual.constraint_violation, 0.0)
        self.assertLessEqual(float(problem.weights @ result.best_individual.genes), problem.capacity)
        self.assertLess(result.best_fitness, 0.0)

    def test_zdt1_front_improves_and_is_non_dominated(self):
        ga = GeneticAlgorithm(60, RealRepresentation(8, (0, 1)), zdt1, multi_objective=True, rng_seed=6)
        result = ga.run(max_generations=80)

        def gap(front):
            return np.mean([f.fitness[1] - (1 - math.sqrt(f.fitness[0])) for f in front])

        ga0 = GeneticAlgorithm(60, RealRepresentation(8, (0, 1)), zdt1, multi_objective=True, rng_seed=6)
        ga0.initialize_population()
        self.assertLess(gap(result.pareto_front), gap(ga0.get_pareto_front()))
        front = [f.fitness for f in result.pareto_front]
        for i, a in enumerate(front):
            for j, b in enumerate(front):
                if i != j:
                    self.assertFalse(all(x <= y for x, y in zip(a, b)) and a != b)
        self.assertGreater(len(front), 5)
        self.assertEqual(result.stats["pareto_front_size"][-1], len(front))

    def test_adaptive_mutation_changes_intensity(self):
        ga = real_ga(adaptive_mutation=True, adaptation_window=5)
        ga.run(max_generations=40)
        self.assertNotEqual(ga.mutation_prob, 0.1)
        self.assertIsNotNone(ga.stats["mutation_success_ratio"][-1])


class RunControlTests(unittest.TestCase):
    def test_same_seed_is_reproducible_and_different_seeds_differ(self):
        a = real_ga(rng_seed=7).run(max_generations=20)
        b = real_ga(rng_seed=7).run(max_generations=20)
        c = real_ga(rng_seed=8).run(max_generations=20)
        self.assertEqual(a.best_fitness, b.best_fitness)
        self.assertNotEqual(a.best_fitness, c.best_fitness)

    def test_zero_generations_initialises_only(self):
        result = real_ga().run(max_generations=0)
        self.assertEqual((result.generation, result.evaluations), (0, 40))

    def test_target_met_at_generation_zero(self):
        result = real_ga().run(max_generations=50, target_fitness=1e9)
        self.assertEqual((result.generation, result.stop_reason), (0, "target_fitness"))

    def test_stagnation_stop(self):
        ga = GeneticAlgorithm(20, RealRepresentation(3, (-1, 1)), helpers.flat, rng_seed=1)
        result = ga.run(max_generations=500, stagnation_generations=7)
        self.assertEqual((result.stop_reason, result.generation), ("stagnation", 7))

    def test_stagnation_tolerance_ignores_tiny_improvements(self):
        strict = real_ga(rng_seed=3).run(max_generations=300, stagnation_generations=25)
        lenient = real_ga(rng_seed=3, stagnation_tolerance=1e-3).run(max_generations=300, stagnation_generations=25)
        self.assertLessEqual(lenient.generation, strict.generation)
        self.assertEqual(lenient.stop_reason, "stagnation")

    def test_invalid_run_arguments(self):
        ga = real_ga()
        with self.assertRaises(ConfigurationError):
            ga.run(max_generations=-1)
        with self.assertRaises(ConfigurationError):
            ga.run(stagnation_generations=0)
        with self.assertRaises(ConfigurationError):
            real_ga(multi_objective=True, fitness_function=zdt1).run(target_fitness=0.0)

    def test_resume_requires_population(self):
        with self.assertRaises(RuntimeError):
            real_ga().run(resume=True)

    def test_resume_continues_the_same_run(self):
        ga = real_ga()
        ga.run(max_generations=10)
        result = ga.run(max_generations=25, resume=True)
        self.assertEqual(result.generation, 25)
        self.assertEqual(len(result.stats["generation"]), 26)

    def test_get_best_and_stats_are_copies(self):
        ga = real_ga()
        with self.assertRaises(RuntimeError):
            ga.get_best()
        ga.run(max_generations=3)
        genes, fitness = ga.get_best()
        genes[:] = 99
        self.assertNotEqual(ga.get_best()[0][0], 99)
        stats = ga.get_stats()
        stats["generation"].append(-1)
        self.assertNotEqual(ga.get_stats()["generation"][-1], -1)

    def test_pareto_front_only_in_multi_objective_mode(self):
        with self.assertRaises(ConfigurationError):
            real_ga().get_pareto_front()

    def test_public_step_api(self):
        ga = real_ga()
        ga.initialize_population()
        parents = ga.select_parents()
        self.assertIsNot(parents[0], ga.population[0])
        children = ga.create_offspring(*parents)
        self.assertEqual(len(children), 2)
        ga.step()
        self.assertEqual(ga.generation, 1)

    def test_statistics_have_one_aligned_row_per_generation(self):
        stats = real_ga().run(max_generations=6).stats
        self.assertTrue(all(len(v) == 7 for v in stats.values()))
        self.assertEqual(stats["evaluations"][-1], 40 + 6 * 40)


class ReplacementTests(unittest.TestCase):
    def test_generational_requires_full_offspring_set(self):
        ga = real_ga()
        ga.initialize_population()
        with self.assertRaises(ConfigurationError):
            ga.replacement([ga.population[0].clone()])
        with self.assertRaises(ConfigurationError):
            ga.replacement([])

    def test_steady_state_creates_few_offspring_per_generation(self):
        ga = real_ga(replacement_method=ReplacementMethod.STEADY_STATE)
        ga.initialize_population()
        before = ga.evaluations
        ga.step()
        self.assertEqual(ga.evaluations - before, 4)  # max(2, 40 // 10)
        self.assertEqual(len(ga.population), 40)

    def test_steady_state_offspring_override_and_improvement(self):
        ga = real_ga(replacement_method=ReplacementMethod.STEADY_STATE, steady_state_offspring=2)
        result = ga.run(max_generations=300)
        self.assertEqual(result.evaluations, 40 + 300 * 2)
        self.assertLess(result.best_fitness, result.stats["best_fitness"][0])

    def test_steady_state_best_fitness_never_worsens(self):
        ga = real_ga(replacement_method=ReplacementMethod.STEADY_STATE)
        result = ga.run(max_generations=60)
        best = result.stats["best_fitness"]
        self.assertTrue(all(b2 <= b1 for b1, b2 in zip(best, best[1:])))

    def test_full_lambda_steady_state_equals_mu_plus_lambda(self):
        ga = real_ga(replacement_method=ReplacementMethod.STEADY_STATE, steady_state_offspring=40)
        ga.initialize_population()
        before = ga.evaluations
        ga.step()
        self.assertEqual(ga.evaluations - before, 40)


class ConstraintTests(unittest.TestCase):
    def test_solution_respects_constraint_when_penalty_is_large(self):
        # minimise sphere subject to x0 >= 1 (violation = 1 - x0)
        ga = GeneticAlgorithm(
            60, RealRepresentation(3, (-5, 5)), sphere, rng_seed=1, constraint_penalty=100.0,
            constraint_function=lambda x: 1.0 - float(x[0]),
        )
        result = ga.run(max_generations=120)
        self.assertGreaterEqual(result.best_individual.genes[0], 1.0 - 1e-9)
        self.assertLess(result.best_fitness, 1.2)

    def test_constrained_multi_objective_prefers_feasible(self):
        ga = GeneticAlgorithm(
            40, RealRepresentation(4, (0, 1)), zdt1, multi_objective=True, rng_seed=2,
            constraint_function=lambda x: 0.3 - float(x[0]),  # require x0 >= 0.3
        )
        result = ga.run(max_generations=40)
        self.assertTrue(all(ind.constraint_violation == 0.0 for ind in result.pareto_front))
        self.assertTrue(all(ind.fitness[0] >= 0.3 - 1e-9 for ind in result.pareto_front))


class FailureHandlingTests(unittest.TestCase):
    def test_failures_are_counted_in_results_and_statistics(self):
        ga = GeneticAlgorithm(30, RealRepresentation(3, (-1, 1)), helpers.sometimes_fails, rng_seed=1)
        with self.assertLogs("ga_engine.evaluation", level="WARNING"):
            result = ga.run(max_generations=5)
        self.assertGreater(result.evaluation_failures, 0)
        self.assertEqual(result.stats["evaluation_failures"][-1], result.evaluation_failures)
        self.assertTrue(math.isfinite(result.best_fitness))

    def test_raise_policy_aborts_the_run(self):
        ga = GeneticAlgorithm(
            10, RealRepresentation(3, (-1, 1)), helpers.always_fails, rng_seed=1,
            evaluation_error_policy=EvaluationErrorPolicy.RAISE,
        )
        with self.assertRaises(EvaluationError):
            ga.run(max_generations=3)


class ParallelEngineTests(unittest.TestCase):
    def test_parallel_run_matches_serial_run(self):
        serial = real_ga(rng_seed=9).run(max_generations=8)
        with real_ga(rng_seed=9, n_workers=2) as ga:
            parallel = ga.run(max_generations=8)
        self.assertEqual(parallel.best_fitness, serial.best_fitness)
        self.assertEqual(parallel.stats["mean_fitness"], serial.stats["mean_fitness"])

    def test_context_manager_closes_pool_even_on_error(self):
        ga = real_ga(n_workers=2)
        with self.assertRaises(RuntimeError):
            with ga:
                self.assertTrue(ga.evaluator.pool_active)
                raise RuntimeError("fail inside block")
        self.assertFalse(ga.evaluator.pool_active)

    def test_run_releases_pool_when_it_started_it(self):
        ga = real_ga(n_workers=2)
        ga.run(max_generations=2)
        self.assertFalse(ga.evaluator.pool_active)


if __name__ == "__main__":
    unittest.main()
