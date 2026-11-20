"""Cross-cutting guarantees that are easy to break by accident.

Each class pins down one behaviour that several modules cooperate on (operator
semantics, selection pressure, restarts, logging hygiene, strict checkpoints,
exact resume, no swallowed exceptions).
"""

import logging
import math
import unittest

import numpy as np

from ga_engine import (
    BinaryRepresentation,
    CrossoverMethod,
    GeneticAlgorithm,
    PermutationRepresentation,
    RealRepresentation,
    ReplacementMethod,
    SelectionMethod,
    enable_console_logging,
)
from ga_engine.benchmarks import onemax, sphere

from . import helpers


class RealExchangeCrossoverKeepsChildrenDistinct(unittest.TestCase):
    def test_children_are_not_both_the_parent_midpoint(self):
        rep = RealRepresentation(6, (-5, 5))
        rng = np.random.default_rng(0)
        p1, p2 = rng.uniform(-5, 5, 6), rng.uniform(-5, 5, 6)
        for method in (CrossoverMethod.SINGLE_POINT, CrossoverMethod.TWO_POINT):
            c1, c2 = rep.crossover(p1, p2, method, rng)
            self.assertFalse(np.allclose(c1, c2), method)
            self.assertFalse(np.allclose(c1, (p1 + p2) / 2), method)


class TwoDimensionalTwoPointCrossover(unittest.TestCase):
    def test_does_not_raise(self):
        rep = RealRepresentation(2, (-5, 5))
        rep.crossover(np.array([1.0, 2.0]), np.array([3.0, 4.0]), CrossoverMethod.TWO_POINT, np.random.default_rng(0))


class SelectionExertsPressure(unittest.TestCase):
    def test_roulette_and_rank_batches_are_not_permutations(self):
        for method in (SelectionMethod.ROULETTE, SelectionMethod.RANK):
            ga = GeneticAlgorithm(30, RealRepresentation(5, (-5, 5)), sphere, selection_method=method, rng_seed=1)
            ga.initialize_population()
            indices = ga.reproducer.select_parent_indices(ga.population, 30)
            self.assertLess(len(set(indices)), 30, method)

    def test_sus_pairs_are_not_population_order_neighbours(self):
        ga = GeneticAlgorithm(40, RealRepresentation(4, (-5, 5)), sphere, selection_method=SelectionMethod.SUS, rng_seed=1)
        ga.initialize_population()
        indices = ga.reproducer.select_parent_indices(ga.population, 40)
        self.assertNotEqual(indices, sorted(indices))


class RestartNewcomersEnterPopulation(unittest.TestCase):
    def test_random_newcomers_enter_the_population(self):
        ga = GeneticAlgorithm(40, RealRepresentation(5, (-5, 5)), sphere, rng_seed=2)
        ga.run(max_generations=60)  # a converged population is better than any random genome
        before = {tuple(i.genes) for i in ga.population}
        ga._inject_random_individuals(10)
        newcomers = [i for i in ga.population if tuple(i.genes) not in before]
        self.assertEqual(len(newcomers), 10)
        self.assertEqual(len(ga.population), 40)

    def test_best_individual_always_survives_injection(self):
        ga = GeneticAlgorithm(10, RealRepresentation(3, (-5, 5)), sphere, rng_seed=3)
        ga.run(max_generations=20)
        best = min(i.fitness for i in ga.population)
        ga._inject_random_individuals(9999)
        self.assertEqual(min(i.fitness for i in ga.population), best)


class LoggingHygiene(unittest.TestCase):
    def test_verbose_never_duplicates_handlers(self):
        package_logger = logging.getLogger("ga_engine")
        before = len(package_logger.handlers)
        self.addCleanup(self._remove_console_handlers, package_logger)
        for _ in range(25):
            GeneticAlgorithm(10, RealRepresentation(2), sphere, verbose=True)
            enable_console_logging()
        console = [h for h in package_logger.handlers if getattr(h, "_ga_engine_console_handler", False)]
        self.assertEqual(len(console), 1)
        self.assertLessEqual(len(package_logger.handlers), before + 1)

    @staticmethod
    def _remove_console_handlers(package_logger):
        for handler in list(package_logger.handlers):
            if getattr(handler, "_ga_engine_console_handler", False):
                package_logger.removeHandler(handler)
        package_logger.setLevel(logging.NOTSET)

    def test_instances_do_not_register_global_loggers(self):
        registry = logging.root.manager.loggerDict
        before = len(registry)
        for _ in range(20):
            GeneticAlgorithm(10, RealRepresentation(2), sphere)
        self.assertEqual(len(registry), before)


class DefaultOperatorsPerRepresentation(unittest.TestCase):
    def test_binary_representation_runs_with_default_arguments(self):
        GeneticAlgorithm(20, BinaryRepresentation(8), onemax, rng_seed=3).run(max_generations=3)

    def test_permutation_representation_runs_with_default_arguments(self):
        GeneticAlgorithm(20, PermutationRepresentation(8), lambda t: float(np.sum(t * np.arange(8))), rng_seed=3).run(
            max_generations=3
        )


class ZeroPenaltyNeverProducesNaN(unittest.TestCase):
    def test_effective_fitness_is_never_nan(self):
        ga = GeneticAlgorithm(10, RealRepresentation(2), sphere, constraint_penalty=0.0)
        from ga_engine import Individual

        for fitness in (1.0, math.inf):
            value = ga.effective_fitness(Individual(np.zeros(2), fitness, constraint_violation=math.inf))
            self.assertFalse(math.isnan(value))


class RestartDoesNotHideStagnation(unittest.TestCase):
    def test_stagnation_still_stops_when_restart_threshold_is_smaller(self):
        ga = GeneticAlgorithm(20, RealRepresentation(3, (-1, 1)), helpers.flat, rng_seed=5, restart_on_stagnation=5)
        result = ga.run(max_generations=60, stagnation_generations=10)
        self.assertEqual((result.stop_reason, result.generation), ("stagnation", 10))

    def test_restart_fires_periodically_on_a_flat_landscape(self):
        ga = GeneticAlgorithm(20, RealRepresentation(3, (-1, 1)), helpers.flat, rng_seed=5, restart_on_stagnation=5)
        with self.assertLogs("ga_engine.engine", level="INFO") as logs:
            ga.run(max_generations=20)
        restarts = [line for line in logs.output if "injecting" in line]
        self.assertEqual(len(restarts), 4)  # generations 5, 10, 15 and 20


class OrderCrossoverWorkedExample(unittest.TestCase):
    def test_goldberg_style_example(self):
        child = PermutationRepresentation._ox_fill(
            np.arange(1, 10), np.array([9, 3, 7, 8, 2, 6, 5, 1, 4]), 3, 6
        )
        np.testing.assert_array_equal(child, [3, 8, 2, 4, 5, 6, 7, 1, 9])


class CheckpointJsonIsStrict(unittest.TestCase):
    def test_no_infinity_tokens(self):
        import json
        import tempfile
        from pathlib import Path

        from ga_engine.benchmarks import zdt1

        ga = GeneticAlgorithm(20, RealRepresentation(4, (0, 1)), zdt1, multi_objective=True, rng_seed=6)
        ga.run(max_generations=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ck.json"
            ga.save_checkpoint(path)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("Infinity", text)
        self.assertNotIn("NaN", text)
        json.loads(text, parse_constant=lambda token: self.fail(token))


class SteadyStateCreatesFewOffspring(unittest.TestCase):
    def test_default_offspring_count_is_much_smaller_than_the_population(self):
        ga = GeneticAlgorithm(
            100, RealRepresentation(3), sphere, replacement_method=ReplacementMethod.STEADY_STATE, rng_seed=7
        )
        self.assertEqual(ga.config.offspring_per_generation, 10)
        self.assertLess(ga.config.offspring_per_generation, ga.population_size)


class ResumeEqualsUninterruptedRun(unittest.TestCase):
    """A checkpoint-resumed run reproduces the uninterrupted run exactly."""

    def test_resumed_run_equals_uninterrupted_run(self):
        import tempfile
        from pathlib import Path

        def build():
            return GeneticAlgorithm(30, RealRepresentation(4, (-5, 5)), sphere, rng_seed=11)

        full = build().run(max_generations=40)
        first = build()
        first.run(max_generations=20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ck.npz"
            first.save_checkpoint(path)
            second = build()
            second.load_checkpoint(path)
        resumed = second.run(max_generations=40, resume=True)
        self.assertEqual(resumed.best_fitness, full.best_fitness)


class NoSwallowedExceptions(unittest.TestCase):
    """No broad exception handler may discard an error silently."""

    BROAD = {"Exception", "BaseException"}

    def swallowing_handlers(self):
        """Yield (file, function) for each broad handler that never re-raises."""
        import ast
        from pathlib import Path

        import ga_engine

        for path in Path(ga_engine.__file__).parent.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for handler in ast.walk(function):
                    if not isinstance(handler, ast.ExceptHandler):
                        continue
                    caught = handler.type
                    names = {n.id for n in ast.walk(caught) if isinstance(n, ast.Name)} if caught else {"BaseException"}
                    if names & self.BROAD and not any(isinstance(n, ast.Raise) for n in ast.walk(handler)):
                        yield path.name, function.name

    def test_only_the_documented_user_code_boundary_catches_without_reraising(self):
        # _call_user_functions converts the user's exception into a message that the
        # evaluator logs, counts and applies the failure policy to - nothing is lost.
        self.assertEqual(set(self.swallowing_handlers()), {("evaluation.py", "_call_user_functions")})


if __name__ == "__main__":
    unittest.main()
