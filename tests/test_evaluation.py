import math
import unittest

import numpy as np

from ga_engine import EvaluationError, EvaluationErrorPolicy
from ga_engine.benchmarks import sphere
from ga_engine.evaluation import Evaluator

from . import helpers


def batch(n=6, dims=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, dims) for _ in range(n)]


class SerialEvaluatorTests(unittest.TestCase):
    def test_returns_fitness_and_violation_aligned_with_input(self):
        genes = batch()
        evaluator = Evaluator(sphere, helpers.violation_of_first_gene)
        fitnesses, violations = evaluator.evaluate(genes)
        self.assertEqual(fitnesses, [sphere(g) for g in genes])
        self.assertEqual(violations, [max(0.0, float(g[0])) for g in genes])
        self.assertEqual(evaluator.evaluations, 6)
        self.assertEqual(evaluator.failures, 0)

    def test_empty_batch(self):
        self.assertEqual(Evaluator(sphere).evaluate([]), ([], []))

    def test_failures_are_penalised_logged_and_counted(self):
        genes = [np.array([-1.0, 0.0]), np.array([1.0, 0.0]), np.array([-0.5, 0.5])]
        evaluator = Evaluator(helpers.sometimes_fails)
        with self.assertLogs("ga_engine.evaluation", level="WARNING") as logs:
            fitnesses, violations = evaluator.evaluate(genes)
        self.assertEqual(fitnesses[1], math.inf)
        self.assertEqual(violations[1], math.inf)
        self.assertEqual(evaluator.failures, 1)
        self.assertIn("ValueError: boom", logs.output[0])
        self.assertIn("1 of 3", logs.output[0])

    def test_raise_policy(self):
        evaluator = Evaluator(helpers.always_fails, error_policy=EvaluationErrorPolicy.RAISE)
        with self.assertRaises(EvaluationError) as context:
            evaluator.evaluate(batch(2))
        self.assertIn("RuntimeError: nope", str(context.exception))

    def test_nan_fitness_is_infinite_but_not_a_failure(self):
        evaluator = Evaluator(helpers.returns_nan)
        fitnesses, _ = evaluator.evaluate(batch(2))
        self.assertEqual(fitnesses, [math.inf, math.inf])
        self.assertEqual(evaluator.failures, 0)

    def test_invalid_result_counts_as_failure(self):
        evaluator = Evaluator(lambda g: "not a number")
        with self.assertLogs("ga_engine.evaluation", level="WARNING"):
            fitnesses, _ = evaluator.evaluate(batch(2))
        self.assertEqual(fitnesses, [math.inf, math.inf])
        self.assertEqual(evaluator.failures, 2)

    def test_keyboard_interrupt_is_not_swallowed(self):
        def interrupt(genes):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            Evaluator(interrupt).evaluate(batch(1))


class MultiObjectiveEvaluatorTests(unittest.TestCase):
    def test_infers_objective_count_and_uses_it_for_failures(self):
        calls = {"n": 0}

        def sometimes_bad(genes):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("first fails")
            return float(genes[0]), float(genes[1])

        evaluator = Evaluator(sometimes_bad, multi_objective=True)
        with self.assertLogs("ga_engine.evaluation", level="WARNING"):
            fitnesses, _ = evaluator.evaluate(batch(3, dims=2))
        self.assertEqual(evaluator.objective_count, 2)
        self.assertEqual(fitnesses[0], (math.inf, math.inf))

    def test_inconsistent_objective_counts_raise(self):
        genes = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        with self.assertRaises(EvaluationError):
            Evaluator(helpers.inconsistent_objectives, multi_objective=True).evaluate(genes)

    def test_all_failing_first_batch_cannot_infer(self):
        with self.assertRaises(EvaluationError):
            Evaluator(helpers.always_fails, multi_objective=True).evaluate(batch(2))

    def test_scalar_result_in_multi_objective_mode_is_a_failure(self):
        evaluator = Evaluator(sphere, multi_objective=True)
        with self.assertRaises(EvaluationError):
            evaluator.evaluate(batch(2))


class ParallelEvaluatorTests(unittest.TestCase):
    def test_parallel_results_match_serial_results(self):
        genes = batch(10)
        serial = Evaluator(sphere, helpers.violation_of_first_gene).evaluate(genes)
        parallel = Evaluator(sphere, helpers.violation_of_first_gene, n_workers=2)
        try:
            self.assertTrue(parallel.start())
            self.assertFalse(parallel.start())  # already running
            self.assertTrue(parallel.pool_active)
            self.assertEqual(parallel.evaluate(genes), serial)
        finally:
            parallel.close()
        self.assertFalse(parallel.pool_active)

    def test_failures_cross_process_boundary_with_their_message(self):
        evaluator = Evaluator(helpers.always_fails, n_workers=2, error_policy=EvaluationErrorPolicy.RAISE)
        try:
            with self.assertRaises(EvaluationError) as context:
                evaluator.evaluate(batch(4))
        finally:
            evaluator.terminate()
        self.assertIn("RuntimeError: nope", str(context.exception))

    def test_unpicklable_function_is_rejected_up_front(self):
        evaluator = Evaluator(lambda g: 0.0, n_workers=2)
        with self.assertRaises(TypeError):
            evaluator.start()

    def test_serial_mode_never_creates_a_pool(self):
        evaluator = Evaluator(sphere, n_workers=1)
        self.assertFalse(evaluator.start())
        self.assertFalse(evaluator.pool_active)


if __name__ == "__main__":
    unittest.main()
