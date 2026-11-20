import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ga_engine import (
    BinaryRepresentation,
    CheckpointError,
    GeneticAlgorithm,
    RealRepresentation,
)
from ga_engine.benchmarks import onemax, sphere, zdt1
from ga_engine.checkpoint import _atomic_write, decode_value, encode_value


def scalar_ga(**kwargs):
    settings = dict(rng_seed=11)
    settings.update(kwargs)
    return GeneticAlgorithm(30, RealRepresentation(4, (-5, 5)), sphere, **settings)


def mo_ga(**kwargs):
    settings = dict(rng_seed=12, multi_objective=True)
    settings.update(kwargs)
    return GeneticAlgorithm(30, RealRepresentation(5, (0, 1)), zdt1, **settings)


class ValueEncodingTests(unittest.TestCase):
    def test_non_finite_floats_roundtrip_as_tagged_objects(self):
        original = {"a": [math.inf, -math.inf, 1.5], "b": (math.nan,), "c": np.float64(2.0), "d": np.int64(3)}
        encoded = encode_value(original)
        text = json.dumps(encoded, allow_nan=False)  # must not raise
        self.assertNotIn("Infinity", text)
        decoded = decode_value(json.loads(text))
        self.assertEqual(decoded["a"], [math.inf, -math.inf, 1.5])
        self.assertTrue(math.isnan(decoded["b"][0]))
        self.assertEqual((decoded["c"], decoded["d"]), (2.0, 3))


class FileFormatTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_json_checkpoint_is_strict_json(self):
        ga = mo_ga()
        ga.run(max_generations=3)  # crowding distances are infinite on the boundary
        path = self.dir / "state.json"
        ga.save_checkpoint(path)

        def reject(token):
            raise AssertionError(f"non-standard JSON token {token}")

        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)

    def test_roundtrip_preserves_population_for_both_formats(self):
        for maker in (scalar_ga, mo_ga):
            for suffix in (".json", ".npz"):
                with self.subTest(maker=maker.__name__, suffix=suffix):
                    ga = maker()
                    ga.run(max_generations=5)
                    path = self.dir / f"{maker.__name__}{suffix}"
                    ga.save_checkpoint(path)
                    other = maker(rng_seed=999)
                    other.load_checkpoint(path)
                    self.assertEqual(other.generation, ga.generation)
                    for a, b in zip(ga.population, other.population):
                        np.testing.assert_array_equal(a.genes, b.genes)
                        self.assertEqual(a.genes.dtype, b.genes.dtype)
                        self.assertEqual((a.fitness, a.age), (b.fitness, b.age))
                    self.assertEqual(other.best_fitness_history, ga.best_fitness_history)
                    self.assertEqual(other.get_stats(), ga.get_stats())

    def test_binary_genes_keep_their_dtype(self):
        ga = GeneticAlgorithm(10, BinaryRepresentation(8), onemax, rng_seed=1)
        ga.run(max_generations=2)
        ga.save_checkpoint(self.dir / "bin.json")
        other = GeneticAlgorithm(10, BinaryRepresentation(8), onemax, rng_seed=2)
        other.load_checkpoint(self.dir / "bin.json")
        self.assertEqual(other.population[0].genes.dtype, np.int8)

    def test_unsupported_suffix_and_empty_population(self):
        ga = scalar_ga()
        with self.assertRaises(CheckpointError):
            ga.save_checkpoint(self.dir / "x.txt")
        with self.assertRaises(CheckpointError):
            ga.save_checkpoint(self.dir / "x.json")  # not initialised yet
        with self.assertRaises(CheckpointError):
            ga.load_checkpoint(self.dir / "x.txt")

    def test_missing_file_propagates_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            scalar_ga().load_checkpoint(self.dir / "absent.json")

    def test_corrupt_files_raise_checkpoint_error(self):
        garbage_json = self.dir / "bad.json"
        garbage_json.write_text("{not json", encoding="utf-8")
        garbage_npz = self.dir / "bad.npz"
        garbage_npz.write_bytes(b"this is not a zip archive")
        incomplete = self.dir / "incomplete.json"
        incomplete.write_text(json.dumps({"version": 3}), encoding="utf-8")
        for path in (garbage_json, garbage_npz, incomplete):
            with self.subTest(path=path.name), self.assertRaises(CheckpointError):
                scalar_ga().load_checkpoint(path)

    def test_unsupported_version(self):
        ga = scalar_ga()
        ga.run(max_generations=1)
        path = self.dir / "v.json"
        ga.save_checkpoint(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["version"] = 99
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(CheckpointError):
            scalar_ga().load_checkpoint(path)

    def test_incompatible_engines_are_rejected_without_side_effects(self):
        ga = scalar_ga()
        ga.run(max_generations=3)
        path = self.dir / "c.json"
        ga.save_checkpoint(path)
        mismatches = {
            "population": GeneticAlgorithm(20, RealRepresentation(4, (-5, 5)), sphere, rng_seed=1),
            "dimensions": GeneticAlgorithm(30, RealRepresentation(5, (-5, 5)), sphere, rng_seed=1),
            "bounds": GeneticAlgorithm(30, RealRepresentation(4, (-1, 1)), sphere, rng_seed=1),
            "representation": GeneticAlgorithm(30, BinaryRepresentation(4), onemax, rng_seed=1),
            "objective mode": GeneticAlgorithm(
                30, RealRepresentation(4, (-5, 5)), zdt1, multi_objective=True, rng_seed=1
            ),
        }
        for name, engine in mismatches.items():
            with self.subTest(name), self.assertRaises(CheckpointError):
                engine.load_checkpoint(path)
            self.assertEqual(engine.population, [], name)
            self.assertEqual(engine.generation, 0, name)

    def test_per_gene_bounds_roundtrip_and_mismatch(self):
        pairs = [(0.0, 1.0), (-10.0, 10.0), (100.0, 500.0)]

        def build(bounds, seed=3):
            return GeneticAlgorithm(20, RealRepresentation(3, bounds), lambda x: float(np.sum(x**2)), rng_seed=seed)

        ga = build(pairs)
        ga.run(max_generations=4)
        path = self.dir / "pergene.json"
        ga.save_checkpoint(path)
        other = build(pairs, seed=99)
        other.load_checkpoint(path)
        for a, b in zip(ga.population, other.population):
            np.testing.assert_array_equal(a.genes, b.genes)
        for wrong in ((-1.0, 1.0), [(0.0, 1.0), (-10.0, 10.0), (100.0, 600.0)]):
            with self.subTest(wrong=wrong), self.assertRaises(CheckpointError):
                build(wrong).load_checkpoint(path)

    def test_checkpoint_with_invalid_genes_is_rejected(self):
        ga = scalar_ga()
        ga.run(max_generations=1)
        path = self.dir / "g.json"
        ga.save_checkpoint(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["population"][0]["genes"][0] = 1e9  # far outside the bounds
        path.write_text(json.dumps(payload), encoding="utf-8")
        target = scalar_ga()
        with self.assertRaises(CheckpointError):
            target.load_checkpoint(path)
        self.assertEqual(target.population, [])

    def test_atomic_write_leaves_original_intact_on_failure(self):
        target = self.dir / "keep.json"
        target.write_text("original", encoding="utf-8")

        def failing_writer(handle):
            handle.write(b"partial")
            raise OSError("disk full")

        with self.assertRaises(OSError):
            _atomic_write(target, failing_writer)
        self.assertEqual(target.read_text(encoding="utf-8"), "original")
        self.assertEqual([p.name for p in self.dir.iterdir()], ["keep.json"])  # no temp file left

    def test_successful_save_leaves_no_temporary_files(self):
        ga = scalar_ga()
        ga.run(max_generations=1)
        ga.save_checkpoint(self.dir / "ok.npz")
        self.assertEqual([p.name for p in self.dir.iterdir()], ["ok.npz"])

    def test_reads_legacy_version_2_files(self):
        ga = scalar_ga()
        ga.run(max_generations=4)
        path = self.dir / "legacy.json"
        ga.save_checkpoint(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["version"] = 2
        for key in ("generations_since_restart", "evaluations", "evaluation_failures", "bounds"):
            del payload[key]
        for key in ("evaluations", "evaluation_failures"):
            del payload["stats"][key]
        path.write_text(json.dumps(payload), encoding="utf-8")

        other = scalar_ga(rng_seed=5)
        other.load_checkpoint(path)
        other.run(max_generations=8, resume=True)
        rows = len(other.get_stats()["generation"])
        self.assertEqual(rows, 9)
        self.assertTrue(all(len(v) == rows for v in other.get_stats().values()))


class ResumeReproducibilityTests(unittest.TestCase):
    """A run resumed from a checkpoint must equal the uninterrupted run."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def assert_same_run(self, full, resumed):
        self.assertEqual(resumed.best_fitness, full.best_fitness)
        np.testing.assert_array_equal(resumed.best_individual.genes, full.best_individual.genes)
        self.assertEqual(resumed.generation, full.generation)
        self.assertEqual(resumed.stats, full.stats)

    def check(self, factory, suffix, split=15, total=35):
        full = factory().run(max_generations=total)
        first = factory()
        first.run(max_generations=split)
        path = self.dir / f"ck{suffix}"
        first.save_checkpoint(path)
        second = factory()
        second.load_checkpoint(path)
        resumed = second.run(max_generations=total, resume=True)
        self.assert_same_run(full, resumed)

    def test_single_objective(self):
        for suffix in (".json", ".npz"):
            with self.subTest(suffix):
                self.check(scalar_ga, suffix)

    def test_with_adaptive_mutation(self):
        for suffix in (".json", ".npz"):
            with self.subTest(suffix):
                self.check(lambda: scalar_ga(adaptive_mutation=True, adaptation_window=5), suffix)

    def test_with_restarts(self):
        self.check(lambda: scalar_ga(restart_on_stagnation=3), ".json", split=12, total=40)

    def test_multi_objective(self):
        for suffix in (".json", ".npz"):
            with self.subTest(suffix):
                self.check(mo_ga, suffix)


if __name__ == "__main__":
    unittest.main()
