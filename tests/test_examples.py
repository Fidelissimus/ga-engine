"""Every example must run to completion and be listed in the README."""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))
#: Slow by design (it times serial against parallel evaluation); parallel
#: evaluation itself is covered by the evaluator, engine and documentation tests.
SKIPPED_FOR_SPEED = {"parallel_evaluation.py"}


class ExampleTests(unittest.TestCase):
    def test_examples_exist(self):
        self.assertGreaterEqual(len(EXAMPLES), 8)

    def test_every_example_is_listed_in_the_readme(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        unlisted = [p.name for p in EXAMPLES if f"examples/{p.name}" not in readme]
        self.assertEqual(unlisted, [])

    def test_examples_run_and_print_something(self):
        environment = dict(os.environ, PYTHONPATH=str(ROOT))
        for path in EXAMPLES:
            if path.name in SKIPPED_FOR_SPEED:
                continue
            with self.subTest(example=path.name):
                completed = subprocess.run(
                    [sys.executable, str(path)],
                    cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue(completed.stdout.strip())
                self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
