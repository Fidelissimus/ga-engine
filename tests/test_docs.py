"""The documentation is tested: code runs, outputs match, links resolve, references are current."""

import contextlib
import importlib.util
import inspect
import io
import logging
import re
import unittest
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional

import ga_engine

ROOT = Path(__file__).resolve().parent.parent
MARKDOWN_FILES = sorted(
    [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md", ROOT / "LLM_GUIDE.md"]
    + list((ROOT / "docs").glob("*.md"))
)
MARKDOWN_FILES = [p for p in MARKDOWN_FILES if p.exists()]
#: Files whose Python blocks are executed (the generated reference only shows signatures).
EXECUTED_FILES = [p for p in MARKDOWN_FILES if p.name not in ("api.md", "CHANGELOG.md")]
SKIP_MARKER = "# doctest: skip"


class Block(NamedTuple):
    """One fenced code block of a Markdown file."""

    language: str
    code: str
    line: int


def fenced_blocks(path: Path) -> List[Block]:
    """Parse the fenced code blocks of a Markdown file."""
    blocks: List[Block] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^```(\w*)\s*$", lines[index])
        if match:
            start = index + 1
            index += 1
            body: List[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                body.append(lines[index])
                index += 1
            blocks.append(Block(match.group(1), "\n".join(body) + "\n", start))
        index += 1
    return blocks


def runnable_blocks(path: Path) -> Iterator[tuple]:
    """Yield ``(block, expected_output_or_None)`` for every executable Python block."""
    blocks = fenced_blocks(path)
    for position, block in enumerate(blocks):
        if block.language != "python" or block.code.lstrip().startswith(SKIP_MARKER):
            continue
        following: Optional[Block] = blocks[position + 1] if position + 1 < len(blocks) else None
        expected = None
        if following is not None and following.language == "output":
            expected = following.code
        yield block, expected


class ExecutableDocumentationTests(unittest.TestCase):
    def test_python_blocks_run_and_outputs_match(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        executed = 0
        for path in EXECUTED_FILES:
            for block, expected in runnable_blocks(path):
                label = f"{path.relative_to(ROOT)}:{block.line}"
                buffer = io.StringIO()
                with self.subTest(block=label), contextlib.redirect_stdout(buffer):
                    exec(compile(block.code, label, "exec"), {"__name__": "__main__"})
                    if expected is not None:
                        self.assertEqual(buffer.getvalue().rstrip(), expected.rstrip(), label)
                executed += 1
        self.assertGreater(executed, 15)  # guards against the parser silently finding nothing


class LinkTests(unittest.TestCase):
    @staticmethod
    def anchors(path: Path) -> set:
        text = path.read_text(encoding="utf-8")
        headings = re.findall(r"^#{1,6}\s+(.*?)\s*$", text, flags=re.M)
        slugs = set()
        for heading in headings:
            heading = re.sub(r"`", "", heading)
            slugs.add(re.sub(r"[^\w\- ]", "", heading.lower()).strip().replace(" ", "-"))
        return slugs

    def test_relative_links_and_anchors_resolve(self):
        broken: List[str] = []
        for path in MARKDOWN_FILES:
            in_code = False
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.startswith("```"):
                    in_code = not in_code
                if in_code:
                    continue
                for target in re.findall(r"\]\(([^)\s]+)\)", line):
                    if target.startswith(("http://", "https://", "mailto:")):
                        continue
                    file_part, _, anchor = target.partition("#")
                    destination = (path.parent / file_part).resolve() if file_part else path
                    if not destination.exists():
                        broken.append(f"{path.name}:{number} -> {target} (missing file)")
                    elif anchor and destination.suffix == ".md" and anchor not in self.anchors(destination):
                        broken.append(f"{path.name}:{number} -> {target} (missing anchor)")
        self.assertEqual(broken, [])


class ReferenceTests(unittest.TestCase):
    def test_api_reference_is_up_to_date(self):
        spec = importlib.util.spec_from_file_location("generate_api_docs", ROOT / "tools" / "generate_api_docs.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        current = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
        self.assertEqual(current, module.generate(), "run: python tools/generate_api_docs.py")

    def test_configuration_reference_documents_every_constructor_and_run_parameter(self):
        text = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
        parameters = set(inspect.signature(ga_engine.GeneticAlgorithm.__init__).parameters) - {"self"}
        parameters |= set(inspect.signature(ga_engine.GeneticAlgorithm.run).parameters) - {"self"}
        missing = sorted(p for p in parameters if f"`{p}`" not in text)
        self.assertEqual(missing, [])

    def test_changelog_mentions_the_current_version(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"[{ga_engine.__version__}]", text)


class LlmGuideTests(unittest.TestCase):
    """The LLM briefing must cover the whole public surface, so it cannot go stale."""

    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / "LLM_GUIDE.md").read_text(encoding="utf-8")

    def test_every_public_name_is_mentioned(self):
        missing = [n for n in ga_engine.__all__ if n != "__version__" and n not in self.text]
        self.assertEqual(missing, [])

    def test_every_engine_and_run_parameter_is_mentioned(self):
        parameters: Dict[str, str] = {}
        for owner, function in (
            ("GeneticAlgorithm", ga_engine.GeneticAlgorithm.__init__),
            ("GeneticAlgorithm.run", ga_engine.GeneticAlgorithm.run),
            ("IslandModel", ga_engine.IslandModel.__init__),
            ("IslandModel.run", ga_engine.IslandModel.run),
        ):
            for name in inspect.signature(function).parameters:
                if name != "self":
                    parameters[name] = owner
        missing = sorted(f"{owner}({name})" for name, owner in parameters.items() if f"`{name}`" not in self.text)
        self.assertEqual(missing, [])

    def test_parameter_tables_match_the_signatures(self):
        spec = importlib.util.spec_from_file_location("generate_api_docs", ROOT / "tools" / "generate_api_docs.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)

        def table(marker):
            section = re.search(rf"<!-- {marker}:start -->(.*?)<!-- {marker}:end -->", self.text, re.S).group(1)
            return {n: d.strip() for n, d in re.findall(r"^\| `(\w+)` \| (.*?) \|", section, flags=re.M)}

        def expected(function):
            result = {}
            for name, parameter in inspect.signature(function).parameters.items():
                if name == "self":
                    continue
                result[name] = (
                    "required" if parameter.default is inspect.Parameter.empty
                    else f"`{generator.render_default(parameter.default)}`"
                )
            return result

        self.assertEqual(table("params"), expected(ga_engine.GeneticAlgorithm.__init__))
        self.assertEqual(table("run-params"), expected(ga_engine.GeneticAlgorithm.run))
        self.assertEqual(expected(ga_engine.GeneticAlgorithm.run), expected(ga_engine.IslandModel.run))

    def test_every_enum_member_is_mentioned(self):
        import enum

        missing = []
        for name in ga_engine.__all__:
            symbol = getattr(ga_engine, name)
            if inspect.isclass(symbol) and issubclass(symbol, enum.Enum):
                missing += [f"{name}.{m.name}" for m in symbol if m.name not in self.text]
        self.assertEqual(missing, [])

    def test_llms_txt_points_at_existing_files(self):
        index = (ROOT / "llms.txt").read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\s]+)\)", index):
            if not target.startswith("http"):
                self.assertTrue((ROOT / target.split("#")[0]).exists(), target)


class TroubleshootingTests(unittest.TestCase):
    def test_quoted_error_messages_exist_in_the_source(self):
        """Every message fragment shown in the troubleshooting table must be a real one."""
        text = (ROOT / "docs" / "troubleshooting.md").read_text(encoding="utf-8")
        source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "ga_engine").rglob("*.py"))
        source = re.sub(r'"\s*\n\s*(f?)"', "", source)  # join implicitly concatenated string literals
        fragments = re.findall(r"^\| `([^`]+)` \|", text, flags=re.M)
        self.assertGreater(len(fragments), 8)
        missing = [f for f in fragments if f not in source]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
