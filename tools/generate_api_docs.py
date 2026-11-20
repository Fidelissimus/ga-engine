"""Generate ``docs/api.md`` from the package's docstrings and signatures.

Usage (from the project root)::

    python tools/generate_api_docs.py            # rewrite docs/api.md
    python tools/generate_api_docs.py --check    # exit 1 if docs/api.md is stale

Only the standard library is used. The output is deterministic, so a unit test
(``tests/test_docs.py``) can verify that the committed file is up to date.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "docs" / "api.md"

#: Groups of public names, in the order they appear in the reference.
GROUPS: Sequence[Tuple[str, Sequence[str]]] = (
    ("Engines", ("GeneticAlgorithm", "IslandModel")),
    ("Configuration", ("GAConfig",)),
    (
        "Representations",
        ("Representation", "RealRepresentation", "IntegerRepresentation", "BinaryRepresentation", "PermutationRepresentation"),
    ),
    ("Results and data", ("RunResult", "Individual", "StopReason")),
    (
        "Choices",
        ("SelectionMethod", "CrossoverMethod", "MutationMethod", "ReplacementMethod", "EvaluationErrorPolicy"),
    ),
    ("Errors", ("GAEngineError", "ConfigurationError", "EvaluationError", "CheckpointError")),
    ("Utilities", ("enable_console_logging",)),
)

#: Public names of ``ga_engine.benchmarks`` documented in the reference.
BENCHMARKS = ("sphere", "rastrigin", "rosenbrock", "ackley", "zdt1", "onemax", "KnapsackProblem", "TspProblem")

SECTION_NAMES = ("Args", "Keyword Args", "Returns", "Raises", "Attributes", "Yields")
ROLE = re.compile(r":(?:class|func|meth|mod|attr|exc|data|doc):`~?([^`]+)`")
ITEM = re.compile(r"^(\*{0,2}[\w.]+)(?: \(([^)]*)\))?: (.*)$")


def clean_inline(text: str) -> str:
    """Replace Sphinx roles with Markdown code spans (keeping the last dotted component)."""
    return ROLE.sub(lambda m: f"`{m.group(1).split('.')[-1]}`", text)


def docstring_to_markdown(doc: str) -> str:
    """Convert a Google-style docstring into Markdown."""
    lines = doc.splitlines()
    out: List[str] = []
    section = ""
    bullets: List[str] = []
    index = 0

    def flush() -> None:
        nonlocal bullets
        if bullets:
            out.extend(bullets)
            out.append("")
            bullets = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if line and not line.startswith(" ") and stripped.endswith(":") and stripped[:-1] in SECTION_NAMES:
            flush()
            section = stripped[:-1]
            out.extend([f"**{section}**", ""])
            index += 1
            continue
        if section:
            if line and not line.startswith(" "):
                flush()
                section = ""
                continue
            if not stripped:
                index += 1
                continue
            body = line[4:] if line.startswith("    ") else line.strip()
            match = ITEM.match(body) if not body.startswith(" ") else None
            if match:
                name, kind, text = match.groups()
                label = f"`{name}`" + (f" ({clean_inline(kind)})" if kind else "")
                bullets.append(f"- {label}: {clean_inline(text)}")
            elif bullets:
                bullets[-1] += " " + clean_inline(body.strip())
            else:
                bullets.append(f"- {clean_inline(body.strip())}")
            index += 1
            continue
        if stripped.endswith("::"):  # literal block follows
            out.append(clean_inline(stripped[:-1]))
            out.append("")
            index += 1
            block: List[str] = []
            while index < len(lines) and (not lines[index].strip() or lines[index].startswith("    ")):
                block.append(lines[index][4:] if lines[index].startswith("    ") else "")
                index += 1
            while block and block[0] == "":
                block.pop(0)
            while block and block[-1] == "":
                block.pop()
            out.extend(["```python", *block, "```", ""])
            continue
        out.append(clean_inline(line))
        index += 1
    flush()
    return "\n".join(out).strip()


def render_annotation(annotation: Any) -> str:
    """Render an annotation without the quotes that ``from __future__ import annotations`` adds."""
    return annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)


def render_default(value: Any) -> str:
    """Render a default value; enum members become ``EnumName.MEMBER``."""
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    return repr(value)


def render_parameter(parameter: inspect.Parameter) -> str:
    """Render one parameter as ``name: annotation = default``."""
    text = parameter.name
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        text = "*" + text
    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
        text = "**" + text
    if parameter.annotation is not inspect.Parameter.empty:
        text += f": {render_annotation(parameter.annotation)}"
    if parameter.default is not inspect.Parameter.empty:
        text += f" = {render_default(parameter.default)}"
    return text


def format_signature(name: str, callable_: Callable[..., Any], skip_self: bool = True) -> str:
    """Return a readable (multi-line when long) signature, including the keyword-only marker."""
    signature = inspect.signature(callable_)
    parameters = [p for p in signature.parameters.values() if not (skip_self and p.name in ("self", "cls"))]
    rendered: List[str] = []
    keyword_marker_added = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters)
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not keyword_marker_added:
            rendered.append("*")
            keyword_marker_added = True
        rendered.append(render_parameter(parameter))
    return_part = ""
    if signature.return_annotation is not inspect.Signature.empty:
        return_part = f" -> {render_annotation(signature.return_annotation)}"
    single = f"{name}({', '.join(rendered)}){return_part}"
    if len(single) <= 88:
        return single
    body = "".join(f"    {item},\n" for item in rendered)
    return f"{name}(\n{body}){return_part}"


def public_methods(cls: type) -> List[Tuple[str, Any]]:
    """Public functions and properties defined on ``cls`` itself, in source order."""
    members = []
    for name, value in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if inspect.isfunction(value) or isinstance(value, property):
            members.append((name, value))
    return members


def render_class(cls: type, heading: str) -> List[str]:
    """Render one class (or enum, or dataclass) section."""
    lines = [f"### {heading} `{cls.__name__}`", ""]
    module = cls.__module__
    lines.append(f"*Defined in `{module}`.*")
    lines.append("")
    doc = inspect.getdoc(cls) or ""
    if enum.Enum in cls.__mro__:
        lines.append(docstring_to_markdown(doc))
        lines.extend(["", "| Member | Value |", "| --- | --- |"])
        lines.extend(f"| `{m.name}` | `{m.value!r}` |" for m in cls)
        lines.append("")
        return lines
    init = cls.__dict__.get("__init__")
    if dataclasses.is_dataclass(cls):
        init_sig = format_signature(cls.__name__, cls)
    elif init is not None:
        init_sig = format_signature(cls.__name__, init)
    else:
        init_sig = f"{cls.__name__}()"
    lines.extend(["```python", init_sig, "```", "", docstring_to_markdown(doc), ""])
    methods = public_methods(cls)
    properties = [(n, v) for n, v in methods if isinstance(v, property)]
    functions = [(n, v) for n, v in methods if not isinstance(v, property)]
    if properties:
        lines.extend(["**Properties**", ""])
        for name, prop in properties:
            summary = (inspect.getdoc(prop) or "").split("\n\n")[0].replace("\n", " ")
            lines.append(f"- `{name}`: {clean_inline(summary)}")
        lines.append("")
    for name, function in functions:
        lines.extend([f"#### `{cls.__name__}.{name}`", "", "```python", format_signature(name, function), "```", ""])
        function_doc = inspect.getdoc(function)
        if function_doc:
            lines.extend([docstring_to_markdown(function_doc), ""])
    return lines


def render_function(function: Callable[..., Any], name: str) -> List[str]:
    """Render one module-level function."""
    lines = [f"### `{name}`", "", "```python", format_signature(name, function, skip_self=False), "```", ""]
    doc = inspect.getdoc(function)
    if doc:
        lines.extend([docstring_to_markdown(doc), ""])
    return lines


def render_symbol(module: Any, name: str) -> List[str]:
    """Render a public symbol according to its kind."""
    symbol = getattr(module, name)
    if inspect.isclass(symbol):
        kind = "enum" if issubclass(symbol, enum.Enum) else ("exception" if issubclass(symbol, BaseException) else "class")
        return render_class(symbol, kind)
    return render_function(symbol, name)


def generate() -> str:
    """Return the complete Markdown text of the API reference."""
    sys.path.insert(0, str(PROJECT_ROOT))
    package = importlib.import_module("ga_engine")
    benchmarks = importlib.import_module("ga_engine.benchmarks")
    documented = {name for _, names in GROUPS for name in names}
    missing = sorted(set(package.__all__) - documented - {"__version__"})
    if missing:
        raise SystemExit(f"public names missing from GROUPS in {Path(__file__).name}: {missing}")

    lines = [
        "# API reference",
        "",
        "> Generated from the source by `tools/generate_api_docs.py`. Do not edit by hand;",
        "> run `python tools/generate_api_docs.py` after changing a docstring or signature.",
        "",
        "Everything listed here is importable from `ga_engine` (test problems from `ga_engine.benchmarks`).",
        "",
        "## Contents",
        "",
    ]
    for title, names in GROUPS:
        lines.append(f"- **{title}**: " + ", ".join(f"[`{n}`](#{anchor(package, n)})" for n in names))
    lines.append("- **Test problems** (`ga_engine.benchmarks`): " + ", ".join(f"[`{n}`](#{anchor(benchmarks, n)})" for n in BENCHMARKS))
    lines.append("")
    for title, names in GROUPS:
        lines.extend([f"## {title}", ""])
        for name in names:
            lines.extend(render_symbol(package, name))
    lines.extend(["## Test problems", "", "Importable from `ga_engine.benchmarks`. All are minimisation problems and are picklable.", ""])
    for name in BENCHMARKS:
        lines.extend(render_symbol(benchmarks, name))
    return "\n".join(lines).rstrip() + "\n"


def anchor(module: Any, name: str) -> str:
    """GitHub-style anchor for the heading of ``name``."""
    symbol = getattr(module, name)
    if inspect.isclass(symbol):
        kind = "enum" if issubclass(symbol, enum.Enum) else ("exception" if issubclass(symbol, BaseException) else "class")
        return f"{kind}-{name.lower()}"
    return name.lower()


def main(argv: Sequence[str]) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if docs/api.md is out of date")
    arguments = parser.parse_args(argv)
    text = generate()
    if arguments.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print("docs/api.md is out of date; run: python tools/generate_api_docs.py", file=sys.stderr)
            return 1
        print("docs/api.md is up to date")
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(PROJECT_ROOT)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
