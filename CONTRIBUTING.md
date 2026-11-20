# Contributing to ga-engine

Thank you for helping. This guide explains how to set up, what the project's
standards are, and how to get a change merged.

## Setup

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1      macOS / Linux: source .venv/bin/activate
pip install -e . mypy
```

## The checks every change must pass

```bash
python -m unittest discover -s tests -t .     # tests, including executable documentation
mypy                                          # strict static typing
python tools/generate_api_docs.py --check     # docs/api.md matches the source
```

If the last check fails after you changed a docstring or signature, run
`python tools/generate_api_docs.py` and commit the result.

## Coding standards

- **Separation of concerns.** One module, one responsibility. If a file mixes
  concerns or keeps growing, split it. `engine.py` should stay an orchestrator:
  put new algorithmic logic in a focused component and wire it in.
- **Documentation in the code.** Every public module, class and function has a
  docstring (Google style: `Args`, `Returns`, `Raises`). Public APIs are fully
  type-annotated.
- **Explicit error handling.** Never swallow exceptions. A broad
  `except Exception` is allowed only if it re-raises or sits at a documented
  boundary (a test enforces this). Raise the specific error classes from
  `ga_engine.errors`, with a message that says what to do.
- **Modern, minimal Python.** Dataclasses, enums, `pathlib`, context managers.
  Prefer the standard library. The only runtime dependency is NumPy; adding
  another needs a written justification in the pull request.
- **No placeholders.** No stubs, no `pass`-bodied "TODO" implementations.
- **Fail early.** Validate settings at construction (see `config.py`) rather than
  during a run.

## Tests

Tests use only the standard library (`unittest`), one module per area in
`tests/`. For every behaviour change add a test that fails without the change.

- Test randomised behaviour through fixed seeds, and statistical behaviour
  through tolerances that are not flaky. A seed makes a run reproducible for a given
  NumPy version only; if a test fails on just one NumPy version, check whether it
  depends on an exact random stream and loosen it (the documentation's `output`
  blocks should likewise avoid exact floating-point digits).
- Anything involving worker processes needs picklable, module-level functions
  (see `tests/helpers.py` and `ga_engine/benchmarks.py`).
- Checkpoint changes need a round-trip test for both `.json` and `.npz`, and, if
  the state changes, a resume-equals-uninterrupted test.

## Documentation

Docs are tested. In any Markdown file:

- a fenced `python` block is **executed**; start it with `# doctest: skip` if it
  cannot run (for example, because it needs matplotlib);
- a fenced `output` block placed right after a Python block must equal what that
  block prints;
- relative links and `#anchors` must resolve;
- every constructor and `run()` parameter must appear in
  `docs/configuration.md` and in the tables of `LLM_GUIDE.md` with the correct default.

Update `CHANGELOG.md` for user-visible changes.

## Adding things

| To add... | Do this |
| --- | --- |
| A genome type | Subclass `Representation` (see [docs/user-guide.md](docs/user-guide.md#15-writing-your-own-representation)); export it from `ga_engine/representations/__init__.py` and `ga_engine/__init__.py`; add it to `GROUPS` in `tools/generate_api_docs.py`; document it in the operators doc and in `LLM_GUIDE.md` |
| An operator | Add the enum member; support it in the representation (including `SUPPORTED_*`, `*_PARAMS`); document it; test validity, bounds and parameter validation |
| A setting | Add it to `GeneticAlgorithm.__init__` and `config.build_config` with validation; document it in `docs/configuration.md` and `LLM_GUIDE.md` |

## Pull requests

- Keep changes focused; explain *why* in the description.
- Make sure the three checks above pass.
- Say what you ran and what you could not verify (for example, a platform you
  do not have).
