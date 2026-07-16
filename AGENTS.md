# AGENTS.md

Beamline optimization with machine learning (`blop`). Python package using
`pixi` for environment/task management, `ruff` for lint/format, `pyright` for
type checking, and `pytest` for tests.

## Environment

- All commands run inside a pixi environment. Prefix commands with
  `pixi run -e <env>`. Common envs: `dev` (GPU torch), `dev-cpu` (CPU torch),
  `docs`, and per-Python `py311-cpu`..`py313-cpu`.
- Editable installs are configured in `pixi.toml`; do not `pip install`
  manually. Re-sync with `pixi install -e dev-cpu` if dependencies change.
- `pyproject.toml` is the single source of truth for runtime/dev
  dependencies. `pixi.toml` should pull them in via the editable install
  (`blop = { path = ".", editable = true, extras = [...] }`) rather than
  redeclare them; only platform-specific binary deps (e.g. `graphviz`)
  belong directly in `pixi.toml`.

## Build / Lint / Test commands

- Install / sync env: `pixi install -e dev-cpu`
- Run all pre-commit hooks (ruff format + ruff lint + pyright + nbstripout +
  mdformat): `pixi run -e dev check` (alias for `pre-commit run --all-files`).
- Format only: `pixi run -e dev ruff format`
- Lint only (with autofix): `pixi run -e dev ruff check --fix`
- Type check: `pixi run -e dev pyright`
- Run full test suite: `pixi run -e dev tests`
  (alias for `pytest src/blop/tests`).
- Run a single test file: `pixi run -e dev pytest src/blop/tests/test_utils.py`
- Run a single test by node id:
  `pixi run -e dev pytest src/blop/tests/test_utils.py::test_name`
- Run tests matching a keyword: `pixi run -e dev pytest -k "queueserver"`
- Run tests by marker (defined in `pytest.ini`: `shadow`, `srw`, `test_func`):
  `pixi run -e dev pytest -m test_func`
- Coverage (config in `.coveragerc`): `pixi run -e dev pytest --cov=blop`
- Build docs: `pixi run -e docs build-docs` (or `build-docs-no-exec` to skip
  notebook execution). Tutorials live under `docs/source/tutorials/`.

## Repository layout

- `src/blop/` — library code. Submodules: `ax/`, `bayesian/`, `callbacks/`,
  plus `plans.py`, `plan_stubs.py`, `protocols.py`, `queueserver.py`,
  `utils.py`.
- `src/blop/tests/` — pytest tests, mirrors package layout. `conftest.py`
  holds shared fixtures.
- `sim/` — separate `blop-sim` editable package used by tests/docs.
- `docs/`, `examples/` — Sphinx docs and example scripts/notebooks.

## Code style

Enforced by `ruff` (config in `pyproject.toml [tool.ruff]`) and `pyright`.

- Target Python: `>=3.11`. Use modern syntax (`X | Y` unions, `list[T]`,
  `dict[K, V]`, `match`/`case`) — `pyupgrade` (`UP`) is enabled.
- Line length: **125** chars. Indent with 4 spaces. Use double quotes
  (ruff format default).
- Imports: sorted by ruff `I` (isort). Order: stdlib, third-party, first-party.
  Use **relative imports within `blop`** (e.g. `from .protocols import ...`,
  as in `src/blop/utils.py`). Avoid private/underscore imports across modules
  (`PLC2701` enabled in preview).
- Active rule sets: `B`, `C4`, `E`, `F`, `W`, `I`, `UP`, `SLF`, `PLC2701`,
  `LOG015`, `S101`, `D` (numpy convention). Currently ignored repo-wide:
  `D` (docstring rules — still write numpy-style docstrings on new public
  APIs), `SLF001`, `B901`. In `src/blop/tests/**` `S101`, `SLF001`, `D` are
  ignored — `assert` is fine in tests but not in library code.
- Logging: never call the root logger (`LOG015`). Use a module logger:
  `logger = logging.getLogger(__name__)`.

## Typing

- All new public functions/methods must have full type annotations. `pyright`
  runs in pre-commit; do not introduce new type errors.
- Prefer concrete types from `numpy.typing` (`ArrayLike`, `NDArray`),
  `collections.abc` (`Sequence`, `Iterable`, `Mapping`), and project
  protocols in `src/blop/protocols.py`. Use `Any` only as a last resort.
- Use `from __future__ import annotations` only when needed for forward
  references; existing code generally does not.
- Excluded from pyright (do not introduce new code there without cleaning up
  types if possible): `sim/`, `src/blop/tests/`, `src/blop/bayesian/`,
  `src/blop/ax/qserver_agent.py`.

## Naming

- `snake_case` for functions, methods, variables, modules.
- `PascalCase` for classes and `Enum` types (e.g. `Source(str, Enum)` in
  `src/blop/utils.py`).
- `UPPER_SNAKE_CASE` for module-level constants (e.g. `ID_KEY`).
- Leading underscore (`_name`) for module-private and internal attributes.
  Don't access `_private` members across module boundaries.
- Test files: `test_*.py`; test functions: `test_*`.

## Docstrings & comments

- Numpy-style docstrings (`[tool.ruff.lint.pydocstyle] convention = "numpy"`).
  See `InferredReadable` in `src/blop/utils.py` for a canonical example
  (Parameters section with `name : type` lines).
- Public classes, functions, and modules should have a one-line summary plus
  Parameters / Returns / Raises sections where applicable. Keep comments
  focused on *why*, not *what*.

## Error handling

- Raise specific built-in or library exceptions (`ValueError`, `TypeError`,
  `RuntimeError`, `KeyError`); avoid bare `Exception`.
- Never use bare `except:`; catch the narrowest exception that makes sense.
  Re-raise with `raise ... from err` to preserve context.
- Validate inputs at public API boundaries with clear messages including the
  offending value.
- Don't swallow exceptions silently. If logging and continuing is intended,
  log at `warning`/`error` and add a comment explaining why.

## Tests

- Framework: `pytest`. Place new tests under `src/blop/tests/` mirroring the
  module they cover. Reuse fixtures from `src/blop/tests/conftest.py`.
- Tests may use `assert` and access private members (`S101`, `SLF001`
  ignored under `src/blop/tests/**`).
- Keep tests deterministic: seed RNGs (`numpy`, `torch`) explicitly.
- **Tests must run fast.** Avoid real I/O, network calls, hardware, long
  Bayesian-optimization loops, or heavy `torch` training in unit tests. Mock
  expensive collaborators using `unittest.mock` (`MagicMock`, `patch`,
  `AsyncMock`); reuse the mock fixtures already
  defined in `src/blop/tests/conftest.py` and submodule `conftest.py` files
  before adding new ones.

## Git / PR workflow

- `pre-commit` is mandatory; install once with
  `pixi run -e dev pre-commit install`. CI runs the same hooks plus the
  pyright and pytest jobs defined in `.github/workflows/`.
- Notebooks under `docs/` are stripped by `nbstripout`; do not commit
  notebook outputs. Markdown is reformatted by `mdformat`.
- Don't commit large files (enforced by `check-added-large-files`).
- Never commit secrets or local config (`.opencode/`, `.pixi/`,
  `.mypy_cache/`, etc. are gitignored).
