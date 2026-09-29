# Contributing to hone-models

Thanks for helping. This page holds the conventions the code follows; the design itself is in
[design/current.md](design/current.md).

## Setup and the quality gates

```bash
uv sync --all-extras                 # install everything, dev tools included
scripts/check.sh                     # all quality gates: the definition of "green"
uv run pytest                        # default suite: fast, offline, deterministic
uv run pytest tests/e2e              # acceptance cases only
uv run ruff check . && uv run ruff format .
uv run pyright                       # strict for src/
```

`scripts/check.sh` runs lint, format check, pyright, the default test suite with at least 90 % branch
coverage on `src/`, `uv build`, and a smoke test that installs the built wheel alone in a fresh venv and
runs the README quickstart. A change is ready when it passes. Never weaken a test to make it pass.

The `dev` dependency group (installed by `uv sync` by default, never published) includes the packages
whose contract checks the tests run (`hone-flow`, `hone-lens`). Until they are on PyPI,
`[tool.uv.sources]` in `pyproject.toml` installs them from their GitHub repositories. hone-select's
checkers are not a dependency (hone-select depends on hone-models, so that would be a cycle):
`tests/select_contracts.py` imports them from hone-select when it is installed and otherwise runs the
same checks from a copy. To work on the packages side by side, point the sources at local checkouts,
e.g. `hone-flow = { path = "../hone-flow", editable = true }` (and do not commit that change).

## Keep it simple

Simple, easy to understand, maintainable code comes first after correctness. Build the simplest thing
that meets the documented guarantees:

- No speculative generality: add an abstraction only for two real uses today, or for an extension point
  the design names (the shapes in [design/current.md §10](design/current.md#10-shapes-other-code-can-rely-on),
  the registry, entry points, user-supplied functions).
- Plain Python: functions and small dataclasses before classes; composition before inheritance; a dict
  of functions or a `match` instead of a class hierarchy for variants.
- Flat and explicit: short call chains, no metaclasses, import hooks, monkey-patching or deep decorator
  stacks; no layers that only forward calls.
- One obvious way to configure and call each thing.
- Names over comments; comments say *why*.
- Delete dead code, unused parameters and single-caller helper layers.

| Limit | Value |
|---|---|
| Cyclomatic complexity per function | ≤ 10 (ruff `C901`) |
| Function length | ~40 lines |
| Module length | ~300 lines |
| Parameters per function | ≤ 6 |
| Core dependencies | standard library + pydantic + httpx; anything else needs a reason in `design/decisions.md` |

## Code conventions

- Python ≥ 3.11; `uv` for everything; `hatchling`, `src/` layout; ruff (line length 110); pyright strict
  for `src/`, basic for `tests/`; `py.typed` shipped.
- The public API is `hone_models.__all__`, plus `hone_models.testing` and `hone_models.calls`. Public
  functions and classes have docstrings.
- The core never imports an optional extra (LiteLLM, NVML, typer / rich) or another honeworks package;
  `tests/unit/test_import_boundaries.py` checks it.
- **Explicit failure:** never swallow errors. Problems with a model's answer are reported on the result
  (`error`); problems with the call raise typed exceptions from `hone_models.errors`, with messages that
  say what happened and what to do. A missing value is `None`, never `0`.
- **Records:** every model call emits a span as described in
  [design/current.md §8](design/current.md#8-records-and-replay). Never record secrets.
- **Determinism:** explicit seeds; never the built-in `hash()` for ids (use `hashlib`).
- Time is `datetime.now(UTC)`, stored as ISO-8601 UTC; paths are `pathlib.Path`; logging goes through
  `logging.getLogger("hone_models")`, never `print` in library code.
- Secrets come only from environment variables.
- CLI: `hone-models <group> <verb>`, `--json` for machine-readable output, exit codes 0 / 1 / 2.

## Tests

| Suite | Folder | Runs by default | Purpose |
|---|---|---|---|
| Unit | `tests/unit/` | yes | each module, edge cases, error paths |
| Contract | `tests/contract/` | yes | the record-sink contract |
| Integration | `tests/integration/` | yes | real local resources without a GPU: SQLite, subprocesses, local HTTP servers |
| Acceptance | `tests/e2e/` | yes | the acceptance cases in [design/current.md §11](design/current.md#11-guarantees-acceptance-cases), through the public API and CLI |
| Real models | `tests/gpu/` | no | the `[real]` acceptance cases against a local Ollama and GPU |

- Every acceptance case has a test in `tests/e2e/` named `test_ac<N>_<slug>`, using only the public API,
  that fails if the feature is removed.
- The README quickstart, every ```` ```python ```` block in `docs/` and every file in `examples/` are
  executed by tests. Examples follow [design/current.md §12](design/current.md#12-examples).
- HTTP fixtures live in `tests/fixtures/http/` as small, hand-checked JSON files, including failure
  shapes (fenced JSON, truncated JSON, thinking-only replies, 429s).
- Tests must not start or stop a real Ollama service; session start / stop is tested against a fake
  executable.

### Real-model tests and `scripts/gpu-lock.sh`

Real-model tests are marked `gpu` and run through `scripts/gpu-lock.sh`, which holds a machine-wide
`flock` (default `/tmp/honeworks-gpu.lock`, override with `HONE_GPU_LOCK`) so that only one test run
uses the GPU at a time:

```bash
scripts/gpu-lock.sh uv run pytest -m gpu
```

They skip with a clear reason when Ollama or a model is missing, and unload the models they loaded.
Model choice: `HONE_TEST_OLLAMA_URL`, `HONE_TEST_TEXT_MODEL` (default `gemma4-12b:latest`, the packaged
registry's tag; set it to `gemma4:12b` if you pulled that from the Ollama library),
`HONE_TEST_THINKING_MODEL` (`deepseek-r1:8b`), `HONE_TEST_VISION_MODEL` (`qwen2.5vl:7b`),
`HONE_TEST_EMBED_MODEL` (`nomic-embed-text:latest`).

## Changing the design

Design changes are written down before they are built:

1. Write `design/changes/NNNN-<short-name>.md` with status `proposed` and the sections Status, Context,
   Problem, Options, Decision, Consequences, and Migration and compatibility.
2. The maintainer reviews it; the status becomes `accepted` (or `rejected`).
3. Implement it from [`design/current.md`](design/current.md) and the accepted change records, tests
   first.
4. Update `design/current.md`, set the record to `implemented in <version>`, and add a `CHANGELOG.md`
   entry that links to it.

Smaller implementation choices that need no change record go into
[`design/decisions.md`](design/decisions.md).

## Pull requests

- One branch per change, named `<type>/<short-name>` after the Conventional Commit types (`feat/ftp-storage`).
- Fill in [`.github/pull_request_template.md`](.github/pull_request_template.md): what, why (issue and
  change record), how it was tested, which docs changed, and the end of the `scripts/check.sh` output.
- claude[bot] reviews every pull request, with inline comments and suggested changes
  ([`.github/workflows/claude-review.yml`](.github/workflows/claude-review.yml)); on a pull request from a
  fork, the maintainer starts it with a `@claude review` comment. Answer each thread: agree and fix,
  disagree with a reason, or ask. A thread is resolved when it is fixed or decided.
- `main` accepts changes only through pull requests, with CI green and every review thread resolved.
- The code owners in [`.github/CODEOWNERS`](.github/CODEOWNERS) are asked to review automatically.
- The maintainer merges.

## Commits

Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `build:`, `ci:`),
subject ≤ 72 characters, a body that explains why, one logical change per commit, `scripts/check.sh`
green at every commit. Commits written with an AI tool end with a `Co-Authored-By` line naming it. Never
commit secrets, `.hone/`, model weights or large binaries.

Contributors using AI coding tools will find a short brief for them in [`AGENTS.md`](AGENTS.md); Claude
Code users also get the whole workflow as skills, reviewer agents and hooks in [`.claude/`](.claude/).
