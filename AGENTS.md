# AGENTS.md: hone-models

Notes for contributors who use AI coding tools (Claude Code, Codex, Cursor and others).

## What this is
`hone-models` (import `hone_models`): one interface for local and hosted AI models, with structured
output, GPU scheduling and full call records. Part of the honeworks family of Python packages.

Read before changing code:
- [design/current.md](design/current.md): the design, its rules and the acceptance cases.
- [design/changes/](design/changes/) and [design/decisions.md](design/decisions.md): why it is so.
- [CONTRIBUTING.md](CONTRIBUTING.md): conventions, tests, the simplicity limits.

## Commands
```bash
uv sync --all-extras                 # install everything (dev included)
scripts/check.sh                     # all quality gates; "green" means this passes
uv run pytest                        # default suite (fast, offline, deterministic)
uv run pytest tests/e2e              # acceptance cases only
scripts/gpu-lock.sh uv run pytest -m gpu      # real-model tests, under the machine-wide GPU lock
uv run ruff check . && uv run ruff format .
uv run pyright
```

## Layout
```
src/hone_models/       public API in __init__.py (explicit __all__)
  text.py structured.py budget.py prompt.py   chat client, structured output, context budget, prompts
  decision/            decision questions: emulation on LLMs, question / answer shapes
  embeddings.py        embedder
  speech.py            text to speech (kokoro: extra `speech`; chatterbox: extra `expressive`)
  providers/           one module per provider (Ollama, OpenAI-compatible, Jev, LiteLLM extra)
  registry.py data/    model registry and packaged defaults
  gpu.py runtime/      GPU leases; Ollama sessions and unload
  records.py _tracing.py replay.py calls.py cli.py   spans, trace context, replay, call queries, CLI
  testing/             FakeOllama, FakeSpeech and the record-sink contract check
tests/unit|contract|integration|e2e|gpu
docs/  examples/  design/  scripts/
```

## Rules
1. **Keep it simple:** the simplest code that meets the acceptance cases; limits in CONTRIBUTING.md.
2. **Useful alone:** the core never imports another honeworks package or an optional extra.
3. **Explicit failure:** answer problems on the result (`error`), call problems as typed exceptions;
   `None`, never `0`, for a missing value.
4. **Records:** spans exactly as in design/current.md §8; never record secrets.
5. **Tests first** for public behaviour; each acceptance case has `tests/e2e/test_ac<N>_*`; README,
   docs snippets and `examples/` are executed by tests.
6. **Green means `scripts/check.sh` passes.** Never weaken a test to make it pass.
7. **Shared GPU:** real-model runs go through `scripts/gpu-lock.sh`; never start or stop the Ollama
   service; unload models you load.
8. **Design changes** get a record in `design/changes/`; small choices go into `design/decisions.md`.
9. **Git:** Conventional Commits, small commits. Never push, tag or publish without the maintainer.
