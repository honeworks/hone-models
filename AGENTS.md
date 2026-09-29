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
  media.py             images, music, video (`MediaClient`); _media_files.py measures outputs
  transcribe.py        transcription with word timestamps (faster_whisper: extra `transcribe`)
  providers/           one module per provider (Ollama, OpenAI-compatible, Jev, LiteLLM extra, ComfyUI, command);
                       openai_media.py: OpenAI-compatible images and video
  registry.py data/    model registry; the packaged catalog in data/models/<kind>.toml; _registry_shapes.py:
                       an entry's tables; _registry_select.py: requirements and orderings
  guide.py formats.py catalog.py   model guides; lyrics formats and prompt inputs; installed / install
  gpu.py runtime/      GPU leases; Ollama and ComfyUI sessions and unload; _comfyui_loaded.py
  machine.py           machine state: snapshot, prepare, load (readers in _machine_read.py)
  records.py _tracing.py replay.py calls.py cli.py   spans, trace context, replay, call queries, CLI
  testing/             FakeOllama, FakeSpeech, FakeComfyUI, FakeMedia, FakeTranscriber, the record-sink check
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
9. **Git:** Conventional Commits, small commits. Push a branch and open a pull request only when the user asks ("push", "ship it"). Inside that pull
request's flow, reviewing it on GitHub and pushing fixes the user asked for need no new request.
Never push to `main`, tag or publish unless the maintainer asks.

## Workflow and tooling

One branch per task; a change record before a design change; tests first; the docs updated with the
code (the table in [`.claude/skills/sync-docs/SKILL.md`](.claude/skills/sync-docs/SKILL.md));
`scripts/check.sh` green; a pull request from
[`.github/pull_request_template.md`](.github/pull_request_template.md), reviewed by claude[bot]. Claude
Code users get this flow as skills, reviewer agents and hooks in [`.claude/`](.claude/); see
[`CLAUDE.md`](CLAUDE.md). Other tools: the skills are plain Markdown and can be followed as they are.
