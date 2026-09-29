# hone-models

[![CI](https://github.com/honeworks/hone-models/actions/workflows/ci.yml/badge.svg)](https://github.com/honeworks/hone-models/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/honeworks/hone-models/blob/main/LICENSE)

One interface for local and hosted AI models, with structured output, GPU scheduling and full call records.

Part of **[honeworks](https://github.com/honeworks)**: small, standalone tools for reliable generative-AI
workflows. Works on its own; works better with its siblings.

**Why:** a generative-AI workflow that runs on your own machine calls several models (a local LLM, a
vision model, an embedder, a judge, a voice) that share one GPU, answer in different shapes and fail in
different ways. hone-models gives them one small interface that checks a request fits the model before
sending it, says exactly how a structured answer was obtained, keeps models from fighting over VRAM, and
records every call so you can see and replay what happened.

- **Local and hosted as equals:** Ollama, any OpenAI-compatible server (llama.cpp, vLLM, LM Studio,
  OpenAI) and, with the `litellm` extra, any model LiteLLM knows.
- **Checked before calling:** vision, context size, thinking and JSON-schema support come from a model
  registry, so a prompt that does not fit raises `ContextOverflow` instead of being silently truncated.
- **Structured output that says what happened:** constrain, parse, validate, retry, repair; the result
  tells you which (`structured_path`) and never hides a truncated reply.
- **Decision questions:** yes/no, choice and score questions answered by a decision model (Jev) or
  emulated on any LLM, with calibrated probabilities from token logprobs when available.
- **Embeddings, GPU leases and local model lifecycle** (`mk.gpu.lease`, `mk.session`, `mk.unload`), and
  **machine state**: which models every server holds, partly on the CPU or not, and unloading all but the
  needed ones without breaking another run (`mk.machine.snapshot()`, `prepare()`, `load()`).
- **Local text to speech** (`mk.speech("kokoro-82m")`, extra `speech`): long narration into one WAV file;
  expressive narration with an emotion and intensity per paragraph (`mk.speech("chatterbox")`, extra
  `expressive`).
- **Images, music and video** (`mk.image`, `mk.music`, `mk.video`) through ComfyUI workflows, one registry
  entry per model: named inputs mapped onto workflow nodes, files uploaded once by hash, a GPU lease per
  call or session, failed jobs as results with an `error_kind`
  ([docs/generation.md](https://github.com/honeworks/hone-models/blob/main/docs/generation.md)).
- **A catalog of models with guides:** every model we use or may use, installed or not, with its licence,
  what installs it (`hone-models models install <id>` prints the commands) and a guide saying what it can
  take (`mk.guide(id)`, `mk.select({"features": [...]})`); lyrics and prompt phrases written once and
  converted per model
  ([docs/models-and-guides.md](https://github.com/honeworks/hone-models/blob/main/docs/models-and-guides.md)).
- **Every call recorded** as an OpenTelemetry-style span in a local SQLite file, prompt sections included,
  and replayable with changes.

## Install
```bash
pip install hone-models        # or: uv add hone-models
```
Until it is on PyPI, install it from GitHub:
```bash
pip install "git+https://github.com/honeworks/hone-models"
pip install "hone-models[cli] @ git+https://github.com/honeworks/hone-models"   # with an extra
```

The core needs only pydantic and httpx. Optional extras:

| Extra | Adds | Needs |
|---|---|---|
| `cli` | the `hone-models` command (calls, models, GPU status) | typer, rich |
| `gpu` | NVML memory readings for GPU leases | an NVIDIA driver |
| `litellm` | hosted models through LiteLLM | the provider's API key in an environment variable |
| `speech` | local text to speech with Kokoro-82M | Python < 3.13, torch; spaCy's English model ([docs/speech.md](https://github.com/honeworks/hone-models/blob/main/docs/speech.md)) |
| `expressive` | expressive speech with Chatterbox (emotion and intensity) | Python < 3.13, torch; dependency overrides ([docs/speech.md](https://github.com/honeworks/hone-models/blob/main/docs/speech.md)) |

`jev` is an empty extra: the Jev decision provider needs nothing beyond the core (it calls the API over httpx).

## Quickstart
Needs a local [Ollama](https://ollama.com) with Gemma 4 12B (`ollama pull gemma4:12b`), or change the
model id. Point the registry id `gemma4-12b` at the tag you pulled in `~/.config/hone/models.toml`:
`[models."gemma4-12b"]` with `model = "gemma4:12b"` (see
[the registry docs](https://github.com/honeworks/hone-models/blob/main/docs/registry.md)).

```python
import hone_models as mk
from pydantic import BaseModel

llm = mk.text("gemma4-12b")  # a registry id; "ollama:llama3.2:1b" works ad hoc
r = llm.complete([{"role": "user", "content": "Write a haiku about rain."}])
print(r.text, r.usage, r.finish_reason)


class Verdict(BaseModel):
    score: int
    rationale: str


r = llm.complete(
    [{"role": "user", "content": "Rate this pitch 1-5: a shanty about spreadsheets."}], schema=Verdict
)
print(r.parsed, r.structured_path)  # a Verdict (or r.error set), and how it was obtained
```

Each call is recorded in `.hone/models/spans.db`; `hone-models calls list` shows them.

## Use it with the rest of honeworks
hone-models is the default implementation of the ports the other honeworks packages own, implemented
structurally (it imports none of them):

| Port | hone-models | Entry point group |
|---|---|---|
| `TextClient` | `mk.text(...)` | `hone.text_clients` |
| `DecisionClient` | `mk.decision(...)` | `hone.decision_clients` |
| `Embedder` | `mk.embedder(...)` | `hone.embedders` |
| `GpuLease` | `mk.gpu.GPU` / `mk.gpu.lease` | `hone.gpu_leases` |
| `Replayer` | `mk.replay.Replayer()` | `hone.replayers` (no argument) |
| `MachineProbe` (hone-select) | `mk.machine.Machine()` / `mk.machine.MACHINE` | `hone.machine_probes` (no argument) |

For example, `hone-select` judges and `hone-taste` panels take `mk.decision("gemma4-12b")` as their
decision client, and `hone-flow` steps take `mk.gpu.GPU` as their GPU lease. Spans follow the shared
honeworks records format, so `hone-lens` reads them next to the other packages' records and replays
calls through `mk.replay.Replayer`.

## Documentation
- [docs/](https://github.com/honeworks/hone-models/blob/main/docs/README.md): concepts and guides (registry, structured output, prompts, decisions,
  embeddings, GPU and sessions, records and replay, CLI, testing without models).
- [examples/](https://github.com/honeworks/hone-models/blob/main/examples/README.md): one runnable, offline example per concept (quickstart, registry,
  structured output, context budget, providers, decisions, retries, records, GPU leases, testing), each
  explaining what it shows, how and why; the test suite runs them all.
- [design/](https://github.com/honeworks/hone-models/blob/main/design/README.md): why the package exists, the [current design](https://github.com/honeworks/hone-models/blob/main/design/current.md), and
  the [change records](https://github.com/honeworks/hone-models/tree/main/design/changes) and [decisions](https://github.com/honeworks/hone-models/blob/main/design/decisions.md) behind it.
- [CONTRIBUTING.md](https://github.com/honeworks/hone-models/blob/main/CONTRIBUTING.md): conventions, tests and how design changes are made.

## How this was built
hone-models was specified by a human and built by AI coding agents (Claude) working against written
specifications and acceptance tests; a human reviewed the decisions they made, and commits written with
AI carry a `Co-Authored-By` line. Every design change, with what was found, what was decided and why, is
in [design/changes/](https://github.com/honeworks/hone-models/tree/main/design/changes); the smaller implementation choices,
including those still awaiting the owner's review, are in [design/decisions.md](https://github.com/honeworks/hone-models/blob/main/design/decisions.md).

## Status
Alpha, version 0.1.0. The API follows [design/current.md](https://github.com/honeworks/hone-models/blob/main/design/current.md)
and may still change before 1.0; changes are listed in
[CHANGELOG.md](https://github.com/honeworks/hone-models/blob/main/CHANGELOG.md). Python 3.11 to 3.13
(the speech extras: 3.11 and 3.12).

## License
Apache-2.0. Copyright 2026 Bahman Shadmehr. Third-party material shipped in the package (the voice
clips) and the licences of the optional extras' dependencies are listed in
[THIRD_PARTY_NOTICES.md](https://github.com/honeworks/hone-models/blob/main/THIRD_PARTY_NOTICES.md).
