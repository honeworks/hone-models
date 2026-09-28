# 0001: Initial design

## Status

`implemented in 0.1.0`

## Context

hone-models comes out of OneShotStudio, a local pipeline that generates song ideas, lyrics, music and
music videos with Ollama models and ComfyUI on one 8 GB GPU. Its model code (an Ollama client, model-tag
lookups copied across modules, hand-written JSON extraction and recovery) kept failing in the same ways:
long JSON prompts silently truncated by Ollama's default context window, JSON wrapped in fences or cut
off, `format` passed where Ollama ignores it, thinking models returning an empty `content`, a text-only
model used as the vision judge, fixed 120 s timeouts on large models, and GPU contention between Ollama,
ComfyUI and in-process torch models. Nobody could say afterwards which prompt produced an output.
The early research is in [../history/0000-research.md](../history/0000-research.md).

## Problem

There was no small library that treats local and hosted models as equals, refuses calls that cannot
work before sending them, makes structured output fail loudly instead of silently, shares one GPU
between model calls and other GPU work, and records every call in enough detail to replay it with one
thing changed.

## Options

1. **Use LiteLLM directly.** Excellent hosted-provider coverage and a large model registry, but no local
   lifecycle, no GPU scheduling, no decision questions, and no context budget for local models.
2. **Build on Instructor or PydanticAI** for structured output. Good validation and retry loops, but no
   context budget, truncation detection or thinking-model handling, and a larger dependency than the
   pipeline needs.
3. **Adopt LangChain.** Covers much of the ground, but through a large abstraction surface.
4. **A small package of its own** that talks to Ollama and OpenAI-compatible servers directly, uses
   LiteLLM as an optional provider for hosted models, and implements its own short structured-output
   pipeline.

## Decision

Option 4. The design is described in full in [../current.md](../current.md); its main choices:

- **One client per call type** (`mk.text`, `mk.decision`, `mk.embedder`), bound to one model and resolved
  through a **TOML registry** of models and capabilities, with ad-hoc `provider:model` ids for anything
  not registered.
- **Capabilities are checked before calling.** An undeclared capability is *unknown*, not false: checks
  refuse only what the registry declares impossible, so ad-hoc hosted models stay usable (D-002).
- **Structured output pipeline of its own**: budget, constrain, parse, validate, retry with the error,
  repair; the result reports the path taken. The schema is always put in the system message and the
  provider constraint is added when supported (D-006). The output budget is always sent, so truncation
  is reported as finish reason `length` instead of happening silently (D-007).
- **Answer problems are values, call problems are typed exceptions.**
- **Decision questions** as a call type with one normalized answer shape; emulated on LLMs in one
  schema-constrained call (calibrated from logprobs when available), native for Jev behind a small,
  fixture-tested mapper because the Jev API is documented only in public articles (D-019).
- **Embeddings** were planned for a second version in the research; they were pulled into 0.1 because
  similarity and clustering code needed them from the start.
- **GPU leases** through a flock-guarded JSON ledger with NVML accounting, usable by non-LLM code.
  Model calls do not take a lease on their own (D-009, awaiting owner review).
- **Local lifecycle** (`mk.session`, `mk.unload`) for Ollama only, ported from OneShotStudio's pattern:
  use a running server, else start one with a clean environment and stop only what was started (D-012).
- **Records** as OpenTelemetry-shaped spans in a local SQLite file, with GenAI attribute names, prompt
  sections as character spans, a content-capture switch and secret stripping. Chat spans also carry all
  request params and the schema, so a call can be rebuilt exactly (D-008).
- **Replay** with overrides (a prompt section, the model, params or messages) was also pulled forward
  from the second version, because comparing a call with one thing changed is the most direct way to
  test a fix.
- **Small core**: `pydantic` and `httpx`; the CLI, NVML readings and LiteLLM are extras (D-005).
- **Plain shapes, no imports of other packages**: code that expects a text client, decision client,
  embedder or GPU lease can take hone-models' objects directly, and entry points let tools load them by
  name.

## Consequences

- Better: failures that used to be silent (truncation, empty replies, invalid JSON, wrong model for the
  job) are now either refused before the call or reported on the result; every call can be inspected
  and replayed; GPU users take turns.
- Costs: hone-models maintains its own provider code for Ollama and OpenAI-compatible servers and its own
  structured-output pipeline instead of reusing a larger library.
- Harder: capabilities for ad-hoc models are often unknown, so some errors still surface as provider
  errors; the Jev mapping must be checked against the real API.

## Migration and compatibility

None: this is the first version. For code that called Ollama directly, the migration is: replace chat
calls and model-tag lookups with `mk.text(id)` and a registry file; replace hand-written JSON extraction
with `complete(..., schema=Model)`; replace hard-coded thinking or vision model lists with registry
capabilities and `require=`; replace hand-written server sessions with `mk.session("ollama")` and
`mk.unload`.
