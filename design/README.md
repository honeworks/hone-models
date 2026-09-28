# Why hone-models exists

> One way to call any model, local or hosted, with structured output that holds and GPUs that don't
> fall over, and every call recorded.

## The problem

Calling a model is one line. Calling models *reliably* inside a real project is not. hone-models was
extracted from OneShotStudio, a local pipeline that generates song ideas, lyrics, music and videos with
Ollama models on an 8 GB GPU. That project hit the same failures again and again:

| What went wrong | Why it hurt |
|---|---|
| Ollama's default context window (2048 tokens) silently cut a long JSON prompt | the model answered a question it never fully saw, and nothing reported it |
| JSON came back wrapped in markdown fences, truncated at the output limit, or ignored the schema because `format` was put inside `options` | each call site grew its own `_extract_json` / `_recover_json` |
| A thinking model returned an empty `content` with all the text in `thinking` | an empty string looked like a valid answer |
| A text-only model was picked as the vision judge | the problem surfaced as an HTTP 400 deep inside a run |
| Large models hit a fixed 120 s timeout | slow models failed, fast ones waited too long |
| Ollama, ComfyUI and in-process torch models fought over one GPU | out-of-memory crashes, depending on what happened to run first |
| The same model-tag lookup was copied into several modules | changing a model meant editing many files |
| Nobody could say afterwards which prompt, model and parameters produced an output | no way to compare or replay a call |

## Why existing tools fall short

- **LiteLLM** calls a hundred hosted providers through one API and has a large model registry, but it
  does not manage local model lifecycle or GPU memory and has no notion of decision questions.
  hone-models uses it as an optional provider for hosted models instead of competing with it.
- **Instructor / PydanticAI** do validation and retries well, but not context budgets, truncation
  detection or local-model quirks such as thinking-only replies.
- **llama-swap** swaps local servers, but knows nothing about the calls going through them.
- **LangChain** offers a large abstraction surface; hone-models wants a small, explicit one.

Nothing combined local and hosted models as equals, decision questions as a first-class call type, a
capability registry that includes local facts (VRAM, thinking mode, license), GPU-aware scheduling, and
a full record of every call.

## Who it is for

Python developers who call language models from ordinary software (pipelines, batch jobs, evaluation
code), especially on their own hardware, and who want failures to be loud and every call to leave
evidence.

## Core ideas

1. **Local and hosted are equals.** Ollama, any OpenAI-compatible server (llama.cpp, vLLM, LM Studio,
   OpenAI) and, through the optional LiteLLM provider, any hosted model share one interface and one
   record format.
2. **Check before calling.** A registry of models and their capabilities (vision, thinking, JSON schema,
   logprobs, context size, VRAM, price) lets hone-models refuse a call that cannot work *before* sending
   it: `CapabilityError` for an image to a text-only model, `ContextOverflow` with the numbers for a
   prompt that does not fit.
3. **Structured output never silently truncates.** Budget, constrain, parse, validate, retry with the
   error, repair as a last resort; the result says which of these happened (`structured_path`) and a
   reply cut at the output limit is always reported.
4. **Answer problems are values, call problems are exceptions.** A bad or empty answer sets
   `result.error`; an unreachable server or a missing capability raises a typed error.
5. **Decision questions are a call type.** Yes/no, choice and score questions go to a native decision
   model or are emulated on any LLM, with probabilities marked calibrated only when they really are.
6. **One GPU, many users.** Leases through a small cross-process ledger let model calls and non-LLM GPU
   work (Whisper, diffusion) take turns instead of crashing.
7. **Every call is recorded** as an OpenTelemetry-shaped span in a local SQLite file, including the named
   sections a prompt was built from, and any recorded call can be replayed with one thing changed.
8. **Small and plain.** The core needs only `pydantic` and `httpx`; the CLI, NVML readings and LiteLLM
   are optional extras.

## What it deliberately does not do

- Not an agent framework: no tool loops, memory or chains.
- Not a prompt-management UI or an evaluation tool.
- Not a hosted proxy or gateway service.
- Not a replacement for LiteLLM's hosted-provider coverage; it wraps it.
- Not (yet) streaming, async clients or media generation.

## What is in this folder

| File | What it holds |
|---|---|
| [`current.md`](current.md) | the design as it stands today: concepts, rules and the guaranteed behaviour (acceptance cases) |
| [`changes/`](changes/) | one record per design change: what was found, what was decided, why, and how to migrate |
| [`history/`](history/) | the early research the design started from, kept readable |
| [`decisions.md`](decisions.md) | smaller implementation choices, and the ones awaiting owner review |

A new design change starts as a record in `changes/` with status `proposed`; see
[CONTRIBUTING.md](../CONTRIBUTING.md#changing-the-design). How the package was built is described in the
[README](../README.md#how-this-was-built).
