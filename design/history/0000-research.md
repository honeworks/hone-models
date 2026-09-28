# 0000: Research (September 2026)

This is the research the design started from, rewritten from the original project brief. It describes
what was planned *before* anything was built; where the final design differs, a note says so.

> **Old names.** The brief used working names that were later replaced:
>
> | In the brief | Today |
> |---|---|
> | `modelkit` | `hone-models` (`import hone_models as mk`) |
> | `bestofn` (generate, score, select) | `hone-select` |
> | `tracelens` (workflow analyzer) | `hone-lens` |
> | a workflow runner | `hone-flow` |
> | `ours.*` span attributes | `hone.models.*` |
> | `mk.embed(texts)` | `mk.embedder(model_id).embed(texts)` |

## Where it came from

OneShotStudio is a local pipeline that turns ideas into songs and music videos. It called Ollama and
ComfyUI directly and hit a long list of bugs that had nothing to do with the creative work:

| Problem hit | What the new package should do |
|---|---|
| Ollama's default context (2048, 4096 on the running server) silently truncated a long JSON shot list | check the context budget before every call, set `num_ctx` automatically, fail loudly instead of truncating |
| JSON wrapped in markdown fences; truncated JSON; `format` placed inside `options`, where Ollama ignores it | schema-constrained output where supported, otherwise validate, retry with the error, repair |
| Gemma returned an empty `content` because it is a thinking model | a `thinking` capability, handled per provider |
| A model without vision was used as the vision judge (HTTP 400) | a capability registry: ask for `vision=True`, get an error before the call |
| A large model hit the 120 s request timeout | per-model timeouts from measured speed |
| GPU contention between Ollama, ComfyUI and torch models | lifecycle management and a simple GPU memory lease |
| ComfyUI broken by a leaked CUDA library path on `LD_LIBRARY_PATH` | a clean environment when launching local servers |
| The same model-tag lookup copied into several modules | one registry: id to provider, model tag, defaults, capabilities |

## What existed

- **LiteLLM**: one chat API for 100+ hosted providers and a registry of 2,500+ models. The plan: reuse it
  for hosted chat. It does not manage local lifecycle or GPU memory and has no decision-model interface.
- **Instructor / PydanticAI**: structured output with validation and retries; ideas to borrow.
- **llama-swap**: swaps local model servers on demand behind one OpenAI-compatible endpoint; a possible
  backend or inspiration.
- **LangChain**: a large abstraction surface, where a small explicit one was wanted.

Nobody combined local and hosted models as equals, decision models as a first-class type, a registry
with local facts (VRAM, license, thinking), and GPU-aware lifecycle.

## The concepts sketched

A text client (messages in, text or a validated object out), a decision client (a state and questions
in, answers with probabilities out), later a media client for image, audio and video generation; all
resolved through a registry of model entries with capabilities, over providers for Ollama, llama.cpp /
vLLM, OpenAI-compatible servers, hosted models through LiteLLM, and Jev; with a runtime layer for
lifecycle, GPU leases, timeouts, retries and records.

Decision questions on an LLM were to be emulated by one schema-constrained call, with probabilities
from token logprobs when available and otherwise from the model's stated confidence, marked
uncalibrated. This is what was built.

## Structured output, as planned

1. Budget: estimate prompt tokens; raise before calling if prompt plus output does not fit.
2. Constrain where the provider supports it.
3. Parse: strip fences, parse JSON.
4. Validate with Pydantic.
5. Retry with the validation error.
6. Repair truncated JSON as a last resort and mark the result as repaired.
7. Record the raw response, the attempts and which step succeeded.
8. An option to "reason first, constrain second".

Steps 1 to 7 were built. Step 8 was not: callers can still do it in two calls.

## Local runtime and GPU, as planned

Sessions (use a running server, else start one and stop only what was started), a clean environment for
launched servers, memory accounting from the registry's `vram_gb` and the driver, leases that wait
instead of crashing with out-of-memory, a public lease API for non-LLM GPU work (Whisper, Demucs,
ComfyUI), unloading, and per-model timeouts. Also sketched: warm / unload policies, swap-aware
reordering of queued work, multi-GPU and a llama-swap backend.

Built: sessions, clean environment, leases with NVML accounting and unloading of idle models, the public
lease API, unloading and per-model timeouts. Not built: warm / unload policies, reordering, multi-GPU,
llama-swap.

## Records, as planned

Every call recorded with its request, prompt structure (template id and version, variables, named
sections with character spans), model and version, response, usage and cost, timing, GPU facts,
reliability events and caller correlation ids; stored locally in SQLite or JSON lines with
OpenTelemetry GenAI field names; a `calls list | show | stats` CLI; secret stripping and a capture
switch; replay with overrides so a single prompt section can be changed and tested.

Built as planned, except the timing fields (no streaming, so no time to first token) and retention
policies. Replay, planned for a second version, was built in the first.

## Scope, as planned

- **v1:** registry and CLI; text clients for Ollama, OpenAI-compatible servers and LiteLLM; decision
  clients (Jev and emulation); vision input; the structured-output pipeline; Ollama sessions and unload;
  GPU scheduling; call records; prompt templates with sections; test fakes.
- **v2:** embeddings, swap-aware reordering, multi-GPU, OpenTelemetry export, replay, retention,
  llama-swap, streaming, async clients and rate limiting, cost reporting per run.
- **Later:** a media client for ComfyUI workflows and hosted image APIs.
- **Non-goals:** an agent framework, a prompt-management or evaluation tool, a replacement for LiteLLM's
  coverage, a hosted proxy.

What 0.1.0 shipped is v1 plus embeddings, replay and per-call cost from v2; see
[../changes/0001-initial-design.md](../changes/0001-initial-design.md).

## Open questions at the time, and how they were answered

| Question | Answer |
|---|---|
| Build order relative to the selection package | built in parallel, connected only through plain shapes |
| LiteLLM required, optional or unused | optional extra for hosted models |
| Where interfaces live | each consuming package owns the interface it needs; hone-models matches them structurally |
| Jev access (direct API or gateway, local weights, terms) | direct TypeSafe API assumed from public articles; still unverified |
| Own structured-output layer or Instructor | its own, short pipeline |
| Media generation here or separately | deferred |
| Name | `hone-models`, part of the honeworks family |
| License, Python | Apache-2.0, Python 3.11+ |

## References

- LiteLLM model registry: <https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json>
- Instructor: <https://github.com/567-labs/instructor>
- Structured output benchmark: <https://arxiv.org/html/2501.10868v1>
- Thinking before constraining: <https://arxiv.org/pdf/2601.07525>
- llama-swap: <https://github.com/mostlygeek/llama-swap>
- Jev decision model (TechTarget): <https://www.techtarget.com/it-infrastructure/news/366650696/Jev-decision-model-touted-as-quicker-cheaper-LLM-alternative>
- Building a harness with Jev (LangChain): <https://www.langchain.com/blog/building-a-harness-with-jev>
- Arize Phoenix: <https://github.com/arize-ai/phoenix>
