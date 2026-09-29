# hone-models examples

One runnable file per public concept, in reading order. Each file opens with **What** it shows, **How**
(the calls, in order) and **Why** (the problem it solves, and its pitfalls), then runs top to bottom and
`assert`s the key facts. These are the reference patterns to copy: they use only the public API.

Every example runs offline and deterministically: it starts the packaged `FakeOllama` (see
[testing_with_fake_ollama.py](testing_with_fake_ollama.py)) instead of a real server, and scripts bad
answers with `server.queue(...)`. To run one against your own Ollama, keep the model calls and drop the
`FakeOllama`, `server.queue` / `server.requests` lines and the asserts on the fake's canned answers.

```bash
uv run python examples/quickstart.py        # or: python examples/<file>.py with hone-models installed
```

The test suite runs every example (`tests/e2e/test_ac20_examples.py`). The Design column names the
section of [design/current.md](../design/current.md) that the example illustrates.

## Calling models

| Example | Concept | What it shows | Design |
|---|---|---|---|
| [quickstart.py](quickstart.py) | `mk.text`, `TextResult` | One chat call: text, finish reason, usage, and the recorded span. | §2 |
| [registry.py](registry.py) | registry, model ids | TOML merge, lookups, ad-hoc `provider:model` ids, `require=` / `prefer=` selection. | §3 |
| [chat_and_structured.py](chat_and_structured.py) | structured output | Pydantic or JSON Schema output and every `structured_path`: constrained, parsed, retried, repaired, failed. | §5 |
| [context_budget.py](context_budget.py) | context budget, truncation | `ContextOverflow` before calling, `num_ctx`, and detecting replies cut off at `max_tokens`. | §5 |
| [thinking_models.py](thinking_models.py) | thinking models | Reasoning off by default, `think=True`, thinking-only replies reported as `r.error`. | §3, §4 |
| [vision.py](vision.py) | image inputs | Image parts from files or base64, vision selection, `CapabilityError` for text-only models. | §2, §4 |
| [prompt_sections_and_replay.py](prompt_sections_and_replay.py) | `mk.Prompt`, `Replayer` | Named, versioned sections recorded with character spans; replay without a section or with new params. | §2, §8 |

## Providers

| Example | Concept | What it shows | Design |
|---|---|---|---|
| [openai_compatible.py](openai_compatible.py) | OpenAI-compatible servers | A registry entry for llama.cpp / vLLM / LM Studio / OpenAI: chat, schemas, logprobs, embeddings. | §4 |
| [litellm_provider.py](litellm_provider.py) | LiteLLM (extra) | Hosted models through LiteLLM from a registry entry or a `litellm:` id. | §4 |

## Decisions, embeddings, speech and generation

| Example | Concept | What it shows | Design |
|---|---|---|---|
| [decisions.py](decisions.py) | `mk.decision` (emulated) | Yes/no, choice and score questions in one LLM call; a bad answer fails alone; `calibrated`. | §6 |
| [jev_decisions.py](jev_decisions.py) | native decision model | The same questions sent to Jev: calibrated answers, cost, a missing answer. | §4, §6 |
| [embeddings.py](embeddings.py) | `mk.embedder` | Unit-length vectors, cosine ranking, vector-size checks. | §2 |
| [speech.py](speech.py) | `mk.speech`, `FakeSpeech` | Text to a WAV file: voices, speed, minutes-long narration in chunks, the speech span. | §2, §8 |
| [generation.py](generation.py) | `mk.image`, `mk.music`, ComfyUI entries | An image from a prompt and a reference, song takes in one session, an input the model does not take, an out-of-memory job as a result, read back from the spans. | §2, §4, §8 |
| [expressive_speech.py](expressive_speech.py) | `emotion=`, `intensity=`, `session()`, `chatterbox` | A lesson narrated paragraph by paragraph, each with its own emotion and intensity, in one session (one model load), read back from the spans. | §2, §4, §8 |

## Reliability and records

| Example | Concept | What it shows | Design |
|---|---|---|---|
| [retries_and_errors.py](retries_and_errors.py) | retries, typed errors | Transient errors retried and recorded, the exception hierarchy, `r.error` vs. exceptions. | §4 |
| [records.py](records.py) | records | Sinks, trace context, content capture off, secret stripping, reading spans back. | §8 |
| [calls_cli.py](calls_cli.py) | `hone-models` CLI | `calls list/show/stats`, `models list/check`, and the same queries from Python. | §3, §8 |

## Local GPU and servers

| Example | Concept | What it shows | Design |
|---|---|---|---|
| [gpu_lease.py](gpu_lease.py) | `mk.gpu.lease` | Reserving VRAM across processes, waiting and timeouts, nested leases, unloading idle models, release hooks for other GPU servers, lease attributes. | §7 |
| [sessions.py](sessions.py) | `mk.session`, `mk.unload` | Use or start the Ollama server for a block; free a model's memory now. | §9 |

## Testing

| Example | Concept | What it shows | Design |
|---|---|---|---|
| [testing_with_fake_ollama.py](testing_with_fake_ollama.py) | `FakeOllama` | Tests for your own code: inspect requests, script answers and failures, check a custom sink. | §12 |
