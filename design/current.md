# hone-models: current design (0.1.0)

This is the design as it stands today. Why it looks like this is in [changes/](changes/) and
[decisions.md](decisions.md); the problem it solves is in [README.md](README.md). Section numbers are
stable: code comments and `examples/README.md` refer to them.

## 1. Purpose and scope

One interface for calling AI models, **local and hosted as equals**: chat and text, structured output,
decision questions (native decision models such as Jev, or emulated on LLMs), image input,
embeddings and local text to speech ([0006](changes/0006-speech.md), expressive: [0011](changes/0011-expressive-speech.md)). Around the call it provides what projects otherwise re-implement badly: a **capability
registry**, **structured output that never silently truncates**, **local model lifecycle and GPU
scheduling**, and **full call records** with replay.

Goals:
- `mk.text("gemma4-12b").complete(messages, schema=Model)` behaves the same for Ollama, any
  OpenAI-compatible server (llama.cpp, vLLM, LM Studio, hosted) and hosted models through LiteLLM.
- Capabilities (vision, thinking, context size, JSON schema) are checked **before** calling.
- Every call is recorded, including the named sections a prompt was built from.
- GPU work on one machine is coordinated through leases that non-LLM code can take too.

Out of scope in 0.1: agent loops, a prompt-management UI, a hosted proxy, media generation, streaming,
async clients.

## 2. Public API

```python
import hone_models as mk

mk.text(model_id=None, *, require=None, prefer=None, registry=None, sink=None) -> TextClient
mk.decision(model_id, *, registry=None, sink=None) -> DecisionClient
mk.embedder(model_id, *, registry=None, sink=None) -> Embedder
mk.speech(model_id, *, registry=None, sink=None) -> SpeechClient   # extras `speech` (kokoro-82m), `expressive` (chatterbox)
mk.Prompt(template_id, version, sections: dict[str, str | Section], variables=None, system_sections=("system",))
mk.Section(text, version=None)          # a section with its own version
mk.YesNo(instructions), mk.Choice(options, instructions), mk.ScoreQ(instructions, scale=(1, 5), anchors=None)
mk.session(provider="ollama")          # context manager: use the running server, else start it; stop only what it started
mk.unload(model_id)                    # free VRAM now
mk.gpu.lease(name, vram_gb, *, timeout_s=None, trace=None)
mk.gpu.status() -> GpuStatus           # total / used / free MB, leases held, leases waiting
mk.gpu.GPU, mk.gpu.GpuScheduler(ledger, memory=..., processes=..., unload_others=False, stall_s=120, if_busy="unload")
mk.gpu.NullGpuLease(), mk.gpu.FileLockGpuLease(path)
mk.gpu.on_short(name, release | None), mk.gpu.comfyui_free(url), mk.gpu.torch_empty_cache   # release hooks
mk.machine.snapshot() -> dict          # GPUs, model servers, loaded models, the GPU lock, leases (§7)
mk.machine.prepare(needed, *, if_busy="block") -> dict   # only the needed models loaded
mk.machine.load(model_id) -> dict      # warm an Ollama model up
mk.machine.Machine(*, lock_path=None, registry=None, sink=None, gpus=..., processes=..., scheduler=None), mk.machine.MACHINE
mk.registry.load(paths=None) -> Registry
mk.replay.Replayer(sink=None, registry=None)
mk.records.SqliteSpanSink(path), JsonlSpanSink(path), MemorySink(), NullSink()
mk.records.read_spans(path), default_store(), add_secret(value)
mk.current_trace()
mk.errors: HoneModelsError, ConfigError, CapabilityError, ContextOverflow, ProviderError, ModelTimeout, ValidationFailed
mk.PORTS_VERSION
hone_models.calls: find_calls(spans, since=..., model=...), call_stats(calls, by=...)
hone_models.testing: FakeOllama(responder=None), FakeSpeech, FakeSpeech.like(model_id), check_record_sink
```

`mk.*` names are exactly those in `hone_models.__all__`. The examples use only these names, plus
`hone_models.testing` and `hone_models.calls`.

### 2.1 Usage

```python
import hone_models as mk
from pydantic import BaseModel

llm = mk.text("gemma4-12b")                                   # a registry id
r = llm.complete([{"role": "user", "content": "Write a haiku about rain."}])
print(r.text, r.usage, r.finish_reason)

class Verdict(BaseModel):
    score: int
    rationale: str
r = llm.complete([{"role": "user", "content": "Rate this pitch 1-5 ..."}], schema=Verdict)
r.parsed          # a Verdict, or None with r.error set
r.structured_path # "constrained" | "parsed" | "retried" | "repaired" | "failed"

vision = mk.text(require={"vision": True, "min_context": 8000}, prefer="local")
r = vision.complete([{"role": "user", "content": [
        {"type": "text", "text": "Describe the image."},
        {"type": "image", "path": "frame.png"}]}])

prompt = mk.Prompt("song_ideas", "3", sections={
    "system": "You write song pitches.",
    "rules": "Exactly 12 ideas, JSONL.",
    "format_example": '{"title": "Cargo Hold Chaos", ...}',
    "brief": "Theme: {theme}",
}, variables={"theme": "sea shanties"})
r = llm.complete(prompt)          # renders to messages; sections recorded with character spans

dm = mk.decision("gemma4-12b")    # an LLM emulating a decision model
a = dm.decide("The deploy failed twice and customers see 500s.", {
    "urgent": mk.YesNo("Does this need attention now?"),
    "team": mk.Choice(["infra", "backend", "frontend"], "Which team owns it?"),
    "severity": mk.ScoreQ("How severe?", scale=(1, 5)),
})
a["urgent"]["value"], a["team"]["choice"], a["severity"]["raw"]

vecs = mk.embedder("nomic-embed-text").embed(["a song about rain", "a song about the sea"])

tts = mk.speech("kokoro-82m")     # local TTS; long text is split into sentences and joined
s = tts.synthesize("Welcome to the course.", voice="am_michael", speed=1.0, out="intro.wav")
s.path, s.duration_s, s.sample_rate, s.voice, s.model, s.span_id, s.segments   # segments: chunk timings
actor = mk.speech("chatterbox")   # expressive TTS: emotion / intensity per call, a paragraph at a time
actor.synthesize("You did it!", voice="warm_male", emotion="encouraging", intensity=0.6, out="win.wav")
with actor.session():             # load once, keep the lease, free at the end of the block
    for i, line in enumerate(lines):
        actor.synthesize(line, emotion="warm", out=f"line_{i}.wav")

with mk.gpu.lease("whisper-turbo", vram_gb=6):
    run_whisper()                 # non-LLM GPU work takes part in scheduling

with mk.session("ollama"):
    ...
mk.unload("gemma4-12b")
```

### 2.2 Results and errors

`complete` returns a `TextResult` with `text`, `parsed`, `error`, `model`, `finish_reason`, `usage`
(`input_tokens`, `output_tokens`), `structured_path`, `attempts`, `logprobs` and `span_id`.
`synthesize` returns a `SpeechResult` with `path`, `duration_s`, `sample_rate`, `voice`, `model`,
`span_id` and `segments`: a `SpeechSegment(text, start_s, end_s)` per synthesized chunk, from the join
points ([0009](changes/0009-speech-sessions-and-timings.md)); `by_sentence=True` makes each sentence
its own chunk, so the segments are sentence timings; an unknown voice, a speed outside 0.5..2.0, an emotion not in `tts.emotions`, an intensity
outside 0.0..1.0 or empty text raise `ConfigError`. `emotion` / `intensity` are applied by models whose
`capabilities.expressive` is true (`tts.expressive`: Chatterbox) and ignored by the others (Kokoro).
Text is chunked by paragraph (blank lines): a paragraph up to `defaults.max_chunk_chars` (default 400)
is synthesized whole, a longer one is split on sentences into chunks of about equal length; chunks are
joined with 0.25 s of silence, paragraphs with 0.6 s. A plain call loads the model and frees it
afterwards; `with tts.session():` holds the GPU lease for the block, loads the model on its first call
and frees it once at the end (also on an exception); sessions do not nest.

- Problems with the model's **answer** (empty reply, thinking-only reply, JSON still invalid after
  retries, truncated reply) are reported on the result: `error` is set, nothing is silently empty.
- Problems with the **call** raise typed exceptions, all subclasses of `HoneModelsError`, itself a
  `RuntimeError`: `ConfigError` (unknown id, bad registry, missing API key), `CapabilityError` (the model
  is declared unable to do what was asked), `ContextOverflow` (a `CapabilityError`, with the numbers),
  `ProviderError` (HTTP or transport failure, with the status), `ModelTimeout` (a `ProviderError`),
  `ValidationFailed`.

## 3. Registry

TOML files merged in order: packaged defaults (`hone_models/data/models.toml`), the user file
`~/.config/hone/models.toml`, the project file `./hone-models.toml`, then explicit paths.

```toml
[models."gemma4-12b"]
provider = "ollama"
model    = "gemma4-12b:latest"
defaults = { temperature = 0.8 }
[models."gemma4-12b".capabilities]
vision = false
thinking = true            # send think=false unless the caller asks for reasoning
json_schema = true         # Ollama format=<schema>
logprobs = false
max_input_tokens = 32768
max_output_tokens = 8192
vram_gb = 7.4
license = "Gemma Terms of Use"

[models.jev]
provider = "jev"
kind = "decision"
api_key_env = "TYPESAFE_API_KEY"
[models.jev.capabilities]
questions = ["yes_no", "choice", "score"]
calibrated = true
price = { input_per_mtok = 0.042, output_per_mtok = 0.0 }

[models."gpt-4.1-mini"]
provider = "openai_compatible"
model = "gpt-4.1-mini"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
```

The packaged defaults are `gemma4-12b`, `qwen2.5vl-7b`, `deepseek-r1-8b`, `nomic-embed-text`, `jev`,
`gpt-4.1-mini`, `kokoro-82m` and `chatterbox`. `kind` is `chat` (default), `embedding`, `decision` or
`speech`; a vision model may declare an image's prompt cost (`image_tokens` flat, or `image_patch_px`);
a speech model lists its voices in `capabilities.voices` (the first is the default) and whether
it applies emotion and intensity in `capabilities.expressive`.

- **Unknown is not false.** Every capability defaults to unknown (`None`). Pre-call checks refuse only
  what the registry *declares* impossible; selection matches only declared values.
- **Ad-hoc ids.** `provider:model` ids (`ollama:llama3.2:1b`, `openai:gpt-4.1`, `litellm:<model>`) work
  without a registry entry. Ollama ids are probed through `/api/show`; LiteLLM ids take hints from
  LiteLLM's model map; the rest stay unknown.
- **Selection.** `require={...}` takes `min_context` (context size at least n) and any capability name
  (exact match); unknown keys raise `ConfigError`. `prefer=` is `"local"`, `"hosted"`, `"cheapest"`
  (local models cost 0, unknown hosted prices last) or `"fastest"` (by measured `speed_tok_s`, unknown
  last); without it candidates are ordered by id. No match raises `CapabilityError` listing the closest
  candidates.
- **CLI** (`cli` extra): `hone-models models list | show <id> | check <id>`. `check` sends a short smoke
  call, measures tokens per second and writes `speed_tok_s` to the user registry.

## 4. Providers

| Provider | Talks to | Notes |
|---|---|---|
| `ollama` | `/api/chat`, `/api/embed`, `/api/show`, `/api/tags` | `think` handling; `format=<schema>`; images as base64; `options.num_ctx` from the context budget; `keep_alive` for unload |
| `openai_compatible` | `/v1/chat/completions`, `/v1/embeddings` | `response_format` JSON schema when supported; token logprobs when returned; covers llama.cpp, vLLM, LM Studio, Ollama's `/v1`, OpenAI |
| `litellm` (extra) | any model LiteLLM knows | same OpenAI-shaped body; LiteLLM retries transient errors itself |
| `kokoro` (extra `speech`) | Kokoro-82M in-process (torch) | speech-only; English voices need spaCy's `en_core_web_sm`, installed by the user (not on PyPI): without it `synthesize` raises `ConfigError` before the lease ([0008](changes/0008-speech-extra-and-the-spacy-model.md)); weights from Hugging Face (`hexgrad/Kokoro-82M`) on first use into `HF_HOME`; loaded per call inside a GPU lease and freed afterwards; 24 kHz; `defaults.device` overrides CUDA / CPU |
| `chatterbox` (extra `expressive`) | Chatterbox (Resemble AI, MIT) in-process (torch) | speech-only, expressive: intensity sets `exaggeration` (0.25 + intensity), the emotion sets `cfg_weight` (pacing); voices are packaged reference clips rendered by Kokoro's synthetic voices, plus the voice bundled with the weights; weights (about 3 GB) from `ResembleAI/chatterbox` on first use; loaded per call inside a 5 GB lease and freed afterwards; 24 kHz; `speed` by time-stretching; `defaults.seed` (0) |
| `jev` | TypeSafe AI's decision API | decision-only: a state and questions in, answers out; `TYPESAFE_API_KEY`; base URL configurable. The request / response mapping is built from public articles and **unverified** against the real API |

All HTTP goes through `httpx`. Timeouts are `2 * max_tokens / speed`, never below 120 s and never above
`max_timeout_s` (600 s); the speed is the measured `speed_tok_s`, else 10 tokens/s for a local model
([0010](changes/0010-timeout-without-measured-speed.md)); a hosted model without one gets 120 s. Transient errors (connection errors, 429, 5xx) are retried with jittered
backoff, 3 attempts in all, each retry recorded as a span event; other 4xx errors are not retried. API
keys come only from environment variables named by `api_key_env`.

## 5. Structured output

When `schema=` is given (a Pydantic model class or a JSON Schema dict):

1. **Budget.** Prompt tokens are estimated: characters / 3.5, plus each image at the model's
   `image_tokens`, else `ceil(w / image_patch_px) * ceil(h / image_patch_px)` from the image header (PNG,
   JPEG, GIF, WebP), else 1024 ([0013](changes/0013-context-budget-counts-images.md)). The requested
   output is the `max_tokens` param, else the model's `max_output_tokens`, else 2048, and it is always
   sent, so a cut reply shows up as finish reason `length`. If prompt + output exceeds
   `max_input_tokens`, `ContextOverflow` is raised with the numbers before any request. For Ollama,
   `num_ctx` = (prompt + output) × 1.1, rounded up to a multiple of 2048 (Ollama reloads a model whenever
   `num_ctx` changes), at least the `min_num_ctx` param (or registry default), and capped at
   `max_input_tokens`.
2. **Constrain.** The schema is always added to the system message; the provider constraint (Ollama
   `format`, OpenAI `response_format`) is added when the model declares `json_schema`.
3. **Parse.** Markdown fences and surrounding prose are stripped, then `json.loads`.
4. **Validate** with Pydantic (`TypeAdapter`) for a model class, or with a small JSON Schema validator for
   a dict.
5. **Retry** up to 2 times, sending the validation error back as a user message; each retry is a
   `structured_retry` span event and `usage` sums all attempts.
6. **Repair** truncated JSON as a last resort by closing structures at the last complete element.
7. A reply with finish reason `length` is always reported and makes the path at best `repaired`.

`structured_path` is `constrained` (constraint sent, raw text parsed without cleanup), `parsed` (fences
or prose had to be stripped), `retried`, `repaired` or `failed` (`parsed` is `None`, `error` set).

Thinking models: when the registry declares `thinking = true`, `think=false` is sent unless the caller
asks for reasoning. A reply with no content sets `error` (naming `thinking` when only thinking text came
back), never a silent empty string.

## 6. Decision questions

`dm.decide(state, questions, *, images=(), trace=None)` answers structured questions about a state
(text or a JSON-able object). Questions are plain dicts; `mk.YesNo`, `mk.Choice` and `mk.ScoreQ` build
them:

| Key | Required | Meaning |
|---|---|---|
| `type` | yes | `"yes_no"`, `"choice"` or `"score"` |
| `instructions` | yes | the question in plain English |
| `options` | for `choice` | option labels |
| `scale` | for `score` | `[low, high]`, default `[1, 5]` |
| `anchors` | no | labels for scale points |

Every answer has the same keys:

| Key | Meaning |
|---|---|
| `type` | the question type |
| `value` | yes/no: P(yes); choice: probability of the chosen option; score: `(raw - low) / (high - low)`; `None` on error |
| `choice`, `probabilities`, `raw` | the chosen option, per-option probabilities, the score on its scale |
| `confidence`, `rationale` | as stated by the model |
| `calibrated` | `True` only when probabilities come from token logprobs or a decision model |
| `error` | set when this question could not be answered |

**Emulated on an LLM:** one schema-constrained call at temperature 0 answers every question with
`answer`, `confidence` (0-1) and `rationale`. Invalid answers are sent back for correction (up to 2
retries); what is still invalid becomes that question's `error`, the others stand. When the model
declares `logprobs = true`, P(yes) comes from the token logprobs and is marked calibrated; otherwise the
stated confidence is used and `calibrated` is `False`.

**Native (Jev):** questions are mapped to the decision API. Probabilities outside [0, 1], choice
probabilities over unknown options or summing above 1 become that question's `error`.

## 7. GPU scheduling

- **Accounting.** Total and used memory come from NVML (`gpu` extra), else `nvidia-smi`; reservations
  come from a JSON ledger at `${HONE_HOME}/models/gpu-ledger.json` guarded by `fcntl.flock`. Free memory
  is `total - max(used, reserved)`, so a lease counts before its model loads and other programs count
  too. Entries of dead processes are dropped.
- **Lease.** `lease(name, vram_gb)` reserves memory for the duration of a `with` block. When memory is
  short, idle Ollama models this process loaded through hone-models are unloaded once, then the lease
  polls every 0.5 s until `timeout_s`, and raises `TimeoutError`. `GpuScheduler(unload_others=True)`
  also unloads models other processes loaded. A lease larger than the whole GPU raises
  `CapabilityError` at once. Without GPU information, leases are granted at once but still written to
  the ledger. Between the unloads and the wait, the release hooks registered in this process with
  `on_short(name, release)` are called once, in order (a hook that raises is logged and skipped), for
  GPU users the lease cannot unload itself: `comfyui_free(url)` (`POST /free`), `torch_empty_cache`
  ([0005](changes/0005-release-hooks-for-other-gpu-servers.md)). Memory used by the leasing process itself (per-process NVML / `nvidia-smi` readings) is
  not counted against its lease ([0004](changes/0004-lease-that-can-never-be-granted.md)).
- **No endless waits** ([0004](changes/0004-lease-that-can-never-be-granted.md)). With no other lease
  held and free memory unchanged for `stall_s` (120 s), the wait raises `CapabilityError` naming the
  processes holding GPU memory and any Ollama models still loaded. A wait over 30 s is logged (then every
  5 minutes) and listed in `status().waiting` (a `gpu-waiting.json` next to the ledger).
- **Nested and reentrant.** Reentrant per (process, thread, name). A lease taken inside other leases of
  the same process and thread on the same ledger (any scheduler instance) is part of them: it reserves only
  `vram_gb` minus what they reserved (nothing when covered; a ledger entry with `nested_in` otherwise), so
  a library's own lease inside an app's batch lease cannot wait for the app
  ([0014](changes/0014-nested-leases-in-one-process.md)); when even that cannot fit, the stall error
  names the outer lease. A lease records no span of
  its own; spans made inside it carry `hone.models.gpu.lease_wait_ms`, `hone.models.gpu.vram_before_mb`
  (when known), `hone.models.gpu.unloaded` and `hone.models.gpu.released` (the hooks that ran).
- **Model calls do not lease on their own.** Callers wrap GPU work in `mk.gpu.lease(...)`. Whether local
  model calls should lease automatically is an open question ([decisions.md](decisions.md), D-009).
- `NullGpuLease` (never waits) and `FileLockGpuLease(path)` (one user at a time) have the same shape.
- **Other processes' models** ([0016](changes/0016-machine-state.md)). `GpuScheduler(unload_others=True,
  if_busy="block")` unloads other processes' models only while no other process holds a lease or the
  machine-wide lock; `"unload"` (the default) unloads them anyway.
- **Machine state** ([0016](changes/0016-machine-state.md)). `mk.machine` reports and controls which models
  the machine holds; `Machine` has the shape of hone-select's `MachineProbe` (§10), `MACHINE` is the
  default instance, and the module functions are its methods. Sizes are GB with two decimals; `None` always
  means unknown, never 0; nothing waits, retries or decides a policy.
  - `snapshot()` (no side effects, never raises for a missing reader or a dead server) returns `time`;
    `gpus` (every device from NVML, else one `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu`:
    `index`, `name`, `memory_{total,used,free}_gb`, `utilization_pct`, `processes` `[{pid, memory_gb, mine}]`
    for GPU 0, else `None`; the whole list `None` without a reader); `servers` (`{server, url, running,
    error}` for Ollama at `OLLAMA_HOST` and each ComfyUI server, plus `vram_gb` = the sum of
    `/system_stats` `torch_vram_total`; `running` `True` answered, `False` connection refused, `None` timed
    out or failed, with `error`; 2 s timeout); `loaded_models` (`{server, name, model_id, size_gb,
    vram_gb}`: Ollama from `/api/ps`, `model_id` the registry id whose name, with `:latest` added when it
    has no tag, matches; ComfyUI from the loaded-models file, sizes `None`, dropped when the server holds
    no torch memory, plus an entry with `model_id: None` when the newest `/history` job is not in the file
    or memory is held with nothing named); `gpu_lock` (`{path, held, mine, holder}`: a non-blocking `flock`
    released at once, `None` when the file cannot be opened, not held when it does not exist; `mine` from
    `HONE_GPU_LOCK_HELD=1`; `holder` from `<lock>.holder`, read only when held); `leases` (the ledger:
    `{name, pid, vram_gb, mine}`).
  - ComfyUI servers checked: the distinct `base_url`s of the registry's `comfyui` entries (default
    `HONE_COMFYUI_URL`, else `http://127.0.0.1:8188`) plus `HONE_COMFYUI_URL` when set; none otherwise.
    The loaded-models file is `${HONE_HOME}/models/comfyui-loaded.json` (`{url: [{model_id, job_id, pid,
    time}]}`, guarded by `flock`).
  - `prepare(needed, *, if_busy="block")` resolves the ids (`ConfigError` for an unknown id or `if_busy`),
    reads the servers, and finds `blocked_by`: the lock held and not `mine`, and live leases of other
    processes. Unless blocked with `"block"`, it unloads every Ollama model not needed (`keep_alive: 0`)
    and sends ComfyUI `/free` when it holds a model not needed or unnamed (then clears that server's list
    in the file), then reads the servers again. It returns `needed`, `blocked_by` (`None` or strings),
    `unloaded` (`[{server, name}]`), `released` (server names), `if_busy`, `errors` (`[{server, name,
    error}]`: failed unloads or frees, servers with `running: None`; never raised), `missing` (needed
    Ollama or ComfyUI ids not loaded), `loaded_models` (after) and `need_gb` (the sum of `vram_gb` of the
    needed models that use this GPU: `local`, `comfyui`, `command`; `None` if one is unknown). It takes
    no lease or lock and never loads or waits.
  - `load(model_id)` warms an Ollama model up: `POST /api/generate` with an empty prompt and
    `defaults.keep_alive` (default `"5m"`), then `/api/ps`; returns `{model_id, loaded, seconds, size_gb,
    vram_gb, error}`, `loaded: False` with the error when the server fails or times out. Other providers:
    `loaded: None` and `error: "not supported for <provider>: ..."` (run a small job in a session; speech
    and transcription models load in-process inside a session), with no request.

## 8. Records and replay

### 8.1 Spans

Every call emits one span, `hone.models.chat`, `hone.models.decide`, `hone.models.embed` or
`hone.models.speech` (kind
`client`); an emulated decision is a decide span with its chat call as a child. `mk.machine.prepare` and
`mk.machine.load` emit `hone.models.machine.prepare` / `hone.models.machine.load` (kind `internal`);
`snapshot` emits none. A span is a JSON object:

```json
{
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "parent_span_id": null,
  "name": "hone.models.chat",
  "kind": "client",
  "start_time": "2026-09-27T14:03:11.120Z",
  "end_time": "2026-09-27T14:03:13.402Z",
  "status": {"code": "ok", "message": ""},
  "attributes": {"hone.schema_version": "1", "...": "see §8.3"},
  "events": [{"name": "retry", "time": "...", "attributes": {"attempt": 2}}],
  "resource": {"hone.package": "hone-models", "hone.package.version": "0.1.0", "host.name": "...", "process.pid": 1234},
  "links": []
}
```

Ids are W3C hex (32 characters for traces, 16 for spans); times are ISO-8601 UTC with milliseconds;
`status.code` is `ok`, `error` or `unset`.

### 8.2 Sinks and the SQLite store

A sink has `emit(span)`, `flush()` and `close()`; sinks never raise into the caller (a failure is
logged once and counted). The default is `SqliteSpanSink` at `${HONE_HOME:-.hone}/models/spans.db`,
with the tables `meta` (schema name, `schema_version = 1`, package, creation time), `spans` (one row per
span, JSON columns for attributes, events, resource and links), `blobs` (values over 64 KiB, referenced as
`{"$blob": "<sha256>"}`) and `changes` (a change log for incremental readers). It runs in WAL mode with
a 5 s busy timeout, so several processes can write to one file. `JsonlSpanSink` writes one span per line;
`MemorySink` and `NullSink` are for tests.

### 8.3 Attributes

OpenTelemetry GenAI names where they exist, `hone.models.*` otherwise:

| Attribute | Content |
|---|---|
| `gen_ai.operation.name`, `gen_ai.provider.name` | `chat` / `embeddings` / `speech`; `ollama`, `openai`, `litellm`, `jev`, `kokoro`, `chatterbox` |
| `gen_ai.request.model`, `gen_ai.response.model` | the provider's model name |
| `gen_ai.request.{temperature,top_p,max_tokens,seed}` | |
| `gen_ai.response.finish_reasons`, `gen_ai.usage.{input_tokens,output_tokens}` | |
| `gen_ai.input.messages`, `gen_ai.output.messages` | content |
| `hone.models.model_id`, `hone.models.model_digest` | registry id; exact version when the provider reports one |
| `hone.models.prompt.{template_id,template_version}` | for `mk.Prompt` inputs |
| `hone.models.prompt.sections` | `[{"id", "version", "role", "start", "end", "sha256"}]`: character spans in the rendered text |
| `hone.models.prompt.variables` | content |
| `hone.models.structured.{path,attempts,schema}` | the path taken, attempts, the JSON Schema |
| `hone.models.context.{limit,estimated_prompt_tokens,estimated_image_tokens}` | the budget; the image share only when images were sent |
| `hone.models.request.params` | all call params (used by replay) |
| `hone.models.cost_usd` | when the registry has a price |
| `hone.models.gpu.{lease_wait_ms,vram_before_mb,unloaded,released}` | inside a lease |
| `hone.models.replay_of` | the original span id, on replays |
| `hone.models.decision.{state,questions,answers}` | decide spans; content |
| `hone.models.speech.input` | speech spans: the text; content |
| `hone.models.speech.{voice,speed,chars,chunks,paragraphs,expressive,session,loaded,duration_s,sample_rate}` | speech spans; `loaded`: this call loaded the model |
| `hone.models.speech.{emotion,intensity}` | speech spans, only when given |
| `hone.models.machine.{needed,if_busy,unloaded,released,blocked_by,errors,missing}` | `hone.models.machine.prepare` spans, as returned; status `error` when `errors` is not empty |
| `hone.models.machine.{model_id,loaded,seconds,vram_gb}` | `hone.models.machine.load` spans; status `error` when `loaded` is `False` |
| `hone.schema_version`, and `hone.run_id`, `hone.item`, `hone.step`, `hone.candidate_id`, `hone.scorer`, `hone.lens.finding_id` | on every span; the last six copied from the trace context when present |

HTTP retries are `retry` events; structured-output retries are `structured_retry` events. Not recorded
in 0.1: `hone.models.timing.*` (no streaming, so no time to first token) and
`hone.models.gpu.vram_after_mb` (spans inside a lease end before the lease does).

### 8.4 Content capture and secrets

`HONE_CAPTURE_CONTENT=0`, or `SqliteSpanSink(path, capture_content=False)`, stores every attribute
marked "content" above, and the params, only as `{"sha256", "len"}`. Headers are never recorded. The
value of every `api_key_env` in use, anything shaped like a bearer token or `sk-...` key, and values
registered with `mk.records.add_secret` are replaced with `***` before writing.

### 8.5 Trace context

Every call takes `trace=`, a mapping with an optional W3C `traceparent`
(`00-<trace id>-<parent span id>-01`) and optional `hone.*` keys. The span joins that trace as a child
of the given parent; without one, the current context (a `ContextVar`, readable with
`mk.current_trace()`) is used, else a new trace starts.

### 8.6 Replay

`mk.replay.Replayer().replay_call(span, overrides, *, trace=None)` rebuilds the request of a recorded
chat call, applies overrides, calls the model and records a new span with `hone.models.replay_of`.

- Overrides: `"prompt.sections"` (`{id: new_text | None}`, `None` removes a section; the prompt is
  rebuilt from the recorded section spans, keeping roles and versions), `"model"`, `"params"` (merged)
  or `"messages"` (full replacement, not together with `prompt.sections`). Unknown keys raise
  `ConfigError`.
- The span must have been recorded with content capture on (`ConfigError` otherwise). Everything not
  overridden, including the original model's registry defaults, is taken from the span.
- Spans from other recorders work too: without `hone.models.model_id` the model comes from
  `gen_ai.provider.name` + `gen_ai.request.model` as an ad-hoc id; list and object attributes may be
  JSON strings (the OpenTelemetry encoding); section records without a `role` take the role of the
  message their `start` falls in.
- The new span is returned and also written to the replayer's sink.

### 8.7 The `calls` CLI

`hone-models calls list [--since 1d] [--model X]`, `calls show <span_id>` and
`calls stats --by model|provider|tag` read the default store (or `--db`). Each model request counts once
(an emulated decide span is represented by its chat child); `--by tag` groups by `hone.step`. The same
queries are `hone_models.calls.find_calls` and `call_stats`.

## 9. Local lifecycle

`mk.session("ollama")` checks `GET /api/version`; if the server answers, it is used as is. Otherwise
`ollama serve` is started with a **clean environment** (no `VIRTUAL_ENV`, `LD_LIBRARY_PATH`,
`PYTHONPATH`, `PYTHONHOME`) and `OLLAMA_HOST` set to the URL it waits on, for up to 30 s; on exit only a
server the session started is stopped. A remote `OLLAMA_HOST` is never started. The session yields the
server URL. `mk.unload(model_id)` sends `keep_alive: 0`. Both are Ollama-only in 0.1 (`ConfigError` for
other providers).

## 10. Shapes other code can rely on

hone-models imports no other package's interfaces. Its clients have plain, documented shapes, so code
that expects a text client, a decision client, an embedder, a GPU lease or a call replayer can take them
directly. `mk.PORTS_VERSION` is `"1"`.

| Object | Shape |
|---|---|
| `mk.text(...)` | `complete(messages, *, schema=None, trace=None, **params) -> TextResult` (§2.2); unknown params are ignored |
| `mk.decision(...)` | `decide(state, questions, *, images=(), trace=None) -> {name: answer}` (§6) |
| `mk.embedder(...)` | `model_id`, `dimensions`, `embed(texts, *, trace=None) -> list[list[float]]`: L2-normalized, input order, `[]` for no input |
| `mk.gpu`, `mk.gpu.GPU`, `NullGpuLease`, `FileLockGpuLease` | `lease(name, vram_gb, *, timeout_s=None, trace=None)` context manager, reentrant, `TimeoutError` on timeout (§7) |
| `mk.replay.Replayer()` | `replay_call(span, overrides, *, trace=None) -> span` (§8.6) |
| a record sink | `emit(span)`, `flush()`, `close()` (§8.2); `hone_models.testing.check_record_sink` checks one |
| `mk.machine.Machine()`, `mk.machine.MACHINE` | `snapshot() -> dict`, `prepare(needed, *, if_busy="block") -> dict`, `load(model_id) -> dict`, with the keys in §7 (hone-select's `MachineProbe`) |

Entry points, so tools can load these by name without importing hone-models:

| Group | Name | Object |
|---|---|---|
| `hone.text_clients` | `hone_models` | `hone_models:text` (takes a model id) |
| `hone.decision_clients` | `hone_models` | `hone_models:decision` (takes a model id) |
| `hone.embedders` | `hone_models` | `hone_models:embedder` (takes a model id) |
| `hone.gpu_leases` | `hone_models` | `hone_models.gpu:GPU` |
| `hone.replayers` | `hone_models` | `hone_models.replay:Replayer` (no arguments) |
| `hone.machine_probes` | `hone_models` | `hone_models.machine:Machine` (no arguments: the default probe) |

The contract checks for these shapes run in the test suite against the real implementations with fake
transports (AC-17).

## 11. Guarantees (acceptance cases)

Each case has a test in `tests/e2e/` named `test_ac<N>_*`; cases marked **[real]** also run against real
models in `tests/gpu/`.

| AC | Scenario | Expected |
|---|---|---|
| AC-1 | `mk.text("ollama:<model>")` chat against a mock Ollama | text, usage, finish_reason; span recorded with `gen_ai.*` names |
| AC-2 | A thinking model returns empty `content` and filled `thinking` | `think=false` sent per capability; if content is still empty, `error` is set, never a silent empty string |
| AC-3 | Structured output: fenced JSON; truncated JSON; invalid, then valid on retry | `structured_path` = parsed / repaired / retried; validation errors sent back on retry |
| AC-4 | Prompt + output exceeds the context | `ContextOverflow` before any HTTP call; the message includes the numbers |
| AC-5 | Image sent to a non-vision model | `CapabilityError` before calling |
| AC-6 | `require={"vision": True}` with `prefer="local"` | picks the local vision model; no match raises an error listing candidates |
| AC-7 | `mk.Prompt` with sections | rendered text correct; section ids, versions and character spans recorded and consistent with the text |
| AC-8 | Decision emulation with a fake transport | answer normalization (§6); per-question errors; contract check passes |
| AC-9 | Jev provider with recorded fixtures | mapping to answers; `calibrated=True`; contract check passes |
| AC-10 | OpenAI-compatible provider with logprobs | calibrated yes/no from token logprobs |
| AC-11 | Transient 429, then success; a 400 | the 429 is retried and recorded as a span event; the 400 is not retried |
| AC-12 | GPU lease: two processes contend (fake NVML reporting 8 GB) | the second waits until the first releases; a timeout raises `TimeoutError`; the ledger drops dead PIDs |
| AC-13 | Secrets | a planted API key never appears in the SQLite file |
| AC-14 | Content capture off | messages stored as hashes and lengths only |
| AC-15 | Replay with a section removed | new span with `replay_of`; the request lacks the section; other params identical |
| AC-16 | Registry merge, ad-hoc ids, `models list/show` CLI | merged view; `--json` output |
| AC-17 | Contract checks for the text client, decision client, embedder, GPU lease, replayer, record sink and machine probe | all pass with fake transports |
| AC-18 **[real]** | Ollama: chat, JSON schema and decision emulation on a text model; a vision model on a generated PNG; embeddings; think handling on a thinking model | all succeed under the GPU lock; contract checks pass; models unloaded afterwards |
| AC-19 **[real]** | GPU lease with real NVML | status reports real memory; the lease / unload cycle works |
| AC-20 | Examples | every `examples/*.py` runs offline, opens with a What / How / Why docstring, uses only the public API and is listed in `examples/README.md` |
| AC-21 **[real]** | Speech: a minutes-long script with `FakeSpeech`; one sentence with `kokoro-82m` | one WAV of the right length and one `hone.models.speech` span, text hashed with capture off; the real WAV is non-silent and the GPU is freed afterwards |
| AC-22 **[real]** | Expressive speech: paragraphs with an emotion and intensity each in one session (`FakeSpeech(expressive=True)`); one line calm and excited with `chatterbox` in one session | the span records emotion, intensity, `expressive`, `session` and paragraph count, a paragraph that fits is one chunk, and the session loads the model once; the real takes are non-silent, the excited one varies more in pitch and is louder and higher, and the GPU is freed afterwards |
| AC-23 | A lease inside another in the same thread; a lease no one can grant (fake memory) | the nested lease reserves only what the outer one does not cover and never waits for it; the impossible one raises `CapabilityError` after `stall_s`, naming the holders |
| AC-32 **[real]** | `mk.machine.snapshot()` with a fake NVML (8 GB, one foreign process), FakeOllama with a model partly on the CPU, ComfyUI (fake) after a job, after `/free`, after a job hone-models did not run, and refused; no NVML and no `nvidia-smi`; Ollama answering 500; the lock held by another process, by our own `gpu-lock.sh`, and free | GB values and `vram_gb < size_gb` as scripted; ComfyUI named by registry id, dropped after `/free`, unnamed for a foreign job; a registry without `comfyui` entries checks none; `gpus` is `None`; Ollama `running: None` with an error and no entries; `held` / `mine` / `holder` right; the real card is reported |
| AC-33 **[real]** | `prepare(["a"])` with Ollama holding `a` and `b` and ComfyUI a model not needed; with ComfyUI holding only needed models; with another process's lease (`if_busy` `"block"` and `"unload"`); with the lock held elsewhere; with the unload of `b` failing | `b` unloaded, ComfyUI freed, span recorded; ComfyUI kept; blocked: nothing sent, `blocked_by` names the holder; `"unload"`: `b` unloaded, `blocked_by` still reported; failure: in `errors`, span status `error`, `b` still loaded; on the real machine `prepare([])` leaves nothing loaded |
| AC-34 | `load("a")` on FakeOllama; a model partly on the CPU; a server timing out; a `comfyui` entry; `GpuScheduler(unload_others=True, if_busy="block")` short of memory while another process holds a lease | an empty-prompt `/api/generate` with `keep_alive`, `loaded: True` with seconds; `vram_gb < size_gb`; `loaded: False` with the error; `loaded: None`, "not supported", no request; the other process's models are not unloaded and the lease times out as before |

## 12. Examples

`examples/` holds one runnable file per public concept, indexed in reading order in
`examples/README.md` with the section of this document it illustrates. Each file opens with a docstring
saying **what** it shows, **how** (the calls, in order) and **why** (the problem it solves), runs top to
bottom against the packaged `FakeOllama` (no network, no GPU), prints a few lines and asserts the key
facts. The examples are the reference patterns to copy, so they use only the public API (§2).

`hone_models.testing.FakeOllama` is a local HTTP server with Ollama's API (and its OpenAI-compatible
`/v1` endpoints) that sets `OLLAMA_HOST` while it runs. Chat replies echo the last user message, or a
sample object matching the schema; embeddings are stable hash vectors. `server.queue(path, *answers)`
scripts the next answers for one path (a dict is merged over the default reply, an int is an HTTP error
status); `FakeOllama(responder=fn)` computes answers from the request, `fn(path, body)` returning a dict
merged over the default reply or `None` ([0007](changes/0007-fake-ollama-responder.md)); queued answers
come first; `server.requests` records each request's path, body and headers; `start()` / `stop()` do what
the `with` block does.

`hone_models.testing.FakeSpeech` stands in for `mk.speech(...)`: same API, splitting and span, no model and
no GPU lease; it writes silence of about `words / 2.5` seconds and keeps each call in `calls` as
`(text, voice, speed, emotion, intensity)`; `FakeSpeech(expressive=True)` stands in for an expressive model
(Kokoro's voices plus the `chatterbox` entry's); `FakeSpeech.like(model_id)` copies a registry entry's id,
voices, expressiveness and chunk size ([0012](changes/0012-fake-speech-expressive-voices.md));
`loads` / `frees` count what the real client would load and free.

## 13. Known limits of 0.1.0

- The Jev request / response mapping is unverified against the real API; it is tested only against
  recorded fixtures ([decisions.md](decisions.md), D-019).
- `deepseek-r1:8b` on Ollama can return only thinking text even with `think=false` when `max_tokens` is
  very small. hone-models reports it as `result.error`, but the hint in that message ("pass
  think=False") is then not the fix; raising `max_tokens` is.
- Model calls do not take GPU leases automatically (D-009, awaiting owner review).
- Sessions and unloading are Ollama-only.
- `mk.machine` sees only `flock` users of the lock file and lease holders; a process using the GPU with
  neither shows only in `gpus[].processes` and does not block `prepare`. Between `prepare`'s read and its
  unloads another process can load a model or take a lease (the state after is read again).
  `FileLockGpuLease` writes no `.holder` file, so a Python holder of the lock shows as `holder: None`.
- No streaming, async clients, `hone.models.timing.*` attributes or media generation other than speech.
- The `speech` extra (Kokoro) needs Python < 3.13 and loads the model on every call (about a second)
  outside a session.
- The `expressive` extra (Chatterbox) needs Python < 3.13, uv overrides of its exact torch / numpy pins
  (D-021) and loads about 3 GB on every call outside a session; emotion is per call, not per sentence.
