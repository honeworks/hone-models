# Decisions

Implementation choices too small for a change record, made while building hone-models where the design
was silent or ambiguous. Larger choices are explained in the change records; entries here that belong
to one say so. Each entry keeps its original number, which commit messages and code comments refer to.

**Review status.** Every entry is marked either "no review needed" (recorded for information) or
"awaiting owner review" (a human still has to decide). When the owner decides, the entry gets a line
saying what was decided.

## Awaiting owner review

| Entry | Question for the owner |
|---|---|
| [D-009](#d-009-no-automatic-gpu-lease-around-model-calls) | Should local model calls lease the GPU automatically in a later version (using the registry's `vram_gb`)? |
| [D-016](#d-016-sibling-packages-as-development-only-path-sources) | Remove the `[tool.uv.sources]` path sources once the sibling packages are published, before hone-models is published. |
| [D-019](#d-019-the-jev-api-mapping-is-unverified) | Verify the Jev API mapping against the real API (endpoint, auth, field names, token counts). |
| [D-022](#d-022-the-packaged-gemma4-12b-entry-and-its-ollama-tag) | Before publishing, point the packaged `gemma4-12b` entry at the public Ollama tag `gemma4:12b` (and keep the local tag through a user registry alias). |

## D-001: `records.py` is a public module
The span sinks live in one public module, `hone_models.records` (with `read_spans`, `add_secret` and
`default_store`), rather than a private module plus a public re-export layer that would only forward
names. No review needed.

## D-002: an undeclared capability is unknown, not false
All capability fields default to `None`. Selection with `require=` matches only declared values;
pre-call checks (vision, context size) refuse only when the registry *declares* the model cannot do it.
Ollama ad-hoc ids are probed through `/api/show`, so they are usually known. Blocking unknown models
would make ad-hoc hosted ids unusable for images. Part of [0001](changes/0001-initial-design.md).
No review needed.

## D-003: packaged registry defaults
The packaged registry ships `gemma4-12b`, `qwen2.5vl-7b`, `nomic-embed-text`, `jev`, `gpt-4.1-mini` and
`deepseek-r1-8b` (a thinking model). There is no placeholder LiteLLM entry: LiteLLM models are reachable
ad hoc as `litellm:<model>`. Defaults should be real, useful entries. No review needed.

## D-004: `require=` and `prefer=` keys
`require=` accepts `min_context` (`max_input_tokens >= n`) and any capability name (exact match);
unknown keys raise `ConfigError`. Without `prefer=` candidates are ordered by id; `cheapest` treats
local models as cost 0 and unknown hosted prices as infinite; `fastest` uses `speed_tok_s` (unknown
last). No review needed.

## D-005: the CLI is an extra
typer and rich are in the `cli` extra; without it the `hone-models` script prints an install hint. The
core stays standard library + pydantic + httpx. `--registry PATH` (repeatable) adds registry files to
every registry command. No review needed.

## D-006: the schema instruction is always added; what `constrained` means
With a schema, the schema is always appended to the system message (Ollama's documentation recommends
it even with `format`); the provider constraint (`format` / `response_format`) is added only when the
model declares `json_schema`. Path `constrained` means the constraint was sent and the raw text parsed
without cleanup; `parsed` means fences or prose had to be stripped. One code path for every provider.
Part of [0001](changes/0001-initial-design.md). No review needed.

## D-007: output budget, `num_ctx` rounding, empty replies
The requested output is the `max_tokens` param, else the model's `max_output_tokens`, else 2048, and it
is always sent (`num_predict`), so truncation is reported as finish reason `length` instead of happening
silently. `num_ctx` = (prompt + output) × 1.1, rounded up to a multiple of 2048 (Ollama reloads a model
whenever `num_ctx` changes), capped at `max_input_tokens`. Images are not counted in the estimate. An
empty reply sets `result.error` (naming `thinking` when only thinking text came back). Validation
retries are `structured_retry` span events; `usage` sums all attempts. Part of
[0001](changes/0001-initial-design.md). No review needed.

## D-008: extra span attributes for replay
Replay must rebuild a request with every other parameter identical, but the standard GenAI attributes
cover only temperature, top_p, max_tokens and seed. Chat spans therefore also carry
`hone.models.request.params` (all call params) and `hone.models.structured.schema`; decide spans carry
`hone.models.decision.state`. The state, questions and answers count as content (hashed when capture is
off). Part of [0001](changes/0001-initial-design.md). No review needed.

## D-009: no automatic GPU lease around model calls
`mk.text(...).complete` does not take a GPU lease itself; callers wrap GPU work in `mk.gpu.lease(...)`,
and spans made inside carry `hone.models.gpu.*`. The client API has no GPU option, and an implicit lease
would block calls on machines without NVML. `hone.models.timing.*` is not recorded either (no streaming,
so no time to first token). Part of [0001](changes/0001-initial-design.md).
**Awaiting owner review:** should local model calls lease automatically in a later version (using the
registry's `vram_gb`)? If yes, it becomes a new change record.

## D-010: calibrated yes/no lives in decision emulation
Token logprobs are exposed on `TextResult.logprobs`; turning them into a calibrated P(yes) happens in one
place, the decision emulation, so AC-10 is tested through `mk.decision` on an OpenAI-compatible model
with `logprobs = true`. No review needed.

## D-011: GPU lease details
- Free memory = `total - max(used, reserved)`: a lease counts before its model loads, and other programs
  count too.
- Without GPU information (no NVML, no `nvidia-smi`) leases are granted at once but still written to the
  ledger. A lease larger than the whole GPU raises `CapabilityError` at once instead of waiting.
- When memory is short, idle Ollama models are unloaded once per lease attempt, then the lease polls; no
  unload is attempted once `timeout_s` has passed. "Idle" means loaded by this process through
  hone-models, whatever it is doing now; models on remote Ollama servers are unloaded too.
- `unload_others` is a `GpuScheduler(unload_others=True)` argument; it also unloads what `/api/ps` lists
  on the default server.
- Spans inside a lease carry `lease_wait_ms`, `vram_before_mb` (when known) and `unloaded`;
  `vram_after_mb` is not recorded because those spans end before the lease does. A lease records no span
  of its own and does not change the trace context.
- Reentrancy is keyed by (pid, thread, name). The ledger lives under `HONE_HOME`; processes with
  different `HONE_HOME`s don't share reservations, but NVML's `used` still counts their models.
  Malformed ledger entries are dropped.

No review needed.

## D-012: sessions and unload are Ollama-only
`mk.session(provider)` raises `ConfigError` for anything but `"ollama"`. It checks `GET /api/version`,
else starts `ollama serve` (clean environment, `OLLAMA_HOST` set to the URL it waits on), waits up to
30 s, yields the server URL, and on exit terminates only what it started. `mk.unload` raises
`ConfigError` for non-Ollama models: there is no standard unload API elsewhere. Part of
[0001](changes/0001-initial-design.md). No review needed.

## D-013: prompt variables, decision content and invalid provider answers
`{name}` placeholders without a matching variable are left as they are (prompts often contain literal
braces in format examples). `hone.models.decision.questions` counts as content. Jev answers with a
probability outside [0, 1] (or NaN), or choice probabilities over unknown options or summing above 1,
become that question's `error`. Malformed embedding responses raise `ProviderError`; partial token
logprob entries are skipped. No review needed.

## D-014: replay and `calls` CLI details
- `Replayer.replay_call` replays chat spans only (spans with `gen_ai.input.messages`), needs content
  capture (`ConfigError` otherwise) and rejects unknown override keys. `messages` and `prompt.sections`
  overrides are exclusive.
- Section overrides rebuild the `mk.Prompt` from the recorded section spans (roles and versions kept);
  replacement texts are used as given (no variable filling); rebuilt prompts record no variables.
- Spans from other recorders fall back to `gen_ai.provider.name` + `gen_ai.request.model` (as an ad-hoc
  id) and `gen_ai.request.*` params. Replayed params include the original model's registry defaults.
- The new span is returned and also written to the replayer's sink, following that sink's
  content-capture setting.
- `calls` commands read `$HONE_HOME/models/spans.db` (or `--db`); `--by tag` groups by `hone.step`, the
  only tag-like attribute a model call carries; each model request counts once (an emulated decide span
  is represented by its chat child).
- `models check` sends "Say OK." with `max_tokens=64`; tokens/s = output tokens / wall time (including
  model load time); it rewrites the user registry file, keeping other entries but not comments.

No review needed.

## D-015: LiteLLM provider details
LiteLLM is called with the OpenAI-shaped body shared with the OpenAI-compatible provider. Its own
`num_retries=2` handles transient errors, so those retries are not `retry` span events. Errors map to
`ModelTimeout` / `ProviderError` (with the status); `base_url` is sent as `api_base`. Capability hints
from LiteLLM's model map apply to ad-hoc `litellm:<model>` ids only. `gen_ai.provider.name` is
`"litellm"`. `think` is Ollama-only. No review needed.

## D-016: sibling packages as development-only sources
The contract tests import the checks from the packages that consume hone-models' shapes (`hone-select`,
`hone-flow`, `hone-lens`), which are not on PyPI yet. The `dev` extra requires them by version, and
`[tool.uv.sources]` in `pyproject.toml` resolves them to sibling folders (`../hone-select`, ...). The
published wheel carries only the version requirements; the path sources affect local development only.
Part of [0002](changes/0002-replay-entry-point-and-foreign-spans.md).
Update before publishing: the development tools moved from the `dev` extra to a `dev` dependency group
(PEP 735), which is never published, so the published metadata no longer names the siblings; the sources
now point at the siblings' GitHub repositories instead of local folders.
**Awaiting owner review:** remove the `[tool.uv.sources]` table once the siblings are on PyPI.

## D-017: examples run on the packaged `FakeOllama`, scripted with `queue`
Examples use only the public API, so the public fake was extended rather than patched: `queue`,
`start` / `stop`, the `/v1` endpoints and request headers. Each example starts its own fake; the example
test runs them with `OLLAMA_HOST` on a closed port, because an example that read `OLLAMA_HOST` once
reached a real local Ollama. Part of [0003](changes/0003-examples-and-scriptable-fake-ollama.md).
No review needed.

## D-018: public names the examples rely on
Every name an example uses is public and documented: `mk.records.read_spans`, `default_store`,
`add_secret` ([records and replay](../docs/records-and-replay.md)), the `hone_models.calls` module
([CLI](../docs/cli.md)) and `mk.gpu.GpuScheduler(ledger, memory=...)`
([GPU and sessions](../docs/gpu-and-sessions.md)). `FakeOllama` requests also record `headers`. The
example test allows only `hone_models`, `hone_models.testing` and `hone_models.calls` imports and `mk.*`
names in `__all__`. Part of [0003](changes/0003-examples-and-scriptable-fake-ollama.md).
No review needed.

## D-019: the Jev API mapping is unverified
The Jev provider's request and response shapes were reconstructed from public articles (TypeSafe AI,
September 2026) and are tested only against recorded fixtures, behind a small mapper in
`providers/jev.py`. **Awaiting owner review:** verify the mapping against the real API (endpoint, auth,
field names, token counts) and update the mapper and fixtures.

## D-020: Kokoro through its own package, loaded per call
Kokoro-82M is used through the `kokoro` package (torch), not an ONNX port, because it downloads its
weights and voices from Hugging Face by itself and runs on CUDA when present. It installs without system
packages: espeak-ng comes as a library inside the `espeakng-loader` wheel. The spaCy model
`en_core_web_sm` cannot be a dependency of a PyPI package (a direct URL), so users install it themselves
(docs/speech.md) and `synthesize` checks for it before the GPU lease; this repository gets it from the
unpublished `dev` dependency group ([0008](changes/0008-speech-extra-and-the-spacy-model.md)). The model is loaded for each `synthesize` call
and freed afterwards: about a second per call, in exchange for never holding memory on the shared GPU.
`[tool.uv] constraint-dependencies` keeps local development within the torch range the speech extras
are tested on; it does not affect the published metadata. Part of [0006](changes/0006-speech.md).
No review needed.

## D-021: Chatterbox on the machine's torch, through uv overrides
`chatterbox-tts` 0.1.7 pins `torch==2.6.0`, `torchaudio==2.6.0` and `numpy<2` (Python < 3.13). Its code
runs unchanged on torch / torchaudio 2.10 and numpy 2 (checked by the `gpu` test), so the extra requires
it unpinned and `[tool.uv] override-dependencies` replaces the three pins, keeping one torch build for
both speech extras. Overrides only apply in the root project: apps copy the line (docs/speech.md). The
extra also pins `setuptools<81`, because Chatterbox's watermarker (resemble-perth) imports
`pkg_resources` and silently disables itself otherwise, after which Chatterbox fails to construct. The
voices are reference clips rendered by Kokoro (Apache-2.0, synthetic voices) with
`scripts/make-expressive-voices.py` and shipped as FLAC (about 10 s each); no real person's recording is
used. Part of [0011](changes/0011-expressive-speech.md).
No review needed.

## D-022: the packaged `gemma4-12b` entry and its Ollama tag
The packaged registry maps `gemma4-12b` to the Ollama tag `gemma4-12b:latest`, a tag created locally
while hone-models was built; the public Ollama library calls the same model `gemma4:12b`. Until the
switch, the docs tell users to pull `gemma4:12b` and point the entry at it with a user registry file
(`~/.config/hone/models.toml`: `[models."gemma4-12b"] model = "gemma4:12b"`). The switch is not made
yet because the applications built on this machine resolve `gemma4-12b` through the packaged entry.
**Awaiting owner review:** before publishing, change `model` to `gemma4:12b` in
`src/hone_models/data/models.toml`, and keep the local tag on the development machine with a user
registry alias (`[models."gemma4-12b"] model = "gemma4-12b:latest"`).

## D-023: details of `mk.machine` the record left open
Choices made while building [0016](changes/0016-machine-state.md), each the simplest reading of it:
- A lock file that does not exist is not held (`held: False`); only a file that exists but cannot be
  opened is unknown (`held: None`). `held: None` does not block `prepare`, since nobody is known to hold
  the lock. The probe never creates the file.
- Per-process GPU memory comes from the existing GPU 0 reader, so `processes` is filled for GPU 0 and
  `None` (unknown) for other GPUs.
- An unnamed ComfyUI entry has `name: None` and `model_id: None`. It is added when the newest `/history`
  job is not in the loaded-models file, or when the server holds torch memory and the file lists nothing.
  A server with no torch memory lists nothing, whatever the file says.
- `released` lists server names (`"comfyui"`), as in the record, not URLs.
- `missing` lists only needed Ollama and ComfyUI ids: hosted and in-process models are never "loaded"
  on a server, so listing them would always mark them missing.
- `need_gb` sums the `vram_gb` of the needed entries that use this GPU (`local`, `comfyui`, `command`);
  hosted entries count 0.
- `load` sends the entry's `defaults.keep_alive`, else `"5m"` (Ollama's own default, which every chat
  call resets to anyway), and notes the model as loaded by this process, so a short lease may unload it.
- `GpuScheduler(if_busy="block")` checks the lock at `HONE_GPU_LOCK` (else `/tmp/honeworks-gpu.lock`),
  only when it is about to unload other processes' models.
No review needed.
