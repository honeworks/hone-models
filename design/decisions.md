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

## D-024: ComfyUI workflows are read on every call; uploads are remembered per process
0015 §2 says a workflow is read once per client. It is read on every call instead (a few kilobytes):
simpler, and a workflow file changed between two calls is then both used and visible in the records
through `hone.models.media.workflow_sha256`. A file input is uploaded once per process and server
(`input/hone/<sha256><suffix>`, remembered in memory); a new process uploads it again, which ComfyUI
stores over the same name (`overwrite=true`). Part of [0015](changes/0015-generation-models.md).
No review needed.

## D-025: no progress events for ComfyUI jobs
0015 §2 expected progress from `/api/jobs/{id}` "where the server has it". ComfyUI 0.3x's jobs API
reports a status (`pending`, `in_progress`, `completed`, ...) but no progress; progress only goes over the
websocket, which hone-models does not use (no new dependency). So ComfyUI spans carry no `progress`
events; `hone.models.media.queue_wait_ms` comes from the `execution_start` time in the job's history.
`/api/jobs/{id}/cancel` is used to cancel. Part of [0015](changes/0015-generation-models.md).
No review needed.

## D-026: file inputs given as strings, the unknown-memory lease, the error text as content
For `references`, `image` and `source` (inputs that are always files) a string is taken as a path, so
`image="shots/01.png"` works like `Path(...)`; for other inputs only a `pathlib.Path` is a file. A local
model without `capabilities.vram_gb` leases 1 GB, as speech does. `hone.models.media.error` is content
(hashed with capture off), like span status messages, because a provider's error can quote the prompt;
`hone.models.media.inputs` hashes only its text values, so numbers and file records stay readable. Part
of [0015](changes/0015-generation-models.md).
No review needed.

## D-027: the `install` table's fields for projects
0015 §3b names what `install` says (an Ollama name, Hugging Face files with their ComfyUI folder, "a
project to clone and set up", the size, a tier, a note) without field names for projects. The registry
accepts `repo` (to clone), `setup` (commands), `dir_env` (the folder variable) and `check` (a command that
succeeds when the project is set up), next to `source`, `ollama`, `files` (`repo`, `file`, `to`),
`size_gb`, `tier` (1, 2 or 3) and `note`; other keys are an error. The catalog step may add fields.
Part of [0015](changes/0015-generation-models.md).
No review needed.

## D-050: transcription details the record left open
The span also records `hone.models.transcribe.prompt` (content, only when given), because Whisper's
`initial_prompt` changes what is heard, and `session` / `loaded` as speech spans do.
`hone.models.transcribe.language` is set to the language asked for when the span opens and to the
language of the transcript when it ends, so a failed call still shows what was asked. `Transcript.text`
joins the segments' texts with a space; segment and word texts are stripped of faster-whisper's leading
spaces; times are rounded to milliseconds. `timeout_s` cannot interrupt CTranslate2 mid-segment, so
decoding stops between segments (faster-whisper decodes while its segment generator is read) and
`ModelTimeout` is raised; a segment covers at most 30 s of audio. `FakeTranscriber` hears a scripted
`text` (one segment per line, words 0.4 s apart) rather than reading the audio. The user guide is its own
page, `docs/transcription.md`. Part of [0015](changes/0015-generation-models.md).
No review needed.

## D-051: how faster-whisper finds the CUDA 12 cuBLAS and cuDNN 9 libraries
CTranslate2's Linux wheel does not link cuBLAS or cuDNN; it `dlopen`s `libcublas.so.12` and
`libcudnn.so.9` by name the first time a model runs on CUDA, and fails in the middle of the call when
they are missing (this machine has only CUDA 13 system-wide). Before the GPU lease, when the entry runs
on CUDA (`defaults.device`, or `auto` with `ctranslate2.get_cuda_device_count() > 0`), the provider loads
each library with `ctypes.CDLL(..., RTLD_GLOBAL)`: first by name (the system library path,
`LD_LIBRARY_PATH`, or a library torch already loaded), then from `nvidia/cublas/lib` and
`nvidia/cudnn/lib` under each `sys.path` entry, where the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`
wheels put them (they carry `RUNPATH=$ORIGIN`, so cuBLASLt and cuDNN's sub-libraries load next to them).
A library loaded this way satisfies CTranslate2's later `dlopen` by name. When neither place has one,
`ConfigError` names the missing libraries and the wheels to install, or `defaults.device = "cpu"`. The
wheels are not in the `transcribe` extra: they are Linux-only, large, and already present wherever a
CUDA 12 torch is installed (the speech extras); as with Kokoro's spaCy model in
[0008](changes/0008-speech-extra-and-the-spacy-model.md), the user installs them and the call checks
first. Found libraries are remembered for the process. Part of [0015](changes/0015-generation-models.md).

## D-030: hosted image and video requests: the inputs each kind takes, no `response_format`, no seed
0015 §2 lists the request fields of the hosted images and video calls. The `openai_compatible` media row
takes, besides `prompt` and `seed`, only what those endpoints use: images `size`, `n`, `references`;
video `size`, `duration_s` (sent as the string `seconds`), `image` (multipart `input_reference`); plus
the names an entry lists in `inputs`, sent as request fields. The rest of the shared vocabulary
(`negative`, `steps`, `lyrics`, ...) raises `ConfigError` as for any model that does not take it. The
record says to send `response_format = "b64_json"`; OpenAI's gpt-image models always answer in base64
and do not take that parameter, while other models and gateways default to `url`. Both answers are
handled (decoded, or downloaded without the API key), so nothing is sent by default; an entry that wants
one lists `response_format` in `inputs` and sets it in `defaults`. The seed is recorded but not sent:
the endpoints have no seed field. An `openai_compatible` media entry on this host (a local server) takes
the GPU lease like any local model. Part of [0015](changes/0015-generation-models.md).
No review needed.

## D-031: the per-second cost uses the asked-for duration when the file's cannot be read
`cost_usd` for a `per_output_second` price multiplies the measured seconds of output. A video's duration
is read with `ffprobe`, which may not be installed; the hosted API bills the seconds asked for anyway.
So a file whose duration cannot be read counts the call's `duration_s` (including the entry's default);
without either the cost stays `None`. Still the naive estimate of 0015 (`cost_estimated = True`). Part
of [0015](changes/0015-generation-models.md).
No review needed.

## D-032: hosted refusals, polling failures and signed URLs
A moderation refusal at submission comes back from OpenAI as HTTP 400 (`moderation_blocked`); any 4xx
whose text reads as a refusal (`error_kind` gives `refused`) becomes a result with `error_kind =
"refused"` and no job id; other 4xx raise `ProviderError` as today. A failed video job's `error.code`
and `error.message` are joined into `result.error`, and its kind is read from that text, so a
moderation failure is `refused` too. Polling counts transport errors, 429, 5xx and unreadable JSON as
transient; the fifth in a row raises `ProviderError` naming the job id (as ComfyUI's polling does);
another 4xx raises at once. Either way, and on a timeout or interrupt, the job is deleted. Downloaded
image URLs are signed; they are never put in an error message or the records. Part of
[0015](changes/0015-generation-models.md).
No review needed.

## D-040: the LeVo adapter runs the project's generation in its own process, not through `generate.sh`
The project's `generate.py`, run as a script, reseeds NumPy from the clock and accepts only the v1
checkpoint folder names (`songgeneration_base`, ..., `songgeneration_large`), so `songgeneration_v2_medium`
fails its assertion, and seeds set in another process would not reach it. `levo2.py` therefore does what
`generate.sh` and `generate.py`'s `__main__` do, in its own process: the same environment variables and
`sys.path`, the working folder, cuDNN off, the four OmegaConf resolvers, then `generate_lowmem(args)` (or
`generate(args)` when `low_mem` is off and more than 24 GB, 36 GB for a `large` checkpoint, are free, as
the script decides), after seeding Python, NumPy and torch with the call's seed. The checkpoint folder is
`defaults.checkpoint` (default `songgeneration_v2_medium`); `defaults.flash_attn` defaults to off (flash
attention is not installed in the project's environment here, as `--not_use_flash_attn` in the old
`levo2_generate.sh` shows). A request LeVo cannot take (no lyrics, an unknown `generate_type`) and an
exception during the generation (out of memory, no CUDA) are written to `result.json` as the error; a
failure to import the project crashes the adapter, so it surfaces as a `ProviderError` with the traceback.
The adapter takes `lyrics` as given (already in LeVo's form, §3a) and uses LeVo's `descriptions` for the
prompt; LeVo's `prompt_audio_path` and `auto_prompt_audio_type` are not wired yet. Not run against the
real project here (no GPU for this work); the first real run is a `gpu` test run by hand (0015 §9). Part of
[0015](changes/0015-generation-models.md).
No review needed.

## D-041: details of the `command` protocol
0015 §2 fixes the protocol's shape; the details: `request.json` also carries the entry's whole `defaults`
table, for options that are not inputs (LeVo's `low_mem`), next to `model`, `prompt`, `seed`, `out_dir`
and `inputs`. The job folder is `<out name>.job-<id>/` next to `out`, with `request.json`,
`stdout.log`, `stderr.log` and `out/` (the `out_dir`); the program's output goes to those files, never to
pipes, and the folder is removed after a success and kept after a failure. `cwd` defaults to the
`install.dir_env` folder. A program that cannot be found (after expansion, relative paths from `cwd`) is
a `ConfigError` before the lease. An `error` in `result.json` wins over the exit code; without
`result.json` the files are every file directly in `out_dir`, in name order. `log_tail` keeps what a
terminal would show (a `\r` rewrites its line, so progress bars take one line). `result.json`'s `meta` is
read but not recorded: §5 has no attribute for it. The grace between SIGTERM and SIGKILL is
`providers.command.KILL_GRACE_S` (10 s). Part of [0015](changes/0015-generation-models.md).
No review needed.
