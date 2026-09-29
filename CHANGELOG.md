# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Why the design changed is recorded in
[design/changes/](design/changes/).

## [0.1.0] - unreleased

First release. Design: [0001 initial design](design/changes/0001-initial-design.md),
[0002 replay entry point and spans from other recorders](design/changes/0002-replay-entry-point-and-foreign-spans.md),
[0003 examples set and a scriptable FakeOllama](design/changes/0003-examples-and-scriptable-fake-ollama.md).

### Changed
- `gemma4-12b`'s packaged licence is Apache-2.0 (Gemma 4's model card), not "Gemma Terms of Use". The
  machine snapshot checks only ComfyUI servers of entries that have a workflow
  ([decisions.md](design/decisions.md) D-066, D-067).

### Added
- The catalog and model guides: the packaged registry is split by kind
  (`hone_models/data/models/<kind>.toml`) and lists every model in use or worth trying, installed or not,
  each with `install` (Ollama name, Hugging Face files with their ComfyUI folder, a Hugging Face repository,
  or a project to clone and set up; size, tier, note), licence, `commercial_use` and a `guide`.
  `mk.guide(id)` returns a `ModelGuide` (summary, prompt advice, every accepted input with its note,
  features with examples and a source, limits, licence, install state; `as_text()`, `as_dict()`; entry
  point `hone.model_guides`); `capabilities.features` comes from the guide and
  `require={"features": [...]}` / `mk.select(...)` match models declaring every listed feature. Common
  input formats: `lyrics` in one format converted by the entry's `lyrics_format` (`sections`, `levo`,
  `plain`) and `prompt_inputs` written into the prompt (an unknown choice lists the choices).
  `mk.catalog.installed(cfg)` (`yes` / `no` / `unknown`); a generation call to a model that is not
  installed fails before any job; kind `scoring` entries (provider `none`) cannot be called yet. CLI:
  `models guide <id> [--json] [--stale DAYS]`, `models install <id>` (prints the commands, size and free
  disk; `--run`), and `models list` with an `installed` column and `--kind`, `--feature`, `--installed`,
  `--missing`, `--tier` ([0015](design/changes/0015-generation-models.md) §3a, §3b).
- `mk.machine`: `snapshot()` reports every GPU (name, memory, utilization, processes), the Ollama and
  ComfyUI servers and the models they hold (partly on the CPU or not), the machine-wide GPU lock and the
  leases, with `None` for anything unknown; `prepare(needed, if_busy=...)` unloads every model not needed
  unless another process is using the GPU; `load(model_id)` warms an Ollama model up. `Machine` is
  hone-select's `MachineProbe` (entry point `hone.machine_probes`); spans `hone.models.machine.prepare` /
  `.load`. `GpuScheduler(if_busy="block")` leaves other processes' models alone while they use the GPU
  ([0016](design/changes/0016-machine-state.md)).
- Images, music and video: `mk.image`, `mk.music`, `mk.video` return a `MediaClient` whose
  `generate(prompt, *, out, seed=None, timeout_s=None, trace=None, **inputs)` returns a `MediaResult`
  (measured `MediaFile`s, seed, `error` / `error_kind`, job id, naive cost, license, `commercial_use`);
  unknown inputs, missing files and undeclared sizes or durations fail before any request; one
  `hone.models.image|music|video` span per call; a GPU lease per call or per `session()`. The `comfyui`
  provider runs one API-format workflow per registry entry (inputs mapped onto node paths, files uploaded
  once by hash, jobs cancelled on timeout or interrupt, `/free` after a plain call or once per session);
  `mk.session("comfyui")` starts ComfyUI from `HONE_COMFYUI_START` and `mk.unload` frees it. Registry
  kinds `image` / `music` / `video` / `transcription`, generation keys and capabilities, `per_image` /
  `per_output_second` prices; `FakeComfyUI` and `FakeMedia` for tests; entry points
  `hone.image_clients`, `hone.music_clients`, `hone.video_clients`
  ([0015](design/changes/0015-generation-models.md), step 1).
- `THIRD_PARTY_NOTICES.md` (the Kokoro-rendered voice clips; copyleft dependencies of the optional
  speech extras), shipped in the wheel and sdist.
- The context budget counts images (capabilities `image_tokens` / `image_patch_px`, else 1024 per
  image; sizes from the file header), records `hone.models.context.estimated_image_tokens`, and takes a
  `min_num_ctx` floor as a param or registry default
  ([0013](design/changes/0013-context-budget-counts-images.md)).
- `FakeSpeech.like(model_id)`: a fake with a registry entry's id, voices, expressiveness and chunk size;
  `FakeSpeech(expressive=True)` also accepts the Chatterbox voices
  ([0012](design/changes/0012-fake-speech-expressive-voices.md)).
- A local model without a measured speed is assumed to make 10 tokens/s, so long answers get a timeout
  that grows with `max_tokens` (up to `max_timeout_s`) on a fresh machine
  ([0010](design/changes/0010-timeout-without-measured-speed.md)).
- Speech timings: `SpeechResult.segments` (`SpeechSegment(text, start_s, end_s)` per chunk) and
  `synthesize(..., by_sentence=True)` for one segment per sentence
  ([0009](design/changes/0009-speech-sessions-and-timings.md)).
- Kokoro's English voices check for spaCy's `en_core_web_sm` before the GPU lease and say how to install
  it; docs/speech.md gives the line ([0008](design/changes/0008-speech-extra-and-the-spacy-model.md)).
- `FakeOllama(responder=fn)`: answers computed from the request (by schema title, by prompt), for tests
  whose number or order of calls varies ([0007](design/changes/0007-fake-ollama-responder.md)).
- Release hooks for GPU users other than Ollama: `mk.gpu.on_short(name, release)`, called when a lease is
  short of memory, with `mk.gpu.comfyui_free(url)` and `mk.gpu.torch_empty_cache`; spans record
  `hone.models.gpu.released` ([0005](design/changes/0005-release-hooks-for-other-gpu-servers.md)).
- GPU leases no longer wait forever: a lease inside another one in the same thread reserves only what the
  outer one does not cover (the concept-shorts deadlock), memory the leasing process itself uses does not
  count against it, a wait with no other lease held and no change for `stall_s` (120 s) raises
  `CapabilityError` naming the holders (and the outer lease of a nested one), and long waits are logged
  and listed in `status().waiting` ([0004](design/changes/0004-lease-that-can-never-be-granted.md),
  [0014](design/changes/0014-nested-leases-in-one-process.md)).
- Expressive speech: `synthesize(..., emotion=, intensity=)` with a shared emotion vocabulary
  (`tts.emotions`), the `chatterbox` model (Chatterbox, MIT; extra `expressive`) that applies them, and
  `capabilities.expressive` (Kokoro accepts and ignores them); text is chunked by paragraph so a
  paragraph is synthesized whole; the speech span records emotion, intensity, `expressive` and paragraph
  count; `FakeSpeech(expressive=True)` ([0011](design/changes/0011-expressive-speech.md)).
- Speech sessions: `with tts.session():` loads the model once for many `synthesize` calls, holds the
  GPU lease for the block and frees the model at the end; spans record `session` and `loaded`;
  `FakeSpeech.loads` / `frees` ([0011](design/changes/0011-expressive-speech.md),
  [0009](design/changes/0009-speech-sessions-and-timings.md) option 1).
- `mk.speech(model_id)` for local text to speech: `synthesize(text, voice=, speed=, out=)` returns a
  `SpeechResult`; long text is chunked by sentence; `kokoro-82m` (extra `speech`), a
  `hone.models.speech` span, a GPU lease per call, `hone_models.testing.FakeSpeech`
  ([0006](design/changes/0006-speech.md)).
- Model registry: TOML merge (packaged, user, project, explicit), ad-hoc `provider:model` ids,
  capability-based selection (`require=`, `prefer=`), `hone-models models list|show|check` CLI.
- Span sinks (SQLite, JSONL, memory, null), content-capture switch, secret stripping.
- `mk.text(...)` for Ollama and OpenAI-compatible servers: capability checks before calling, context
  budget (`ContextOverflow`), thinking-model handling, structured output pipeline (constrain, parse,
  validate, retry, repair), retries with backoff recorded as span events.
- `mk.Prompt` / `mk.Section`: prompts from named, versioned sections with character spans in records.
- `mk.decision(...)`: emulated on LLMs (one schema-constrained call; yes/no calibrated from token logprobs
  when available) and native for Jev (unverified API mapper); `mk.YesNo`, `mk.Choice`, `mk.ScoreQ`.
- `mk.embedder(...)` for Ollama and OpenAI-compatible servers (L2-normalized).
- `mk.gpu`: cross-process GPU leases through a flock-guarded JSON ledger with NVML / `nvidia-smi` memory
  accounting and unloading of idle Ollama models; `NullGpuLease`, `FileLockGpuLease`.
- `mk.session("ollama")` (use or start the server, stop only what was started) and `mk.unload(model_id)`.
- `mk.replay.Replayer`: replay recorded chat calls with section, model, params or message overrides;
  also spans from other recorders (JSON-string attributes, sections without `role`) (0002).
- CLI: `hone-models calls list|show|stats`.
- LiteLLM provider (extra `litellm`) with capability hints from LiteLLM's model map.
- Entry points `hone.text_clients`, `hone.decision_clients`, `hone.embedders`, `hone.gpu_leases` and
  `hone.replayers` (0002).
- Spans copy `hone.lens.finding_id` from the trace context, like the other correlation keys (0002).
- `hone_models.testing.FakeOllama`: a local fake Ollama server for tests and examples, scriptable with
  `queue(path, *answers)`, with `start()` / `stop()`, Ollama's OpenAI-compatible `/v1` endpoints and
  request `headers` in `server.requests` (0003).
- `examples/`: 18 runnable, offline examples, one per public concept, each with a What / How / Why
  docstring, indexed in `examples/README.md` and executed by the test suite (0003).
- Docs (`docs/`) and a real-model test suite (`tests/gpu/`, run through `scripts/gpu-lock.sh`).

### Fixed (before release)
- `hone.models.cost_usd` is recorded when a reply omits a token count whose price is 0 (Jev reports
  input tokens only).
- `hone-models calls list` never shortens span ids in narrow terminals (they are what `calls show` takes).
