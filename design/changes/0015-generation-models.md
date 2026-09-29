# 0015: Generation models: images, music, video and transcription

## Status

`proposed`

## Context

The owner decided: **"We should be able to call ALL of our models through hone-models."** Today
hone-models covers chat and vision LLMs (Ollama, OpenAI-compatible servers, LiteLLM), decision questions,
embeddings and local speech (Kokoro, Chatterbox: [0006](0006-speech.md), [0011](0011-expressive-speech.md)).
It cannot generate images, music or video, and it cannot transcribe.

The first OneShotStudio (a demo app built on the honeworks packages) filled that gap with its own code: a
`Comfy` class that started ComfyUI through comfy-cli (`comfy launch --background`, `comfy stop`), loaded
UI templates and converted them to the API format, edited workflow nodes by class name, uploaded files by
copying them into ComfyUI's `input/` folder, ran jobs through `comfy run --wait`, and called `POST /free`
between renders; plus its own openai-whisper transcriber for lyric timing, and hand-placed `release()`
calls so ComfyUI and the torch scorers did not meet on the GPU. None of those calls was recorded, none
took a GPU lease, and none went through the registry's capability checks. The rebuilt OneShotStudio left
this open as its decision D-004 ("image / music / video generation: app adapter or hone-models change?").
This record closes D-004 in favour of hone-models: every model call goes through hone-models, with the
same registry, capability checks, records, GPU leases and unloading as text and speech.

The models in use or planned on this machine (RTX 4060 Laptop, 8 GB; 24 CPU cores; 30 GB RAM), checked
on 2026-09-29:

| Kind | Local | Hosted |
|---|---|---|
| image (generate, edit with references) | ComfyUI 0.37: `z_image_turbo_bf16` (installed); FLUX.2-klein-4B and Qwen-Image-Edit-2511 GGUF Q4 + Multiple-Angles LoRA (planned, not downloaded yet; ComfyUI-GGUF node installed) | Any OpenAI-compatible API (OpenAI itself, `OPENAI_API_KEY`, or another gateway with the same endpoints): `POST /images/generations`, `POST /images/edits` (gpt-image-*, and where a gateway serves them gemini-*-image, flux.2-pro, flux.1-kontext-pro, qwen-image-*, seedream) |
| music / songs | ComfyUI: ACE-Step 1.5 turbo, XL turbo, XL sft; MiniMax-Music3; YuE2-3B; HeartMuLa (custom node `ComfyUI_FL-HeartMuLa`). Standalone projects: SongGeneration v2 (`~/ai-stack/levo2`, own Python 3.10 venv, `generate.sh ... --low_mem`); DiffRhythm2 (`~/ai-stack/diffrhythm2`, weights downloaded, no environment yet; its script needs `espeak-ng`) | none today |
| video | ComfyUI: Wan 2.2 I2V-A14B fp8 (high + low noise) with the lightx2v 4-step LoRAs, Wan 2.2 TI2V-5B, LTX-Video 2B 0.9.5 | OpenAI-compatible `POST /videos` (asynchronous: `GET /videos/{id}` until `completed`, then `GET /videos/{id}/content`; `DELETE` to cancel): sora-2, sora-2-pro on OpenAI; Veo 3.1 or Runway where a gateway serves them the same way; 4, 8 or 12 s; fixed sizes |
| transcription with word timestamps | faster-whisper large-v3-turbo (`deepdml/faster-whisper-large-v3-turbo-ct2` in the HF cache), used for lyric timing and intelligibility | not needed now |

ComfyUI is started on this machine with `~/ai-stack/start_comfyui.sh` (its own `.venv`, `python main.py
--reserve-vram 1.5`); comfy-cli is not installed outside the old app's environment. ComfyUI's HTTP API
has `POST /prompt` (queue a job, validation errors come back as `node_errors`), `GET /history/{id}`,
`GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`, `POST /interrupt`, `GET /view` (fetch an output),
`POST /upload/image` (store an input file), `POST /free` and `GET /system_stats`. It has no endpoint that
names its loaded models.

Two pieces of work build on this one: [0016](0016-machine-state.md) (machine state) names ComfyUI's models
by registry id once they are registry entries, and hone-select's experiments
([hone-select 0009](https://github.com/honeworks/hone-select/blob/main/design/changes/0009-experiments.md))
want generation subjects that work like its prompt subject (`client = "hone_models:text"`) and need to
know what each model can take (hone-select 0011).

## Problem

- There is no model kind for images, music, video or transcription, so each app wires ComfyUI, the image
  and video APIs, standalone projects and Whisper by hand: outside the registry, the GPU lease and the
  records.
- ComfyUI holds its models in another process. hone-models can only free it as a whole
  (`mk.gpu.comfyui_free`), cannot say which models it holds, and does not know when it is running.
- Generation is slow and asynchronous (a song or a video takes minutes; a hosted video is a job to poll)
  where every existing call is one blocking request with a 120-600 s timeout.
- Results are files, not text; the records have no place for inputs that are files (reference images,
  source audio) or outputs that are files.
- The inputs differ by model: a workflow node's name for "the prompt", lyrics formats, frame counts
  instead of seconds, fixed sizes and durations on hosted video.
- What a model can do beyond the common inputs is written nowhere a program can read: the trigger phrases
  of an add-on (the Multiple-Angles LoRA changes the camera angle only when the prompt holds one of its
  phrases), a model's prompt style (ACE-Step wants comma-separated tags), its lyric section format, the
  prompt recipes a hosted model is known for. The code that writes prompts, and the person or experiment
  comparing models, have to guess.

## Options

**API shape.**

1. **One `mk.generate(model_id, **inputs)`** for every kind. One name to learn, but the return value,
   the checks and the documentation depend on a kind the caller cannot see in the call, and
   `mk.generate("faster-whisper...")` for transcription reads wrong.
2. **One client per kind, like `mk.text` / `mk.speech` / `mk.embedder`**: `mk.image`, `mk.music`,
   `mk.video` and `mk.transcriber`. Each call site says what it produces; the kind is checked when the
   client is made (as `mk.speech` does). Image, music and video share one implementation and one result
   type, so this costs three thin factories, not three code paths.
3. **Typed methods per operation** (`image.edit(...)`, `music.cover(...)`, `video.from_image(...)`).
   Nicer to read, but the operations a model supports come from its workflow, not from a fixed list;
   every new workflow shape would need a new method.

**ComfyUI server.**

1. **comfy-cli** (`comfy launch --background`, `comfy stop`, `comfy run --wait`), as the old app did. A
   dependency with its own workspace config; `--background` detaches, so stopping means trusting its
   bookkeeping; `run --wait` hides the job id we need to cancel and to record.
2. **Plain HTTP plus an optional start command**: use a running server; when none answers and a start
   command is configured, start it in the foreground as a child process (like `mk.session("ollama")`
   starts `ollama serve`) and stop only what we started. Jobs go through `/prompt`, `/history`, `/view`
   directly.
3. **Embed ComfyUI in-process.** Its own torch, custom nodes and globals in our process: no.

**Standalone Python projects (SongGeneration, DiffRhythm2).**

1. **In-process import.** They pin old Python (3.10) and torch builds and are not packages; importing them
   would break the caller's environment.
2. **A subprocess in the project's own environment**, through a small adapter script that speaks one
   file protocol. Isolation for free, and the process exit frees the GPU for certain.
3. **Wrap them in ComfyUI custom nodes.** Maintaining nodes for other people's research code; no.

**Transcription.** In-process faster-whisper (CTranslate2, no torch, about 1.6 GB of VRAM in fp16), like
Kokoro; or a hosted Whisper through `/audio/transcriptions`. Local first; the hosted one fits the same
client later.

## Decision

Proposed: API option 2, ComfyUI option 2, projects option 2, in-process faster-whisper. The design reuses
what speech introduced (a kind, a provider table, a loaded engine, a lease per call or per session, a span
per call, a fake) and adds three ideas: **named inputs mapped onto a workflow**, **a guide per model**
in the registry that says what the model can take (§3a), and **the registry as the catalog** of every model
we use or may use, installed or not (§3b).

The principle (owner, 2026-09-29): everything about calling a model (its inputs and their formats, the
adapters that reach it, starting and stopping what serves it, its GPU memory) belongs in hone-models.
What to ask a model for (the story, the style, the camera angle, the words of a prompt) belongs to the app.
Where one input can be written the same way for several models, hone-models converts it; where it cannot,
the model's guide says how that model wants it.

### 1. Public API

```python
import hone_models as mk
from pathlib import Path

img = mk.image("z-image-turbo")                         # kind "image"; ConfigError for another kind
r = img.generate("a lighthouse at dusk, oil painting", size="1024x1024", seed=7, out="shots/01.png")
r.path, r.files, r.seed, r.error, r.span_id, r.cost_usd

edit = mk.image("qwen-image-edit-2511")                 # references: files, uploaded and recorded by hash
edit.generate("the same girl, seen from behind", references=[Path("sheet/front.png")], out="sheet/back.png")

song = mk.music("ace-step-1.5-xl-turbo")
r = song.generate("dark trap, 95 bpm, female vocals", lyrics=text, duration_s=150, seed=3, bpm=95,
                  out="takes/take_1.flac")

clip = mk.video("wan2.2-i2v-14b")
clip.generate("slow push-in, candle flicker", image=Path("shots/01.png"), duration_s=5, out="clips/01.mp4")

with song.session():                                    # lease, server and model held for the block
    for i, seed in enumerate(seeds):
        song.generate(style, lyrics=text, seed=seed, out=f"takes/take_{i}.flac")

stt = mk.transcriber("faster-whisper-large-v3-turbo")
t = stt.transcribe("takes/take_1.flac", language="en", prompt=text)   # prompt biases toward the lyrics
t.text, t.language, t.duration_s, t.segments[0].words[0]              # Word(text, start_s, end_s, probability)

with mk.session("comfyui"):                             # keep one ComfyUI server up across clients
    ...
```

- `mk.image(model_id)`, `mk.music(model_id)`, `mk.video(model_id)` return a `MediaClient` (one class; the
  factory checks the entry's `kind`). Signature, all keyword-only after the prompt:
  `generate(prompt, *, out, seed=None, timeout_s=None, trace=None, **inputs) -> MediaResult`.
- **Named inputs.** Besides `prompt` and `seed`, a small shared vocabulary: `negative`, `size`
  (`"WxH"`), `references` (list of image paths), `image` (a start frame), `source` and `strength`
  (audio or video to transform, 0-1), `lyrics`, `duration_s`, `n`, `steps`. A model may accept more
  (`bpm`, `key`, `time_signature`, `language`, `quality`, ...) in its own vocabulary; the entry lists
  what it accepts (§3). **An input the model does not accept raises `ConfigError` before any request**,
  naming the accepted ones. (Text calls ignore unknown params; for generation a silently ignored
  `duration_s` would waste minutes of GPU time.) A `pathlib.Path` value is a file input.
- `seed=None` takes the entry's `defaults.seed`, else a random seed; the seed used is always on the
  result and the span, so every output can be made again.
- `out` is the output file. If the model returns several files (`n > 1`, stems), they are
  `<stem>_1<suffix>`, `<stem>_2<suffix>`, ...; without a suffix, the provider's is used. Parent folders are
  created.
- `MediaResult`: `files: list[MediaFile]`, `path` (the first file, or `None`), `model`, `seed`,
  `error`, `error_kind`, `span_id`, `job_id` (the provider's), `cost_usd` (`None` when unknown),
  `cost_estimated`, `elapsed_s`, `license`, `commercial_use` (both from the registry, `None` when not
  declared).
- **A failed result says enough to decide what to do next** (owner): `error_kind` is one of
  `out_of_memory` (retry later, smaller, or after freeing memory), `refused` (moderation or policy: change
  the prompt), `invalid_input` (a value the model rejected while running: change the input), `no_output`
  (the job finished without files) or `failed` (anything else), with the provider's message in `error`
  and the job id kept, so a script can retry, fall back to another model or give up on its own terms.
  `MediaFile`: `path`, `sha256`, `bytes`, `mime`, `width`, `height`, `duration_s` (each `None` when
  unknown or not applicable, never 0). Images are measured with the header reader from 0013, WAV with
  `wave`, FLAC from its STREAMINFO block, anything else with `ffprobe` when it is on `PATH`.
- `mk.transcriber(model_id)` returns a `Transcriber`:
  `transcribe(audio, *, language=None, prompt=None, words=True, timeout_s=None, trace=None) ->
  Transcript` with `text`, `language`, `duration_s`, `segments: list[TranscriptSegment(text, start_s,
  end_s, words)]`, `words: list[Word(text, start_s, end_s, probability)]`, `model`, `span_id`. Aligning
  known lyrics to the heard words stays in the app: it is not a model call.
- Both clients have `session()` with the speech semantics ([0009](0009-speech-sessions-and-timings.md)):
  the GPU lease is taken once, the model stays loaded for the block and is freed once at the end (also on
  an exception); sessions of one client do not nest.
- `mk.session("comfyui")` joins `mk.session("ollama")`: use the running server, else start it, stop on exit
  only what it started. `mk.unload(model_id)` for a ComfyUI entry sends `POST /free` (ComfyUI can only
  free everything at once; documented).
- `mk.guide(model_id)` returns the model's `ModelGuide` (§3a); `mk.select(kind="image",
  require={"features": ["camera angle"]})` finds the models that declare a feature.

### 2. Providers

| Provider | Kinds | Talks to | Notes |
|---|---|---|---|
| `comfyui` | image, music, video | a ComfyUI server over HTTP (`base_url`, default `HONE_COMFYUI_URL`, else `http://127.0.0.1:8188`) | one API-format workflow per entry; no new dependency |
| `openai_compatible` | image, video (new); chat, embedding | `/images/generations`, `/images/edits`, `/videos` | the existing provider, extended; OpenAI and any gateway with these endpoints |
| `command` | music (any kind) | a subprocess in the project's own environment | SongGeneration now, DiffRhythm2 later |
| `faster_whisper` (extra `transcribe`) | transcription | faster-whisper in-process | like Kokoro: loaded per call or per session |

**`comfyui`.**

- *Workflow.* Each registry entry names an API-format workflow JSON (`workflow`, exported from ComfyUI
  with "Export (API)"; no UI-to-API conversion). A relative path is resolved against the registry file
  that declares it; packaged workflows live in `hone_models/data/workflows/`. The workflow is read once
  per client and its SHA-256 is recorded, so a changed workflow file is visible in the records.
- *Inputs.* The entry's `inputs` table maps each input name to one or more `"<node id>.<input>"` paths
  (node ids, not class names: unambiguous within one file). `size` fills the `width` and `height`
  mappings. A mapping may convert seconds to frames, `{ path = "55.length", per_second = 16, add = 1 }`,
  the one conversion two real models need (Wan at 16 fps, LTX at 25 fps). File inputs are uploaded with
  `POST /upload/image` (it stores any file type in `input/`; ComfyUI's `LoadAudio` reads from there too)
  into the subfolder `hone/` under their SHA-256 name, so a reference used for 40 shots is uploaded once,
  and this also works for a ComfyUI on another host. `references` maps to a list of slots (`LoadImage`
  nodes); a slot without a reference is removed from the copy of the workflow with the links to it, and
  if a required input is then missing ComfyUI's own validation says so. Inputs with a common format
  (`lyrics`, prompt inputs such as `camera_angle`) are converted first (§3a); the rest are passed as given,
  and the model's guide says what values it takes (ACE-Step's `keyscale`, for example).
- *Outputs.* `outputs` lists the node ids whose files are the result (default: every file in the job's
  history). Each file is fetched with `GET /view` and written to `out`; nothing is moved inside ComfyUI.
- *Job.* `POST /prompt` with a client id. A `400` with `node_errors` (missing node class, missing model
  file, a bad value) raises `ConfigError` naming the node and the input. Then `GET /history/{id}` is
  polled (1 s, then 2 s after the first minute; `/api/jobs/{id}` adds progress where the server has it)
  until the job is done. There is no websocket client, so no new dependency.
- *Server.* A call uses the server at `base_url` if `GET /system_stats` answers. **Only a session starts
  one** (owner): `client.session()` or `mk.session("comfyui")`, when no server answers and
  `HONE_COMFYUI_START` is set (a command line, e.g. `~/ai-stack/start_comfyui.sh --port 8188`), starts it
  in the foreground as a child in its own process group, with the clean environment of
  `mk.session("ollama")` (no `VIRTUAL_ENV`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `PYTHONHOME`), waits up to
  `HONE_COMFYUI_START_S` (default 300 s), and stops it at the end of the block. A plain call without a
  running server raises `ProviderError` saying to start ComfyUI or to wrap the calls in a session. A
  server that was already running is never stopped; a remote `base_url` is never started.
- *Memory.* Outside a session a call ends with `POST /free`, as a speech call frees its model; inside a
  session the model stays loaded and is freed once at the end. When a client is first used, it registers
  the release hook `on_short("comfyui:<url>", comfyui_free(url))` ([0005](0005-release-hooks-for-other-gpu-servers.md)),
  so a later lease for any model in this process can make ComfyUI let go.
- *Loaded-models file.* Because ComfyUI cannot name its loaded models, hone-models keeps
  `${HONE_HOME}/models/comfyui-loaded.json` (guarded by `flock`, like the lease ledger): per server URL,
  the registry ids run since the last `/free`, each with the job id, pid and time. A job adds its id; any
  `/free` hone-models sends (end of call, end of session, release hook, `mk.unload`, 0016's `prepare`)
  clears the server's list. 0016 reads it (§7).

**`openai_compatible` for images and video.**

- Images: without `references`, `POST {base_url}/images/generations` (JSON: `model`, `prompt`, `size`,
  `n`, `quality`, `response_format = "b64_json"`); with them, `POST /images/edits` as multipart (`image[]`,
  optional `mask` later). `b64_json` is decoded straight to the output file; a `url` answer is downloaded.
  Bytes never reach the records. `revised_prompt` is recorded when returned.
- Video: `POST /videos` (multipart when `image` is given, as `input_reference`), then `GET /videos/{id}`
  every 5 s (progress events on the span), then `GET /videos/{id}/content`. Status `failed` becomes
  `result.error` with the provider's message.
- Sizes and durations the entry declares (`capabilities.sizes`, `durations_s`, `max_duration_s`) are
  checked before the request (`CapabilityError`), so Sora's "4, 8 or 12 seconds" fails in the test with
  the fake instead of after a paid request.
- Submitting uses the existing retries (connection errors, 429, 5xx). **A submitted job is never submitted
  again** automatically: polling tolerates transient errors (up to 5 in a row, then `ProviderError` with
  the job id, so the caller can fetch it later), and a hosted job costs money.
- Cost: the registry `price` gains `per_image` and `per_output_second`; `cost_usd` = images × `per_image`
  or seconds × `per_output_second`; `None` without a price. This is a **naive estimate** (owner: enough
  for now): one flat price per entry, although hosted prices vary by size and quality. The result and the
  span carry `cost_estimated = True`, and the code and docs say the estimate is naive.

**`command`** (subprocess in the project's environment).

- The entry gives `command` (a list with the placeholders `{request}`, `{out_dir}`, `{adapter:<name>}`),
  optional `cwd` and `env`. hone-models writes `request.json` (the prompt, the inputs, the seed, `out_dir`,
  file inputs as absolute paths) into a fresh job folder next to `out`, runs the command in its own
  process group with the clean environment, and waits.
- The protocol is small: the program writes its files into `out_dir` and may write `result.json` there
  with `{"files": [...], "error": null, "meta": {...}}`. Exit code 0 with files is success; an `error` in
  `result.json`, or exit 0 without files, is `result.error`; any other exit code is `ProviderError` with
  the last 40 lines of stderr.
- hone-models ships one adapter script per project in `hone_models/data/adapters/` (`levo2.py` now,
  `diffrhythm2.py` later), each run with the project's own Python and importing only the standard library
  before handing over to the project: it converts `request.json` to the project's input (LeVo's JSONL
  with `idx`, `gt_lyric`, `descriptions`, the lyrics already converted to LeVo's section format by §3a;
  `--low_mem --not_use_flash_attn` from the entry's defaults), sets
  the seeds the project does not take as options (Python, NumPy, torch), and writes `result.json`.
- On timeout or cancellation the process group gets `SIGTERM`, then `SIGKILL` after 10 s. The process's
  exit frees its GPU memory; no unload step is needed.

**`faster_whisper`** (extra `transcribe` = `faster-whisper`; no torch). The model comes from the HF cache
or Hugging Face (`model` is the repo id or a local path), `defaults` give `device`, `compute_type`
(`float16` on CUDA), `beam_size`, `vad_filter`. Loaded per call inside the lease and freed afterwards, or
once per session. CTranslate2 needs the CUDA 12 cuBLAS and cuDNN 9 libraries; when they are missing,
`transcribe` raises `ConfigError` before the lease, naming the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`
wheels (how they are found is an implementation note for decisions.md, as for Kokoro's spaCy model in
[0008](0008-speech-extra-and-the-spacy-model.md)).

### 3. Registry entries

`kind` gains `image`, `music`, `video` and `transcription`; `provider` gains `comfyui`, `command` and
`faster_whisper`. New entry keys: `workflow`, `inputs` (for `comfyui` a table of node paths; for the others
a list of extra input names passed through as request fields), `outputs` (`comfyui`), `command`, `cwd`,
`env` (`command`). New capabilities, all `None` when not declared (D-002): `max_references`, `sizes`,
`max_duration_s`, `durations_s`, `word_timestamps`, `commercial_use`, and `features` (the names of the
guide's features, §3a, filled from the guide so selection can match them). `commercial_use` never blocks a
call (owner): it is information, copied onto every result and span, and `require={"commercial_use": True}`
keeps non-commercial models such as YuE2 out when a caller asks for that; `license` stays the text.
A `guide` table and `lyrics_format` / `prompt_inputs` keys are new too (§3a). `vram_gb` is the
peak measured on the 8 GB card with ComfyUI's offloading, not the file size. `max_timeout_s` defaults by
kind: image 600 s, music 1800 s, video 3600 s, transcription 600 s. `local` is true for `comfyui` and
`command` on a local host and for `faster_whisper`.

```toml
[models."z-image-turbo"]
provider = "comfyui"
kind = "image"
workflow = "workflows/z-image-turbo.json"          # API format; relative to this file
outputs = ["9"]
defaults = { size = "1024x1024", steps = 8 }
[models."z-image-turbo".inputs]
prompt = "6.text"
seed = "3.seed"
steps = "3.steps"
width = "13.width"
height = "13.height"
[models."z-image-turbo".capabilities]
vram_gb = 7.0                                      # to be measured by the gpu test
max_references = 0
license = "Apache-2.0"
commercial_use = true

[models."qwen-image-edit-2511"]
provider = "comfyui"
kind = "image"
workflow = "workflows/qwen-image-edit-2511-gguf.json"   # UnetLoaderGGUF + Multiple-Angles LoRA
[models."qwen-image-edit-2511".inputs]
prompt = "111.prompt"
seed = "3.seed"
references = ["78.image", "106.image", "108.image"]    # up to three; unused slots are removed
[models."qwen-image-edit-2511".capabilities]
max_references = 3
vram_gb = 7.5
license = "Apache-2.0"
commercial_use = true
[models."qwen-image-edit-2511".prompt_inputs.camera_angle]   # a named input written into the prompt
place = "append"
choices = { left_45 = "<LoRA phrase>", right_45 = "<LoRA phrase>", top_down = "<LoRA phrase>", close_up = "<LoRA phrase>" }
[models."qwen-image-edit-2511".guide]
summary = "Edits or re-renders up to three reference images; keeps a character's identity."
prompt = "Plain sentences saying what to change; call the references 'image 1', 'image 2', 'image 3'."
source = "https://huggingface.co/Qwen/Qwen-Image-Edit-2511"
checked = 2026-09-29
[[models."qwen-image-edit-2511".guide.features]]
name = "camera angle"
how = "The Multiple-Angles LoRA moves the camera when the prompt holds one of its phrases; the camera_angle input writes it."
input = "camera_angle"
examples = ["<LoRA phrase for 45 degrees left>", "<LoRA phrase for a top-down view>"]
source = "<the LoRA's model card>"                   # the phrases are copied from here when the entry is written

[models."ace-step-1.5-xl-turbo"]
provider = "comfyui"
kind = "music"
workflow = "workflows/ace-step-1.5-xl-turbo.json"
outputs = ["111"]
lyrics_format = "sections"                         # the common format as is (§3a)
[models."ace-step-1.5-xl-turbo".inputs]
prompt = "94.tags"
lyrics = "94.lyrics"
seed = "109.value"
duration_s = ["94.duration", "98.seconds"]
bpm = "94.bpm"
key = "94.keyscale"
time_signature = "94.timesignature"
language = "94.language"
[models."ace-step-1.5-xl-turbo".capabilities]
max_duration_s = 600
vram_gb = 7.0
[models."ace-step-1.5-xl-turbo".guide]
summary = "Full songs with vocals up to 10 minutes; fast; follows genre and instrument tags well."
prompt = "Comma-separated tags: genre, mood, instruments, vocal type, tempo (e.g. 'dark trap, 95 bpm, female vocals, 808')."
inputs = { key = "e.g. 'C minor', 'F# major'", time_signature = "'4' or '3'", language = "ISO code of the sung language" }
source = "https://github.com/ace-step/ACE-Step-1.5"
checked = 2026-09-29

[models."wan2.2-i2v-14b"]
provider = "comfyui"
kind = "video"
workflow = "workflows/wan2.2-i2v-14b-lightx2v.json"
[models."wan2.2-i2v-14b".inputs]
prompt = "6.text"
negative = "7.text"
seed = "57.noise_seed"
image = "52.image"
duration_s = { path = "50.length", per_second = 16, add = 1 }
width = "50.width"
height = "50.height"
[models."wan2.2-i2v-14b".capabilities]
max_duration_s = 5
vram_gb = 7.5
license = "Apache-2.0"
commercial_use = true

[models."songgeneration-v2-medium"]
provider = "command"
kind = "music"
command = ["~/ai-stack/levo2/.venv/bin/python", "{adapter:levo2}", "{request}"]
cwd = "~/ai-stack/levo2"
inputs = ["generate_type"]                         # mixed | vocal | bgm | separate
defaults = { generate_type = "mixed", low_mem = true }
lyrics_format = "levo"                             # converted from the common format (§3a)
[models."songgeneration-v2-medium".capabilities]
vram_gb = 7.5

[models."sora-2"]
provider = "openai_compatible"
kind = "video"
model = "sora-2"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
[models."sora-2".capabilities]
sizes = ["720x1280", "1280x720"]
durations_s = [4, 8, 12]
max_references = 1
price = { per_output_second = 0.10 }

[models."gpt-image-1.5"]
provider = "openai_compatible"
kind = "image"
model = "gpt-image-1.5"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
inputs = ["quality", "background"]
[models."gpt-image-1.5".capabilities]
sizes = ["1024x1024", "1536x1024", "1024x1536"]
[models."gpt-image-1.5".guide]
summary = "Follows long, precise instructions; renders text in images; edits with reference images."
prompt = "Full sentences; say what to keep from each reference."
source = "https://platform.openai.com/docs/guides/image-generation"
checked = 2026-09-29
[[models."gpt-image-1.5".guide.features]]
name = "restyle a reference"
how = "Give the picture as a reference and name the target form in the prompt; there are no commands in the API, only instructions."
examples = ["Turn image 1 into a detailed technical blueprint: white lines on blue, labelled dimensions."]
source = "https://platform.openai.com/docs/guides/image-generation"

[models."faster-whisper-large-v3-turbo"]
provider = "faster_whisper"
kind = "transcription"
model = "deepdml/faster-whisper-large-v3-turbo-ct2"
defaults = { compute_type = "float16", beam_size = 5 }
[models."faster-whisper-large-v3-turbo".capabilities]
word_timestamps = true
vram_gb = 2.0
license = "MIT"
commercial_use = true
```

(Node ids above are illustrative except ACE-Step's, which are those of the workflow in use today. The
`<LoRA phrase>` values are placeholders: the real phrases are copied from the LoRA's model card when the
entry is written.)

### 3b. The catalog: every model we use or may use, installed or not

Owner: "We should have all the possible models in hone-models, even the ones that are promising but are
not installed yet." So the packaged registry becomes the **catalog**: one entry for every model we
use, and for every model worth trying, whether or not this machine has it. Being installed is a fact
hone-models checks on the machine, not something the entry claims.

**What an entry adds for this:**

```toml
[models."flux.2-klein-4b"]
provider = "comfyui"
kind = "image"
workflow = "workflows/flux.2-klein-4b.json"
# ... inputs, capabilities, guide as in §3 and §3a ...
[models."flux.2-klein-4b".install]
source = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
files = [{ repo = "Comfy-Org/flux2-klein", file = "flux-2-klein-4b.safetensors", to = "diffusion_models" }]
size_gb = 7.8
tier = 1                                   # 1: test first; 2: worth a try; 3: to learn the ceiling
note = "generation and reference editing in one small model; fits the GPU"
```

- `install` says where the model comes from and what to fetch: an Ollama name (`ollama = "hf.co/..."`),
  Hugging Face files with their ComfyUI folder, or a project to clone and set up (`command` entries), plus
  the size, a tier and a one-line note on why it is in the catalog. Every entry has one, installed or not.
- `hone-models models list` gains an `installed` column (`yes`, `no`, `unknown`) and the filters
  `--installed`, `--missing`, `--tier N` (with the existing `--kind`, `--feature`). What "installed" means
  per provider: Ollama `/api/tags` names the model; the ComfyUI files exist under `HONE_COMFYUI_DIR` (or
  the server's `/object_info` lists them); the Hugging Face files are in the cache; a `command` project's
  folder variable points at a folder whose check command succeeds. Unknown when the server does not
  answer or the variable is unset, never "no" by guess.
- `hone-models models install <id>` **prints** the commands that would fetch the model (an `ollama pull`,
  `hf download ... --local-dir ~/ComfyUI/models/<folder>`, the clone and setup steps) with the size and
  the free disk space; it downloads nothing (owner: downloads are started by the owner). `--run` exists
  for the owner to run them from the tool, never used by code or agents without the owner asking.
- A call to a model that is not installed fails before any job, with `ConfigError` naming the entry and
  saying `hone-models models install <id>`.
- Kinds hone-models cannot call yet (the scoring models, until the change that adds them) are still
  catalog entries: they have `install`, a guide and a license, and calling them raises `ConfigError`
  "no client for kind '<kind>' yet". Knowing a model and calling it are separate steps.

**Contents.** The packaged catalog holds every model in the owner's research catalog of 2026-09-29 and
every model on this machine, with its license and `commercial_use` flag. No machine paths: folders come
from variables (`HONE_COMFYUI_DIR`, `HONE_LEVO2_DIR`, ...). The workspace's `hone-models.toml` only
overrides (prices, another gateway, measured `vram_gb`, project-local models).

| Kind | Installed here (2026-09-29) | In the catalog, not installed |
|---|---|---|
| chat: writers | hemmingway-1 (non-commercial; owner: a real option), muse-glimmer-30b, styletune-12b / 26b-a4b / 31b, equinox-31b, hearthfire-24b, meromero-26b-a4b, meromero-v2-31b, cydonia-24b | orion-26b-a4b, artemis-31b, qwen3.6-35b-a3b-styletune, pantheon-reasoning-26b-a4b, gemma4-writer-31b-d, nemotron-3.5-30b-a3b-antislop, gemma-3-27b-antislop |
| chat: general, reasoning, vision | qwen3.8-27b, qwen3.6-27b, qwen3.6-35b-a3b, gemma4-12b / 26b-a4b / 31b, nemotron-3.5-lightning, gpt-oss-20b, deepseek-r1-8b / 14b / 32b, ornith-1.5-9b / 35b, qwen2.5vl-7b | |
| embedding | nomic-embed-text | |
| image | z-image-turbo | flux.2-klein-4b, qwen-image-edit-2511 (+ multiple-angles LoRA), krea-2-turbo, ming-image-0.1-design, character-sheet LoRA; ceiling only (non-commercial): flux.2-klein-9b, flux.2-dev, qwen-image-2.1 |
| music | ace-step-1.5-turbo / xl-turbo / xl-sft, minimax-music3, yue2-3b (non-commercial), heartmula-3b, heartmula-rl-3b, stable-audio-open-1.0, songgeneration-v2-medium | diffrhythm2 (weights here, no environment yet), stable-audio-3-small-music / medium, mulacover (non-commercial) |
| video | wan2.2-i2v-14b (+ lightx2v), wan2.2-ti2v-5b, ltx-video-2b-0.9.5 | minimax-h3, ltx-2.5, scail-2 |
| speech | kokoro-82m, chatterbox | |
| transcription | faster-whisper-large-v3-turbo, large-v3, medium | |
| hosted (`openai_compatible`) | gpt-image-1.5, sora-2, sora-2-pro, gpt-4.1-mini, jev (existing) | other gateways' models by a user or project entry |
| scoring (callable later) | skywork-reward-v2-0.6b, litbench-rm-3b, dinov2-small, audiobox-aesthetics, muq-mulan-large, muq-large-msd, htdemucs, mel-band-roformer, depth-anything-v2-small | story-reward-8b, beat-this, songeval, florence-2, paddleocr-vl-1.6, hpsv3, pickscore, aesthetic-v2.5, seedvr2-3b / 7b, rife |

The packaged registry is split by kind (`hone_models/data/models/<kind>.toml`), one file per kind, so
the catalog stays readable at this size. Adding a promising model is one entry with `install`, a guide
and, for ComfyUI, a workflow; it needs no code. A generation entry's workflow is written and proven with
`hone-models models check <id>` (a tiny job) when the model is first installed; until then the entry
carries `workflow = None`, and calling it says the workflow is missing.

Machine paths are not in the package: a `command` entry reads its project folder from a variable
(`HONE_LEVO2_DIR` for SongGeneration), and a missing folder or variable is a `ConfigError` naming it. An
entry whose ComfyUI model file or custom node is missing fails at `/prompt` validation with a message naming
the file, like any other `node_errors`.

### 3a. Model guides: what each model can take

Two things, both in the registry, so a program, a person and an experiment read the same facts:

**Common formats, converted by hone-models where one format fits many models.**

- `lyrics` has one common format: section tags on their own line (`[intro]`, `[verse]`, `[pre-chorus]`,
  `[chorus]`, `[bridge]`, `[inst]`, `[outro]`), then one sung line per line. An entry's `lyrics_format`
  names the converter: `sections` (passed as is; ACE-Step, HeartMuLa, YuE2 read these tags), `levo`
  (SongGeneration's own form: its tag names, lines joined with `.`, sections with ` ; `), `plain` (tags
  removed). A new format is a small function in `hone_models/formats.py`, tested with the entry.
- `duration_s` and `size` are converted already (frames per second, width and height).
- `prompt_inputs` are named inputs that end up as words in the prompt, for models that are steered by
  fixed phrases: `camera_angle = "left_45"` appends the phrase the entry maps `left_45` to. The choices
  are the entry's; an unknown choice raises `ConfigError` listing them. The call site says what it wants;
  the phrase is the model's business.

**A guide where no common format fits.** An entry's `guide` table is documentation for whoever writes the
prompt (a person, the app's prompt-writing code, an LLM that is given the guide), never used by the call
itself:

| Key | Content |
|---|---|
| `summary` | one line: what the model is good at |
| `prompt` | how to write its prompt (tags, sentences, what to avoid) |
| `inputs` | a short note per model-specific input (`key = "e.g. 'C minor'"`) |
| `features` | special abilities, each with `name`, `how`, `input` (the named input that does it, if any), a few `examples`, and a `source` |
| `source`, `checked` | where current tips and samples live (the model card, the provider's guide), and the date the guide was last compared with it |

Features are samples, not catalogues (owner): the guide says the model *can* do something, shows a few
examples, and points to the source for the full, current list. `hone-models models guide --stale 90`
lists the guides not checked for 90 days.

Reading a guide:

```python
g = mk.guide("qwen-image-edit-2511")
g.summary, g.prompt, g.features[0].name, g.features[0].examples, g.source
g.inputs            # every input the model takes: common ones, its own, prompt inputs with their choices
g.as_text()         # one block of plain text, for a person or for an LLM prompt
```

```bash
hone-models models guide qwen-image-edit-2511          # the same, printed
hone-models models guide qwen-image-edit-2511 --json
hone-models models list --kind music --feature "lyrics"   # which models can do it
```

`ModelGuide` also carries the entry's `kind`, `license`, `commercial_use`, `sizes`, `durations_s`,
`max_duration_s` and `max_references`, so one object answers "what can I ask this model for". An entry
without a `guide` still has one, built from its inputs and capabilities, with `summary = None`.

### 4. GPU

- **Local generation and transcription calls lease by themselves**, as speech does: `mk.gpu.lease(model_id,
  capabilities.vram_gb)` around the job (a session holds it for the block). Hosted calls take no lease.
  D-009 (whether *text* calls should lease automatically) is not changed by this.
- The lease is taken in the calling process while the memory is used by another process (ComfyUI, the
  `command` child). That is what the ledger is for: the reservation counts before the other process
  loads, and `free = total - max(used, reserved)` keeps counting it while it runs.
- Freeing: `comfyui` sends `POST /free` at the end of a plain call or a session; `command` frees by
  exiting; `faster_whisper` frees like Kokoro (`close()`, CUDA cache emptied).
- Nested leases ([0014](0014-nested-leases-in-one-process.md)) cover a hone-flow GPU step whose own lease
  wraps these calls.
- The machine-wide lock (`scripts/gpu-lock.sh`) stays a tool for test runs and scripts; library calls
  coordinate through the ledger, as today.
- ComfyUI's offloading moves weights into system RAM (Wan 14B's two 14 GB experts, 30 GB of RAM). The lease
  does not schedule RAM; the docs say not to run two such jobs at once.

### 5. Records

One span per call, kind `client`: `hone.models.image`, `hone.models.music`, `hone.models.video`,
`hone.models.transcribe`; `gen_ai.operation.name` is `image`, `music`, `video` or `transcription`;
`gen_ai.provider.name` is `comfyui`, `openai`, `command` or `faster_whisper`; `gen_ai.request.seed` the
seed used.

| Attribute | Content |
|---|---|
| `hone.models.media.prompt` | the prompt; content |
| `hone.models.media.inputs` | every other input; file inputs as `{"path", "sha256", "bytes"}`; lyrics and texts are content |
| `hone.models.media.outputs` | `[{"path", "sha256", "bytes", "mime", "width", "height", "duration_s"}]` |
| `hone.models.media.{job_id,workflow_sha256,revised_prompt}` | the provider's job id; the ComfyUI workflow's hash; the hosted model's rewrite (content) |
| `hone.models.media.{queue_wait_ms,session,loaded,freed,server_started}` | time queued before running; in a session; this call loaded the model / freed it / started the server |
| `hone.models.media.log_tail` | `command`: the last 40 lines of stderr, capped at 4 KiB; content |
| `hone.models.media.{error,error_kind}` | the error and its kind, as on the result |
| `hone.models.media.{license,commercial_use,cost_estimated}` | from the registry; `cost_estimated` true when the cost is the naive flat-price estimate |
| `hone.models.transcribe.{audio,language,duration_s,words_count}` | the audio as `{"path", "sha256", "bytes"}` |
| `hone.models.transcribe.{text,words}` | content; long word lists go to the blob table as today |
| `hone.models.cost_usd`, `hone.models.gpu.*`, the trace context | as today |

Events: `progress` (at most one per 10 %), `retry` as today, `cancelled`. Content capture off hashes every
"content" attribute as today; API keys are scrubbed as today (`OPENAI_API_KEY` is an `api_key_env`).
**No bytes are recorded**: files are on disk and in the span by path and hash. Replay (§8.6) stays
chat-only; the recorded inputs, seed and workflow hash are what a re-run needs.

### 6. Errors, timeouts and cancellation

The existing rule, applied to jobs: **a job the provider accepted and ran without usable output is a
result** (`result.error`, `files` empty, span status `error`); **a job that could not be submitted, run
or fetched is an exception**.

| Case | Outcome |
|---|---|
| unknown input, file input missing, size or duration not declared by the model | `ConfigError` / `CapabilityError` before any request or lease |
| ComfyUI `node_errors` (missing node, model file, bad value) | `ConfigError` naming node and input |
| no server, transport or HTTP failure, `command` non-zero exit | `ProviderError` |
| ComfyUI `execution_error` (including CUDA out of memory), hosted status `failed`, a moderation refusal, a finished job without files | `result.error` with the provider's message and `result.error_kind` (`out_of_memory`, `refused`, `invalid_input`, `no_output`, `failed`) |
| no result within `timeout_s` (default: the entry's `max_timeout_s`) | the job is cancelled at the provider, then `ModelTimeout` |

Cancelling a job: ComfyUI `POST /api/jobs/{id}/cancel` (older servers: remove it from `/queue`, or
`/interrupt` if it is the one running); hosted video `DELETE /videos/{id}`; `command` process group
terminated. The same cleanup runs when the wait ends with any exception, so `Ctrl-C` or a hone-flow
cancellation that raises in the calling thread does not leave a job burning GPU time. A timeout covers
queueing and running; a job queued behind someone else's in ComfyUI counts.

### 7. Machine state ([0016](0016-machine-state.md)) with ComfyUI models in the registry

With this change ComfyUI's models are registry entries, so 0016 should adapt (this answers its open
question 2 and most of 3):

- **`snapshot()`** lists ComfyUI's models by id: one `loaded_models` entry per registry id in
  `comfyui-loaded.json` for that server, `{"server": "comfyui", "name": <workflow file>, "model_id": id,
  "size_gb": None, "vram_gb": None}`, and the server-level `torch_vram_total` from `/system_stats` on the
  server entry. The list is dropped when `/system_stats` shows no torch memory (ComfyUI restarted or freed
  by someone else). When the newest job in `GET /history` is not one hone-models recorded (someone used the
  ComfyUI UI), an entry with `model_id: None` is added, so "unknown" stays explicit.
- **`prepare(needed)`** takes registry ids only; the pseudo-name `"comfyui"` goes away. ComfyUI can only
  free everything, so `prepare` sends `/free` when it holds any model not in `needed` or any unknown
  memory, and keeps it when every model it holds is needed. It then clears the loaded-models file.
  `need_gb` includes the ComfyUI entries' `vram_gb`.
- **Which ComfyUI servers to check:** the distinct `base_url`s of the registry's `comfyui` entries (plus
  `HONE_COMFYUI_URL`); a registry without `comfyui` entries checks none, so other machines do not report a
  refused connection.
- `command` jobs show up as GPU processes (`gpus[].processes`) and hold a lease while they run, so
  `prepare` is already blocked by them; `faster_whisper` in a session is this process's own lease.
- 0016's warm-up `mk.machine.load(id)` (accepted by the owner) loads Ollama models; for `comfyui` and
  `command` entries loading means running a job, so `load` reports "not supported" there, and a caller
  that wants a warm model runs one small job in a session.

### 8. Shapes for other packages

design/current.md §10 gains rows for `mk.image(...)` / `mk.music(...)` / `mk.video(...)`:
`kind`, `model_id`, `generate(prompt, *, out, seed=None, timeout_s=None, trace=None, **inputs) ->
MediaResult`; and `mk.transcriber(...)`: `transcribe(audio, *, language=None, prompt=None, words=True,
timeout_s=None, trace=None) -> Transcript`. Entry points: `hone.image_clients`, `hone.music_clients`,
`hone.video_clients`, `hone.transcribers`, each `hone_models = "hone_models:<factory>"`, as for text.
`mk.guide(model_id) -> ModelGuide` (§3a) is a shape too: `id`, `kind`, `summary`, `prompt`, `inputs`
(name → note, choices for prompt inputs), `features` (name, how, input, examples, source), `source`,
`checked`, `license`, `commercial_use`, and the size and duration limits; `as_text()` and a JSON form.
Entry point `hone.model_guides` / `hone_models = "hone_models:guide"` exposes `mk.guide` the same way.
`mk.PORTS_VERSION` stays `"1"` (additive). hone-select uses these in its change record 0011: a generation
subject (`[generate] client = "hone_models:music"`, inputs filled from the case and the setup, files
written into the sample's `workdir`, `result.error` and `error_kind` as a failed sample), and the guides
to plan per-model prompts and to mark cases a model cannot do as "not applicable" instead of failed.

### 9. Testing without a GPU

- `hone_models.testing.FakeComfyUI`: a local HTTP server like `FakeOllama` with `/system_stats`,
  `/prompt` (validates that every mapped node and input exists, else `node_errors`), `/history/{id}`,
  `/api/jobs/{id}` and `/cancel`, `/interrupt`, `/queue`, `/upload/image`, `/view`, `/free`. A job
  "runs" for a scripted time and writes a small valid PNG, a WAV of `duration_s` of silence or a tiny
  packaged MP4; `queue(...)` scripts failures (`node_errors`, `execution_error`, a job that never ends);
  `requests` records every request and `submitted` every filled workflow, so tests assert the mapping.
- `hone_models.testing.FakeMedia.like(model_id)` and `FakeTranscriber` stand in for the clients in apps'
  tests, like `FakeSpeech` (same API, spans, validation; no server, no lease; `calls` kept).
- Hosted images and video: `respx` fixtures in `tests/fixtures/http/` (b64 and url answers, a
  `queued -> processing -> completed` video, `failed`, 429, a moderation refusal).
- `command`: a fake project script in `tests/fixtures/` that honours the protocol, fails, or hangs.
- `faster_whisper`: a fake `faster_whisper` module, as the speech tests fake `kokoro`.
- Server start and stop: a fake start executable, as for Ollama.

Acceptance cases (0016 takes AC-32 and AC-33):

| AC | Scenario | Expected |
|---|---|---|
| AC-24 | An image through `FakeComfyUI` with a reference used twice; a song with `duration_s` mapped to two nodes; a video with seconds converted to frames | inputs land on the mapped nodes; the reference is uploaded once by hash; outputs written to `out` with hash, size and dimensions or duration; one span each with inputs, outputs and the workflow hash; a lease with the registry's `vram_gb`; `/free` after a plain call, once after a session |
| AC-25 | Failures: unknown input; `node_errors`; `execution_error` (out of memory); a job that never ends; `KeyboardInterrupt` while waiting | `ConfigError` before any request; `ConfigError` naming the node; `result.error` with `error_kind = "out_of_memory"` and status `error`; the job cancelled and `ModelTimeout`; the job cancelled and the interrupt re-raised |
| AC-26 | Hosted images (generation, edit with references, b64 and url) and video (polled, then downloaded; `failed`; timeout) through respx | files written, no bytes in the SQLite file, cost from `per_image` / `per_output_second`, `failed` as `result.error`, `DELETE` sent on timeout, the planted `OPENAI_API_KEY` never stored; a size or duration the model does not declare raises before any request |
| AC-27 | `command` provider with the fake project: success, `result.json` error, non-zero exit, hang | files collected; `result.error`; `ProviderError` with the stderr tail; the process group killed on timeout |
| AC-28 | `mk.session("comfyui")` with no server and a fake start command; again with a running server; a plain call with no server | started once, reused by two clients, stopped at the end; a running server is never stopped; the plain call raises `ProviderError` saying to start ComfyUI or use a session, and starts nothing |
| AC-29 | Transcription with the fake module, capture on and off | words with times on the result; one span; text and words hashed with capture off |
| AC-30 **[real]** | Under `scripts/gpu-lock.sh`: a 512×512 `z-image-turbo` image, 10 s of `ace-step-1.5-turbo`, a Kokoro sentence transcribed by `faster-whisper-large-v3-turbo` | non-empty outputs; the transcript contains the sentence's words in order; GPU memory back to where it started; ComfyUI started only if the test started it, and stopped |
| AC-31 | Guides, formats and the catalog: `models list --installed` / `--missing` with FakeOllama tags, a fake ComfyUI folder and a fake HF cache; `models install <id>` for an Ollama, a ComfyUI and a `command` entry; a call to a model that is not installed; a catalog entry of a kind with no client; lyrics in the common format sent to a `sections` and a `levo` entry; `camera_angle` with a known and an unknown choice; `mk.guide`, `as_text()`, `models guide --json`, `models list --feature`; `require={"features": [...]}`; an entry without a guide | the two entries receive their own lyric forms; the phrase appended to the prompt, the unknown choice a `ConfigError` listing the choices; the guide lists every accepted input with its note, features with examples and source; selection returns only the models declaring the feature; the bare entry's guide is built from its inputs; installed is `yes` / `no` / `unknown` as the fakes say (a server that does not answer is `unknown`); `install` prints the pull, download or setup commands with the size and downloads nothing; the uninstalled call and the client-less kind raise `ConfigError` before any job, naming what to do |

Video and the standalone projects get `gpu` tests marked `slow`, run by hand.

### 10. Scope: now and later

| Now (this change) | Later, same pattern, no API change |
|---|---|
| kinds image, music, video, transcription; `MediaClient`, `Transcriber`, sessions | image masks for edits; video remix (`/videos/{id}/remix`) as named inputs |
| the catalog (§3b): every model in the research catalog and on this machine, installed or not, with `install`, license and guide; workflows for the installed ComfyUI models | a workflow for each catalog model when it is installed; new promising models as entries |
| ACE-Step cover / repaint (`source`, `strength`) as its own entry with its own workflow | |
| `openai_compatible` images and video: gpt-image and Sora 2 on OpenAI; gemini image, FLUX, Qwen image, Seedream, Veo 3.1 and Runway wherever a gateway serves them with the same endpoints | hosted transcription (`/audio/transcriptions` with word timestamps) as an `openai_compatible` transcription entry |
| `command` with the SongGeneration v2 adapter | DiffRhythm2: an adapter once its environment exists (it needs `espeak-ng`, which may need a user-space build: no sudo) |
| `faster_whisper` (extra `transcribe`) | calling the scoring models (their entries exist now), if open question 1 says so |
| `models list --installed / --missing / --tier`, `models install <id>` (prints; `--run` for the owner) | |
| guides, `lyrics_format` (`sections`, `levo`, `plain`), `prompt_inputs`, `mk.guide`, `models guide`, `--feature` | more converters when a second model shares a format |
| 0016 adapted as in §7 | media replay; per-call RAM scheduling |

Suggested order of work (each step its own pull request, `scripts/check.sh` green): (1) media client,
records, `comfyui` provider, `FakeComfyUI`, `mk.session("comfyui")`; (2) hosted images and video; (3)
`faster_whisper` and the `Transcriber`; (4) the `command` provider and the LeVo adapter; (5) guides,
formats and prompt inputs, then the packaged entries with their workflows (each proven by
`models check` on this machine); (6) the §7 hooks for 0016. Size: about 1,400 lines of source in new
modules (`media.py`, `transcribe.py`, `guide.py`, `formats.py`,
`providers/comfyui.py`, `providers/openai_media.py`, `providers/command.py`,
`providers/faster_whisper.py`, `testing/fake_comfyui.py`), each under the 300-line limit; no new core
dependency.

## Consequences

- Apps and experiments call every model the same way: a registry id, a call, a file, a span, a lease.
  OneShotStudio's D-004 is closed without an app adapter.
- Adding a ComfyUI model is a registry entry, an exported workflow file and a guide, no code; adding a
  standalone project is an entry plus a short adapter script.
- Every generated file is traceable: prompt, inputs, seed, workflow hash, output hash, time, cost.
- Outside a session each ComfyUI call reloads its model (seconds to a minute for the large ones), as a
  speech call does; batches use `session()`. The ComfyUI output folder keeps its own copies of outputs
  (ComfyUI has no delete endpoint); workflows save under a `hone/` prefix so they are easy to clean.
- Callers write the common inputs once (lyrics, duration, size, prompt inputs such as a camera angle) and
  hone-models converts them per model. For what cannot be shared, each model's guide says what it takes,
  with samples and a source, so prompt-writing code and experiments stop guessing. Guides go stale as
  models and providers change; `checked` and `models guide --stale` make that visible, nothing more.
- Workflows are tied to ComfyUI's node names and the model files on disk; a ComfyUI update that renames a
  node shows up as a `ConfigError` from `/prompt` validation, and the packaged workflows need an update.
- The `command` provider runs programs named in a registry file. Registry files are already trusted
  configuration (they name servers and key variables); the docs say so explicitly.
- A lease per local call means a ComfyUI job waits for GPU memory like any other model, and ComfyUI is
  asked to let go when another lease is short; the machine state (0016) can name its models.

## Migration and compatibility

Additive: new factories (`mk.image`, `mk.music`, `mk.video`, `mk.transcriber`), `mk.guide`, result types,
registry kinds, providers, capabilities and keys (`guide`, `lyrics_format`, `prompt_inputs`, `install`),
the `models guide` and `models install` commands and the `--kind` / `--feature` / `--installed` /
`--missing` / `--tier` filters, the packaged registry split into one file per kind, `price` fields (defaulting to 0 as the existing ones do), span
names and attributes, entry-point groups, the `transcribe` extra, fakes and acceptance cases. Existing
registries, spans, clients and the lease are unchanged; `mk.session` and `mk.unload` accept one more
provider. `mk.gpu.comfyui_free` stays.

Apps stop doing themselves: starting and stopping ComfyUI (comfy-cli or scripts), editing workflow
nodes, uploading files into ComfyUI's `input/` folder, polling jobs, calling `/free` between renders,
calling the image and video APIs, running Whisper, and most hand-placed `release()` calls. OneShotStudio's
`comfy.py`, `renderer.py` and its Whisper transcriber become registry entries and calls; its workflow
JSON files move next to its `hone-models.toml`. The UI-format templates it converted with
`comfy run --print-prompt` are exported once in API format instead.

## Owner answers (2026-09-29)

1. Clients per kind (`mk.image`, `mk.music`, `mk.video`, `mk.transcriber`): **yes**.
2. An input the model does not accept raises `ConfigError`: **yes**.
3. Starting ComfyUI: **only sessions start it**; a plain call without a running server fails and says how
   to start one (§2).
4. Free after each call: **yes** for a plain call; a session keeps the model loaded until it ends (§2).
5. Accepted jobs that fail are results, out of memory included: **yes**, and the result must say enough
   for the caller to decide what to do: `error_kind` (§1).
6. What ships: **more; every model on the machine** and, later in review, **every promising model not
   installed yet**: the catalog (§3b), with machine paths from variables.
7. Standalone-project adapters: **in hone-models**. Everything about calling a model with the inputs it
   takes, and managing its resources, belongs in hone-models (Decision).
8. `commercial_use`: **yes**, as information that never blocks a call, on every result and span, and
   usable in `require=` (§3).
9. Hosted prices: **a naive estimate is enough for now**, marked `cost_estimated` and documented as naive
   (§2).
10. (Added in review) Model-specific abilities, formats and trigger phrases go into the registry: common
    formats are converted where one fits, and every model has a guide with samples and a source (§3a).

## Open questions for the owner

1. **Calling the scoring models.** They are catalog entries now (§3b). hone-taste still loads them
   itself. By the rule "every model through hone-models" calling them belongs here too, as new kinds
   (`image_embedding`, `audio_score`, `separation`, ...) in a separate change record after this one.
   Agreed?
2. **Guides for hosted models that change often.** The proposal keeps a few samples and the source link,
   checked by hand. Enough, or should `models guide --refresh` fetch the source page and show what
   changed (network, and a page format that breaks)? Recommended: by hand for now.
