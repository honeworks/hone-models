# Images, music and video

`mk.image(model_id)`, `mk.music(model_id)` and `mk.video(model_id)` generate files with a registered
model. The three return the same client, `MediaClient`; the factory checks that the entry's `kind` is
the one asked for (`ConfigError` otherwise).

```python no-run
import hone_models as mk
from pathlib import Path

img = mk.image("z-image-turbo")
r = img.generate("a lighthouse at dusk, oil painting", size="1024x1024", seed=7, out="shots/01.png")
r.path, r.files, r.seed, r.error, r.error_kind, r.span_id, r.cost_usd

song = mk.music("ace-step-1.5-xl-turbo")
with song.session():  # lease, server and model held for the block
    for i, seed in enumerate([1, 2, 3]):
        song.generate(
            "dark trap, 95 bpm, female vocals",
            lyrics=text,
            duration_s=150,
            seed=seed,
            out=f"takes/take_{i}.flac",
        )

clip = mk.video("wan2.2-i2v-14b")
clip.generate("slow push-in, candle flicker", image=Path("shots/01.png"), duration_s=5, out="clips/01.mp4")
```

The packaged catalog lists these models with their install tables and guides, but their ComfyUI
workflows are not written yet: calling one says "no workflow yet for '<id>'" until its workflow is
exported and proven (`hone-models models check <id>`). Until then, add your own entries (below). Lyrics are
written once in a common format and converted per model, and inputs such as `camera_angle` become the
model's own phrases: see [models-and-guides.md](models-and-guides.md).

## Calling

`generate(prompt, *, out, seed=None, timeout_s=None, trace=None, **inputs) -> MediaResult`

- **Named inputs.** Besides `prompt` and `seed` there is a small shared vocabulary: `negative`, `size`
  (`"WxH"`), `references` (a list of image files), `image` (a start frame), `source` and `strength`
  (audio or video to transform, 0-1), `lyrics`, `duration_s`, `n`, `steps`. A model may take more
  (`bpm`, `key`, ...); `client.inputs` lists what it takes. **An input the model does not take raises
  `ConfigError` before any request**, naming the ones it takes: a silently ignored `duration_s` would
  waste minutes of GPU time. The entry's `defaults` fill inputs the call leaves out.
- **Files.** A `pathlib.Path` value is a file input (for `references`, `image` and `source` a string is
  a path too). A missing file raises `ConfigError` before anything runs; the span records each file as
  `{"path", "sha256", "bytes"}`, never its bytes.
- **Checks.** A size or duration the entry does not declare (`capabilities.sizes`, `durations_s`,
  `max_duration_s`) or more references than `max_references` raise `CapabilityError` before the request.
  Capabilities the entry leaves out are unknown, and unknown is allowed.
- **Seed.** `seed=None` takes the entry's `defaults.seed`, else a random one. The seed used is on the
  result and the span, so every output can be made again.
- **Output.** `out` is the file to write (parent folders are created). Several files (`n > 1`, stems) are
  `<stem>_1<suffix>`, `<stem>_2<suffix>`, ...; an `out` without a suffix takes the provider's.
- **Timeout.** `timeout_s` defaults to the entry's `max_timeout_s` (by kind: image 600 s, music 1800 s,
  video 3600 s). When it runs out the job is cancelled at the provider and `ModelTimeout` is raised; any
  other exception while waiting (`Ctrl-C` included) cancels the job too, then goes on.

## Results

`MediaResult` has `files` (a `MediaFile` per file), `path` (the first file, or `None`), `model`,
`seed`, `span_id`, `job_id` (the provider's), `elapsed_s`, `error`, `error_kind`, `cost_usd`,
`cost_estimated`, `license` and `commercial_use` (both from the registry, `None` when not declared).
`MediaFile` has `path`, `sha256`, `bytes`, `mime`, `width`, `height` and `duration_s`, each `None` when
unknown or not applicable, never 0. Files are recognised by their content: images by their header, WAV
and FLAC by their own headers, anything else (video) with `ffprobe` when it is on `PATH`.

**A job that ran without usable output is a result, not an exception**: `files` is empty, `error` holds
the provider's message and `error_kind` says what to do next:

| `error_kind` | Meaning |
|---|---|
| `out_of_memory` | retry later, smaller, or after freeing memory |
| `refused` | moderation or policy: change the prompt |
| `invalid_input` | the model rejected a value while running: change the input |
| `no_output` | the job finished without files |
| `failed` | anything else |

A job that could not be submitted, run or fetched raises: `ConfigError` for a workflow ComfyUI refuses
(its `node_errors`, naming the node and the input), `ProviderError` for a server that is not there, a
transport failure or an HTTP error, `ModelTimeout` for a timeout.

**Cost** is a naive estimate: one flat price per entry, `capabilities.price.per_image` times the images
or `per_output_second` times the seconds of output; `cost_estimated` is `True` when it is set, and
`cost_usd` is `None` without a price. Real hosted prices vary with size and quality.

## GPU and sessions

A local model's call takes a GPU lease by itself (`mk.gpu.lease(model_id, capabilities.vram_gb)`), so it
waits for memory like any other lease; hosted models take none. A plain call frees the model when it is
done (ComfyUI: `POST /free`). `with client.session():` holds the lease, the server and the loaded model
for the calls in the block and frees them once at the end, also on an exception; sessions of one client
do not nest.

ComfyUI keeps its models in another process and can only free everything at once. `mk.unload(model_id)`
for a ComfyUI entry sends `POST /free`, freeing every model that server holds. hone-models notes which
registry models each server ran since it last freed them in `${HONE_HOME}/models/comfyui-loaded.json`.
The first call to a server registers a release hook (`mk.gpu.on_short`), so a later lease in the same
process that is short of memory can make ComfyUI let go.

**Starting ComfyUI.** A plain call only uses a running server: without one it raises `ProviderError`
saying to start ComfyUI or to use a session. A session (`client.session()` or `mk.session("comfyui")`)
starts one when none answers and `HONE_COMFYUI_START` holds the command line that starts it:

```bash
export HONE_COMFYUI_START="$HOME/ComfyUI/start.sh --port 8188"   # your own start script
export HONE_COMFYUI_URL="http://127.0.0.1:8188"                  # the default
```

It runs in its own process group with a clean environment (no `VIRTUAL_ENV`, `LD_LIBRARY_PATH`,
`PYTHONPATH`, `PYTHONHOME`), hone-models waits up to `HONE_COMFYUI_START_S` seconds (300) for
`/system_stats` to answer, and stops it at the end of the block. A server that was already running is
never stopped, and a remote one is never started.

```python no-run
with mk.session("comfyui"):  # one server for several clients
    mk.image("z-image-turbo").generate("a harbour", out="a.png")
    mk.music("ace-step-1.5-turbo").generate("lo-fi, rain", duration_s=30, out="b.flac")
```

Models that offload to system RAM (Wan 14B) can fill it; the lease does not schedule RAM, so do not run
two such jobs at once.

## ComfyUI entries

A `comfyui` entry names a workflow exported from ComfyUI with **Export (API)** and maps each named input
to one or more `"<node id>.<input>"` paths of that workflow. A relative `workflow` path is resolved
against the registry file that declares it. The workflow's SHA-256 is recorded on every span, so a
changed workflow file shows in the records.

```toml
[models."my-image"]
provider = "comfyui"
kind = "image"
workflow = "workflows/my-image.json"           # relative to this file
outputs = ["9"]                                 # the save nodes whose files are the result (default: all)
defaults = { size = "1024x1024", steps = 8 }
[models."my-image".inputs]
prompt = "6.text"
seed = "3.seed"
steps = "3.steps"
width = "13.width"                              # `size` fills width and height
height = "13.height"
references = ["78.image", "106.image"]          # slots, in order; an unused slot is removed with its links
[models."my-image".capabilities]
vram_gb = 7.0
max_references = 2
license = "Apache-2.0"
commercial_use = true

[models."my-video".inputs]
duration_s = { path = "50.length", per_second = 16, add = 1 }   # seconds to frames: 5 s -> 81
```

- A file input is uploaded with `POST /upload/image` into `input/hone/` under its SHA-256 name, once per
  process and server, so a reference used for 40 shots is sent once; this also works with a ComfyUI on
  another host.
- The outputs are fetched with `GET /view` and written to `out`; ComfyUI keeps its own copies in its
  output folder (it has no delete endpoint), so save under a `hone/` prefix to find them.
- The job is queued with `POST /prompt` and `GET /history/{id}` is polled every second (every two after
  the first minute). ComfyUI's HTTP API reports no progress, so there are no progress events.
- An entry without a `workflow` raises `ConfigError` saying so; a mapping to a node or input the workflow
  does not have raises `ConfigError` naming it, before anything is queued.

A test run through the packaged fake server, with a workflow of three nodes:

```python
import json
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeComfyUI

Path("flows").mkdir(exist_ok=True)
Path("flows/tiny.json").write_text(
    json.dumps(
        {
            "3": {
                "class_type": "KSampler",
                "inputs": {"seed": 0, "positive": ["6", 0], "latent_image": ["13", 0]},
            },
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "13": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "hone/tiny", "images": ["3", 0]}},
        }
    )
)
Path("flows/models.toml").write_text("""
[models.tiny]
provider = "comfyui"
kind = "image"
workflow = "tiny.json"
inputs = { prompt = "6.text", seed = "3.seed", width = "13.width", height = "13.height" }
capabilities = { vram_gb = 0.5 }
""")
registry = mk.registry.load(["flows/models.toml"])
with FakeComfyUI() as server:
    r = mk.image("tiny", registry=registry).generate(
        "a harbour", size="96x64", seed=3, out="shots/harbour.png"
    )
print(r.path, r.files[0].width, r.files[0].height, r.seed)
assert server.submitted[0]["6"]["inputs"]["text"] == "a harbour"
assert (r.files[0].width, r.files[0].height) == (96, 64)
```

## Hosted images and video

An `openai_compatible` entry of kind `image` or `video` calls a hosted API with OpenAI's endpoints:
OpenAI itself, or any gateway that serves the same ones. The API key comes from the variable named by
`api_key_env` and is never recorded. Hosted models take no GPU lease.

```toml
[models."gpt-image-1.5"]
provider = "openai_compatible"
kind = "image"
model = "gpt-image-1.5"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
inputs = ["quality", "background"]              # extra inputs, sent as request fields
[models."gpt-image-1.5".capabilities]
sizes = ["1024x1024", "1536x1024", "1024x1536"]
price = { per_image = 0.04 }                    # a naive flat estimate

[models."sora-2"]
provider = "openai_compatible"
kind = "video"
model = "sora-2"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
defaults = { duration_s = 4, size = "720x1280" }
[models."sora-2".capabilities]
sizes = ["720x1280", "1280x720"]
durations_s = [4, 8, 12]
price = { per_output_second = 0.10 }
```

```python no-run
import hone_models as mk
from pathlib import Path

img = mk.image("gpt-image-1.5")
img.generate("a lighthouse at dusk, oil painting", size="1024x1024", quality="low", out="shots/01.png")
img.generate("the same lighthouse in winter", references=[Path("shots/01.png")], out="shots/02.png")

clip = mk.video("sora-2")
r = clip.generate("slow push-in", image=Path("shots/01.png"), duration_s=8, out="clips/01.mp4")
r.job_id, r.cost_usd, r.cost_estimated  # 'video_...', 0.8, True
```

- **Images** take `size`, `n` and `references`, plus the names the entry lists in `inputs`. Without
  references the call is `POST /images/generations`; with them `POST /images/edits`, one `image[]` part
  per reference. The images come back as `b64_json` (decoded straight to `out`) or as a `url`
  (downloaded; the API key is not sent to that host). No `response_format` is sent: models that can
  answer either way choose; to ask for one, list `response_format` in `inputs` and set it in `defaults`.
  A rewritten prompt (`revised_prompt`) is recorded on the span.
- **Video** takes `size`, `duration_s` (sent as `seconds`) and `image` (a start frame, sent as
  `input_reference`). The job is created with `POST /videos`, `GET /videos/{id}` is polled every
  5 seconds (a `progress` event on the span at most once per 10 %), and the file is fetched from
  `GET /videos/{id}/content`. `result.job_id` is the provider's id.
- **Sizes and durations** the entry declares are checked before the request, so Sora's "4, 8 or 12
  seconds" fails at once (`CapabilityError`) instead of after a paid request.
- **Failures.** Submitting retries connection errors, 429 and 5xx like every other call. A submitted
  video job is never submitted again: polling tolerates transient errors, and after five in a row
  raises `ProviderError` with the job id, so you can fetch the video yourself later. On a timeout, or
  any exception while waiting (`Ctrl-C` included), the job is deleted (`DELETE /videos/{id}`). A
  moderation refusal, at submission or as a failed job, is a result with `error_kind = "refused"`; a job
  with status `failed` is a result with the provider's message.
- **Cost** is naive: `per_image` times the images, or `per_output_second` times the seconds of video
  (measured with `ffprobe`, else the `duration_s` asked for); real prices vary by size and quality.

## Standalone projects (`command` entries)

Some models ship as research code with their own Python, torch and scripts (SongGeneration, DiffRhythm2):
importing them would break the caller's environment. A `command` entry runs one job as a subprocess in
the project's own environment instead, through a small file protocol. The process exit frees the GPU
memory, so there is no unload step; the call holds a GPU lease for the entry's `vram_gb` while it runs.

```toml
[models."songgeneration-v2-medium"]
provider = "command"
kind = "music"
command = ["$HONE_LEVO2_DIR/.venv/bin/python", "{adapter:levo2}", "{request}"]
inputs = ["generate_type"]                         # mixed | vocal | bgm | separate
defaults = { generate_type = "mixed", low_mem = true }
[models."songgeneration-v2-medium".install]
dir_env = "HONE_LEVO2_DIR"                         # the project folder; also the default cwd
[models."songgeneration-v2-medium".capabilities]
vram_gb = 7.5
```

- `command` is a list. `{request}` is the path of the job's `request.json`, `{out_dir}` the folder the
  program writes into, `{adapter:<name>}` a script shipped in `hone_models/data/adapters/` (`levo2`).
  `~`, `$VAR` and `${VAR}` are expanded in `command`, `cwd` and the `env` values; an unset variable is a
  `ConfigError` naming it.
- `install.dir_env` names the project folder variable: when it is unset or not a folder the call raises
  `ConfigError` before the lease, and `cwd` defaults to that folder. No machine paths go into the
  entry.
- The program runs in its own process group with the clean environment of `mk.session("ollama")`
  (without `VIRTUAL_ENV`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `PYTHONHOME`) plus the entry's `env`. Its
  stdin is empty.
- **Registry files are trusted configuration.** A `command` entry runs the program it names with your
  permissions, just as other entries name servers and key variables. Load only registry files you would
  run as a script.

**The protocol.** hone-models makes a fresh job folder next to `out` (`<out name>.job-<id>/`) and writes
`request.json` there:

```json
{"model": "songgeneration-v2-medium", "prompt": "female, pop, piano", "seed": 7,
 "out_dir": "/abs/takes/take.flac.job-3f2a9c1b04de/out",
 "inputs": {"lyrics": "[verse] ...", "generate_type": "mixed", "source": "/abs/ref.wav"},
 "defaults": {"generate_type": "mixed", "low_mem": true}}
```

`inputs` holds the checked inputs with the entry's defaults applied, file inputs as absolute paths;
`defaults` is the entry's whole `defaults` table, for options that are not inputs (`low_mem`). The
program writes its files into `out_dir` and may write `out_dir/result.json`:
`{"files": ["audios/take.flac"], "error": null, "meta": {...}}` (files relative to `out_dir`).

| The program | The call |
|---|---|
| `result.json` has an `error` (whatever the exit code) | `result.error`, `error_kind` from the message (`out_of_memory`, ...) |
| exits 0 with files | the files listed in `result.json`, else every file in `out_dir` by name, moved to `out` |
| exits 0 without files | `result.error`, `error_kind = "no_output"` |
| exits with another code, or is killed by a signal | `ProviderError` with the last 40 lines of stderr |
| runs past `timeout_s`, or the caller is interrupted | SIGTERM to the process group, SIGKILL 10 s later; `ModelTimeout` (or the interrupt re-raised) |

The last 40 lines of stderr (4 KiB at most, content) are recorded as `hone.models.media.log_tail`. The
job folder is removed after a success and kept after a failure, with `request.json`, `stdout.log` and
`stderr.log`, for a look at what went wrong.

**Writing an adapter.** An adapter is one script in `hone_models/data/adapters/<name>.py`, run with the
project's own Python (which may be older than hone-models' own: `levo2.py` runs on 3.10). It imports
only the standard library until the request is converted, so a bad request is reported without loading
the project; then it hands over:

1. Read `request.json`; convert it to the project's input. Report a request the project cannot take as
   `{"error": "..."}` in `result.json` and exit 0.
2. Set up what the project's own launch script sets (environment variables, `sys.path`, the working
   folder), import the project, and set the seeds it does not take as options (Python, NumPy, torch).
3. Run the generation; write `result.json` with the files, or with the error of a run that failed
   (out of memory, no GPU) and exit 0. A failure to import the project is left to crash: a non-zero exit
   with its traceback in stderr is a `ProviderError`, which is what it is.

`levo2.py` writes LeVo's JSONL (`idx`, `gt_lyric` = the `lyrics` input as given, already in LeVo's
section format; `descriptions` = the prompt, comma-separated tags), takes `generate_type` from the inputs
and `low_mem`, `flash_attn` (default off) and `checkpoint` (default `songgeneration_v2_medium`) from
`defaults`, and runs the same generation as `generate.sh <checkpoint> <jsonl> <out_dir> --low_mem
--not_use_flash_attn`. Test an adapter's conversion by calling its functions directly, and the whole
chain with a fake project folder (see `tests/integration/test_levo2_command.py`).

A command entry run end to end, with a tiny program in place of a project:

```python
import os
import sys
from pathlib import Path

import hone_models as mk

Path("tone").mkdir(exist_ok=True)
Path("tone/make.py").write_text("""
import json, sys, wave
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
with wave.open(str(Path(request["out_dir"]) / "tone.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
    w.writeframes(b"\\0\\0" * 8000 * int(request["inputs"]["duration_s"]))
print("made a tone", file=sys.stderr)
""")
Path("tone/models.toml").write_text("""
[models.tone]
provider = "command"
kind = "music"
command = ["$TONE_PYTHON", "make.py", "{request}"]
install = { dir_env = "TONE_DIR" }
capabilities = { vram_gb = 0.5 }
""")
os.environ["TONE_PYTHON"], os.environ["TONE_DIR"] = sys.executable, str(Path("tone").resolve())
song = mk.music("tone", registry=mk.registry.load(["tone/models.toml"]))
r = song.generate("a test tone", duration_s=2, out="takes/tone.wav")
print(r.path, r.files[0].duration_s, r.job_id)
assert r.files[0].duration_s == 2.0
```

## Records

Each call records one span, `hone.models.image`, `hone.models.music` or `hone.models.video`, with
`gen_ai.request.seed`, the prompt (`hone.models.media.prompt`, content), the inputs
(`hone.models.media.inputs`: files as `{"path", "sha256", "bytes"}`, texts such as lyrics are content),
the outputs (`hone.models.media.outputs`), the job id, the workflow hash, whether the call was in a
session, loaded the model, freed it or started the server, the time it queued, the error and its kind,
the license, `commercial_use`, and the naive cost; a hosted image model's `revised_prompt` (content) and a
hosted video job's `progress` events. See [records-and-replay.md](records-and-replay.md).
the license, `commercial_use`, the naive cost, and for `command` entries the end of stderr
(`hone.models.media.log_tail`). See [records-and-replay.md](records-and-replay.md).

## Testing

`hone_models.testing.FakeMedia` stands in for the clients in an application's tests: same validation,
span and result, no server and no lease. `FakeMedia.like(model_id)` copies a registry entry (its id,
kind, inputs, defaults and capabilities). It writes a black PNG of the requested `size`, silence of
`duration_s` or a one-second MP4, keeps each call in `calls` as `(prompt, inputs, seed)`, and
`fail_next(message)` makes the next call a failed job.

```python
from hone_models.testing import FakeMedia

song = FakeMedia(kind="music")
take = song.generate("lo-fi, rain", duration_s=3, seed=5, out="takes/take.wav")
print(take.path, take.files[0].duration_s, song.calls[0])
song.fail_next("CUDA out of memory")
failed = song.generate("lo-fi, rain", duration_s=3, out="takes/take.wav")
assert (failed.error_kind, failed.files) == ("out_of_memory", [])
```

`hone_models.testing.FakeComfyUI` is a local ComfyUI server for testing workflows and entries (see
[testing.md](testing.md)).
