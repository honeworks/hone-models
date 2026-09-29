# The model catalog and model guides

The packaged registry is a **catalog**: every model we use or may use, whether or not this machine has it,
from the writers on Ollama to the ComfyUI image, music and video models, the faster-whisper models, the
hosted image and video models and the scorers. Each entry says where the model comes from (`install`),
its licence and whether it may be used commercially (`commercial_use`), and carries a **guide**: what the
model is good at, how to write its prompt, what it can do beyond the common inputs, with a few examples
and a source. Whether a model is installed is checked on the machine, never claimed by the entry.

The catalog lives in `hone_models/data/models/<kind>.toml`, one file per kind. Your own files
(`~/.config/hone/models.toml`, `./hone-models.toml`) add entries or override single keys as before
([registry.md](registry.md)).

## Guides

```python
import hone_models as mk

g = mk.guide("qwen-image-edit-2511")
print(g.summary)  # what the model is good at
print(g.prompt)  # how to write its prompt
print(sorted(g.inputs))  # every input it takes: ['camera_angle']
print(g.features[0].name, g.features[0].examples[0], g.features[0].source)
print(g.license, g.commercial_use, g.max_references, g.installed)
print(g.as_text())  # one block of text, for a person or for an LLM writing prompts
assert g.as_dict()["inputs"]["camera_angle"]["origin"] == "prompt"  # the JSON form
```

`g.inputs` maps each input the model takes (besides `prompt` and `seed`) to its `origin` (`common`: the
shared vocabulary such as `size` or `lyrics`; `own`: the model's own, such as SongGeneration's
`generate_type`; `prompt`: a prompt input, below), a `note`, and for prompt inputs the `choices`. A
guide also carries the entry's `kind`, `sizes`, `durations_s`, `max_duration_s`, `max_references`,
`lyrics_format`, `license`, `commercial_use`, `installed` and the `install` commands, so one object
answers "what can I ask this model for". An entry without a `guide` table still has one, built from
its inputs and capabilities, with `summary = None`.

Features are samples, not catalogues: a guide says the model *can* do something, shows a few examples
and points to the source for the full, current list. `checked` is the date the guide was last compared
with its source; `hone-models models guide --stale 90` lists the guides not checked for 90 days.

A guide is documentation for whoever writes the prompt; it never changes a call.

## Choosing models by what they can do

`capabilities.features` holds the names of the guide's features, so selection can ask for them.
`require={"features": [...]}` is met only by models that declare **every** listed feature:

```python
import hone_models as mk

angles = mk.select({"features": ["camera angle"]}, kind="image")
print([m.id for m in angles])  # ['qwen-image-edit-2511']
songs = mk.select({"features": ["lyrics"]}, kind="music")
assert "songgeneration-v2-medium" in [m.id for m in songs]
assert mk.select({"features": ["camera angle", "lyrics"]}, kind="image") == []
```

`mk.select(require, kind=..., prefer=...)` returns every match, best first; `registry.select` returns the
single best one.

## Common input formats

Where one format fits many models, write it once and hone-models converts it for each model.

**Lyrics** have one common format: section tags on their own line (`[intro]`, `[verse]`,
`[pre-chorus]`, `[chorus]`, `[bridge]`, `[inst]`, `[outro]`), then one sung line per line. The entry's
`lyrics_format` names the converter: `sections` (as is: ACE-Step, HeartMuLa, YuE2, MiniMax-Music3),
`levo` (SongGeneration's `[verse] line. line. ; [chorus] ...`) or `plain` (tags removed). The media client
converts `lyrics` before the provider sees it, and the span records what the model got.

```python
import hone_models as mk
from hone_models.testing import FakeMedia

lyrics = "[intro]\n[verse]\nNeon on the water\nWe run until the morning\n[chorus]\nHold on\n[outro]\n"
print(mk.formats.convert_lyrics(lyrics, "levo"))
# [intro-short] ; [verse] Neon on the water. We run until the morning. ; [chorus] Hold on. ; [outro-short]
print(mk.formats.convert_lyrics(lyrics, "plain"))

song = FakeMedia.like("songgeneration-v2-medium")  # offline stand-in with the entry's inputs
song.generate("female, dark, hip hop", lyrics=lyrics, out="takes/take.wav")
assert song.calls[0][1]["lyrics"].startswith("[intro-short] ; [verse]")
```

**Prompt inputs** are named inputs that end up as words in the prompt, for models steered by fixed
phrases. `qwen-image-edit-2511` has `camera_angle`: `camera_angle="left_45"` writes the Multiple-Angles
LoRA's phrase for that camera position. The call says what it wants; the phrase is the model's business.
An unknown value raises `ConfigError` listing the choices.

```python
import hone_models as mk

cfg = mk.registry.load().get("qwen-image-edit-2511")
prompt, inputs = mk.formats.apply(cfg, "the same girl, seen from the side", {"camera_angle": "left_45"})
print(prompt)  # <sks> front-left quarter view eye-level shot medium shot, the same girl, seen from the side
assert inputs == {}  # the provider never sees camera_angle itself
print(sorted(mk.guide("qwen-image-edit-2511").inputs["camera_angle"].choices or {}))
```

`duration_s` and `size` are converted already (frames per second, width and height; see
[generation.md](generation.md#comfyui-entries)).

## Installed or not

`mk.catalog.installed(cfg)` answers `yes`, `no` or `unknown`, from the entry's `install` table:

| `install` gives | Installed means |
|---|---|
| `ollama = "hf.co/...:Q4_K_M"` (or an Ollama entry) | Ollama's `GET /api/tags` names the model |
| `files = [{ repo, file, to }]` | each file is in `<HONE_COMFYUI_DIR or ~/ComfyUI>/models/<to>/`; without that folder, ComfyUI's `GET /object_info` lists it |
| `hf = "org/name"` | the repository's snapshot is in the Hugging Face cache (`HF_HUB_CACHE`, `$HF_HOME/hub` or `~/.cache/huggingface/hub`) |
| `repo`, `setup`, `dir_env`, `check` (a `command` project) | the variable `dir_env` names a folder where `check` succeeds |

A server that does not answer, or a variable that is not set, is `unknown`, never `no`. A hosted model
has nothing to install (`yes`). A generation call to a model that is certainly not installed fails before
any job with `ConfigError` naming `hone-models models install <id>`; `unknown` goes ahead.

```python
import hone_models as mk

cfg = mk.registry.load().get("z-image-turbo")
print(mk.catalog.installed(cfg))  # yes, no or unknown on this machine
for command in mk.catalog.install_commands(cfg):
    print(command)  # hf download Comfy-Org/z_image_turbo ... --local-dir .../models/...
```

On the command line:

```text
hone-models models list --kind music --installed      # what can run here
hone-models models list --missing --tier 1            # what to fetch first
hone-models models list --feature "camera angle"      # which models can do it
hone-models models guide qwen-image-edit-2511 [--json]
hone-models models install flux.2-klein-4b            # prints the commands, size and free disk
```

`models install` downloads nothing: it prints the `ollama pull`, the `hf download ... --local-dir
<ComfyUI>/models/<folder>` lines (with a `mv` where the repository keeps the file in a subfolder) or the
clone and setup steps, with the size and the free disk space. `--run` runs them; it is for the owner of the
machine, never for code or agents on their own.

`tier` in `install` orders what to try: 1 test first, 2 worth a try, 3 to learn the ceiling. `note` says in
one line why the model is in the catalog.

## Models hone-models cannot call yet

The scorers and tools (reward models, DINOv2, MuQ, Demucs, SeedVR2, RIFE, ...) are catalog entries of kind
`scoring` with provider `none`: they have `install`, a licence and a guide, and every client refuses them
with `ConfigError` "no client for kind 'scoring' yet". A ComfyUI entry whose workflow is not written yet
(the models not installed on the reference machine) says "no workflow yet for '<id>'" when called: a
workflow is exported from ComfyUI with "Export (API)", set as `workflow` and proven with
`hone-models models check <id>` once the model is installed. The installed ones have packaged workflows,
listed with their measured VRAM in [generation.md](generation.md#packaged-comfyui-models).

```python
import hone_models as mk

try:
    mk.text("skywork-reward-v2-0.6b")
except mk.errors.ConfigError as exc:
    print(exc)  # no client for kind 'scoring' yet: ...
```

## Writing a catalog entry

A new promising model is one entry, no code:

```toml
[models."flux.2-klein-4b"]
provider = "comfyui"
kind = "image"
[models."flux.2-klein-4b".capabilities]
license = "Apache-2.0"
commercial_use = true
[models."flux.2-klein-4b".install]
source = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
files = [
  { repo = "Comfy-Org/vae-text-encorder-for-flux-klein-4b", file = "split_files/diffusion_models/flux-2-klein-4b.safetensors", to = "diffusion_models" },
]
size_gb = 16.1
tier = 1
note = "generation and reference editing in one small model; fits the GPU"
[models."flux.2-klein-4b".guide]
summary = "Small FLUX.2: text-to-image and multi-reference editing in one model, 4 steps."
prompt = "Natural-language description; for edits say what to change and what to keep from each reference."
source = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
checked = 2026-09-29
[[models."flux.2-klein-4b".guide.features]]
name = "reference editing"
how = "Give reference images and describe the change."
input = "references"
examples = ["the woman from image 1 wearing the jacket from image 2"]
source = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
```

A `command` project reads its folder from a variable (`dir_env = "HONE_LEVO2_DIR"`), so no machine path is
in the package. `prompt_inputs.<name>` takes `place` (`append` or `prepend`) and `choices` (value ->
phrase).

**Runnable examples:** [model_guides.py](../examples/model_guides.py).
