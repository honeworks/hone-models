"""Model guides and the catalog: what each model can take, which models can do something, and what is
installed.

What: read a model's guide, choose the models that declare a feature, send lyrics written once to two
    music models that want them in different forms, ask for a camera angle by name, and check which
    catalog models are installed and what would install the others.

How: `mk.guide(id)` returns a `ModelGuide` (summary, prompt advice, every accepted input with its note,
    features with examples and a source, limits, licence, install state; `as_text()` for a person or an
    LLM). `mk.select({"features": [...]}, kind=...)` returns the models that declare every listed feature.
    An entry's `lyrics_format` and `prompt_inputs` make the media client convert `lyrics` and write
    `camera_angle` into the prompt before the model sees them. `mk.catalog.installed(cfg)` says `yes`,
    `no` or `unknown`; `mk.catalog.install_commands(cfg)` lists what would fetch the model.

Why: the packaged registry is a catalog of every model we use or may use, installed or not. Code that
    writes prompts, and experiments comparing models, need the same facts a person reads on a model card:
    what the model is good at, its phrases and formats, its licence. Writing lyrics or a camera angle once
    and letting hone-models convert them keeps model details out of the app. Pitfall: a guide is
    documentation; it never changes a call, and it goes stale (`hone-models models guide --stale 90`).

Runs offline: `FakeMedia` stands in for the music and image clients, and a folder made here stands in for
    ComfyUI's model folder (`HONE_COMFYUI_DIR`).
"""

import os
import tempfile
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeMedia

# 1. A model's guide.
g = mk.guide("qwen-image-edit-2511")
print(g.as_text().splitlines()[1])
angle = g.features[0]
print(f"feature: {angle.name}; e.g. {angle.examples[0]}; source {angle.source}")
assert angle.input == "camera_angle"
assert g.commercial_use is True

# 2. Which models can do it: every listed feature must be declared.
print("camera angle:", [m.id for m in mk.select({"features": ["camera angle"]}, kind="image")])
singers = [m.id for m in mk.select({"features": ["lyrics"]}, kind="music")]
print("lyrics:", singers)
assert {"ace-step-1.5-turbo", "heartmula-3b"} <= set(singers)
assert "songgeneration-v2-medium" not in singers  # disabled until its runtime matches (D-080)

# 3. Lyrics written once in the common format, received by each model in its own form.
lyrics = "[intro]\n[verse]\nNeon on the water\nWe run until the morning\n[chorus]\nHold on\n[outro]\n"
work = Path(tempfile.mkdtemp())
for model_id in ("ace-step-1.5-turbo", "songgeneration-v2-medium"):
    song = FakeMedia.like(model_id)
    song.generate("dark pop, female vocals", lyrics=lyrics, out=work / f"{model_id}.wav")
    sent = song.calls[0][1]["lyrics"]
    print(f"{model_id} got: {sent!r}")
assert sent.startswith("[intro-short] ; [verse] Neon on the water. We run")  # SongGeneration's form

# 4. A camera angle by name: the model's own phrase is written into the prompt.
edit = FakeMedia.like("qwen-image-edit-2511")
edit.generate("the same girl", camera_angle="left_45", out=work / "left.png")
print("prompt sent:", edit.calls[0][0])
assert edit.calls[0][0].startswith("<sks> front-left quarter view")
try:
    edit.generate("the same girl", camera_angle="from_space", out=work / "space.png")
except mk.errors.ConfigError as exc:
    print("refused:", str(exc)[:80], "...")

# 5. Installed or not, and what would install it (nothing is downloaded).
models = work / "ComfyUI" / "models"
z_image_files = {
    "diffusion_models": "z_image_turbo_bf16.safetensors",
    "text_encoders": "qwen_3_4b.safetensors",
    "vae": "ae.safetensors",
}
for folder, name in z_image_files.items():
    (models / folder).mkdir(parents=True, exist_ok=True)
    (models / folder / name).write_bytes(b"weights")
os.environ["HONE_COMFYUI_DIR"] = str(work / "ComfyUI")
registry = mk.registry.load()
assert mk.catalog.installed(registry.get("z-image-turbo")) == "yes"
klein = registry.get("flux.2-klein-4b")
print("flux.2-klein-4b installed:", mk.catalog.installed(klein))
for command in mk.catalog.install_commands(klein)[:2]:
    print("  ", command)
assert mk.catalog.installed(klein) == "no"
