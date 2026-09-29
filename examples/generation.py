"""Generation: images and music through a ComfyUI workflow, one registry entry per model.

What: make an image from a prompt and a reference picture, three song takes in one session, and read
    what a failed job says, all through registry entries that map named inputs onto ComfyUI workflows.

How: a `comfyui` entry names an API-format workflow (ComfyUI's "Export (API)") and maps each input to
    `"<node id>.<input>"` paths: `size` fills `width` / `height`, `references` fills a list of image
    slots (unused ones are removed), one input may fill several nodes. `mk.image(id)` / `mk.music(id)`
    return a client; `generate(prompt, *, out, seed=None, **inputs)` runs one job and returns a
    `MediaResult` (`files` with hash, size, dimensions or duration; `seed`; `error` / `error_kind`;
    `span_id`). `with client.session():` keeps the model loaded and the GPU lease held for the block.

Why: every generated file should be traceable (prompt, inputs, seed, workflow hash, output hash) and every
    local job should wait for GPU memory like any other model call. An input the model does not take is a
    `ConfigError` before anything runs, because a silently ignored `duration_s` wastes minutes of GPU
    time; a job that ran out of memory is a result with `error_kind = "out_of_memory"`, so a script can
    retry, fall back or give up. Pitfall: outside a session every call reloads the model and frees it
    afterwards (ComfyUI can only free everything at once); batch calls in a session.

Runs offline: `FakeComfyUI` is a local ComfyUI server whose jobs write a black PNG or silence. To use a
    real ComfyUI, drop the `FakeComfyUI` block and point `HONE_COMFYUI_URL` at your server (default
    http://127.0.0.1:8188); set `HONE_COMFYUI_START` to let a session start it.
"""

import json
import struct
import zlib
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeComfyUI

IMAGE_FLOW = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "positive": ["20", 0], "latent_image": ["13", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "13": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024}},
    "20": {
        "class_type": "ReferenceEncode",
        "inputs": {"text": ["6", 0], "image1": ["78", 0], "image2": ["106", 0]},
    },
    "78": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "106": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "hone/edit", "images": ["3", 0]}},
}
SONG_FLOW = {
    "94": {"class_type": "TextEncodeAceStepAudio", "inputs": {"tags": "", "lyrics": "", "duration": 60}},
    "98": {"class_type": "EmptyAceStepLatentAudio", "inputs": {"seconds": 60}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "positive": ["94", 0], "latent_image": ["98", 0]}},
    "111": {"class_type": "SaveAudio", "inputs": {"filename_prefix": "hone/song", "audio": ["3", 0]}},
}
REGISTRY = """
[models.editor]
provider = "comfyui"
kind = "image"
workflow = "flows/editor.json"
defaults = { size = "512x512" }
capabilities = { vram_gb = 1.0, max_references = 2, license = "Apache-2.0", commercial_use = true }
[models.editor.inputs]
prompt = "6.text"
seed = "3.seed"
width = "13.width"
height = "13.height"
references = ["78.image", "106.image"]

[models.songwriter]
provider = "comfyui"
kind = "music"
workflow = "flows/song.json"
capabilities = { vram_gb = 1.0, max_duration_s = 600 }
[models.songwriter.inputs]
prompt = "94.tags"
lyrics = "94.lyrics"
seed = "3.seed"
duration_s = ["94.duration", "98.seconds"]
"""

Path("flows").mkdir(exist_ok=True)
Path("flows/editor.json").write_text(json.dumps(IMAGE_FLOW))
Path("flows/song.json").write_text(json.dumps(SONG_FLOW))
Path("hone-models.toml").write_text(REGISTRY)  # the project registry file, read by default


def a_reference_picture(path: Path) -> Path:
    """A 16 x 16 black PNG to stand for a character sheet."""
    rows = zlib.compress(bytes(16 * 17))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", 16, 16, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", rows) + chunk(b"IEND", b"")
    )
    return path


sink = mk.records.MemorySink()
front = a_reference_picture(Path("front.png"))

with FakeComfyUI() as server:
    # 1. An image from a prompt and one reference: the second reference slot is removed from the job.
    editor = mk.image("editor", sink=sink)
    print("editor takes:", editor.inputs)
    r = editor.generate("the same girl, seen from behind", references=[front], seed=7, out="sheet/back.png")
    shot = r.files[0]
    print(f"{r.path}: {shot.width}x{shot.height}, sha256 {shot.sha256[:12]}..., seed {r.seed}")
    filled = server.submitted[-1]
    assert filled["6"]["inputs"]["text"] == "the same girl, seen from behind"
    assert filled["78"]["inputs"]["image"].startswith("hone/")  # uploaded under its SHA-256 name
    assert "106" not in filled  # the unused slot and its link are gone
    assert (shot.width, shot.height) == (512, 512)  # the entry's default size
    assert r.commercial_use is True

    # 2. An input the model does not take fails before any request.
    try:
        editor.generate("a harbour", duration_s=5, out="harbour.png")
    except mk.errors.ConfigError as exc:
        print("refused:", str(exc)[:70], "...")

    # 3. Three takes in one session: the model is loaded once and freed once at the end.
    songwriter = mk.music("songwriter", sink=sink)
    frees_before = server.frees
    with songwriter.session():
        takes = [
            songwriter.generate("lo-fi, rain, soft piano", lyrics="[verse]\nrain on the glass", duration_s=4,
                                seed=seed, out=f"takes/take_{seed}")  # ComfyUI's suffix is added
            for seed in (1, 2, 3)
        ]  # fmt: skip
    print("takes:", [str(t.path) for t in takes], "seconds:", takes[0].files[0].duration_s)
    assert server.frees == frees_before + 1
    assert server.submitted[-1]["98"]["inputs"]["seconds"] == 4  # one input, two nodes

    # 4. A job that ran out of memory is a result: decide what to do from `error_kind`.
    server.queue("execution_error", type="torch.OutOfMemoryError", message="CUDA out of memory.")
    failed = songwriter.generate("lo-fi", duration_s=4, out="takes/failed")
    print("failed job:", failed.error_kind, "-", failed.error)
    assert (failed.error_kind, failed.files) == ("out_of_memory", [])

# Every call is one span: the inputs (files by hash), the outputs and the workflow hash.
spans = [s for s in sink.spans if s["name"].startswith("hone.models.")]
print("spans:", [s["name"] for s in spans])
attrs = spans[0]["attributes"]
print("workflow sha256:", attrs["hone.models.media.workflow_sha256"][:12], "...")
assert attrs["hone.models.media.inputs"]["references"][0]["bytes"] == front.stat().st_size
assert spans[-1]["status"]["code"] == "error"
