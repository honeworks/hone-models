"""Registry entries of generation models (change 0015 §3): kinds, providers, keys and capabilities."""

import datetime
from pathlib import Path

import pytest

from hone_models.errors import ConfigError
from hone_models.registry import load

ENTRIES = """
[models.img]
provider = "comfyui"
kind = "image"
workflow = "flows/img.json"
outputs = ["9"]
defaults = { size = "512x512" }
[models.img.inputs]
prompt = "6.text"
references = ["78.image", "106.image"]
duration_s = { path = "50.length", per_second = 16, add = 1 }
[models.img.capabilities]
max_references = 2
sizes = ["512x512"]
commercial_use = true
features = ["camera angle"]
price = { per_image = 0.04 }
[models.img.prompt_inputs.camera_angle]
choices = { left_45 = "turn left" }
[models.img.guide]
summary = "Edits pictures."
checked = 2026-09-29
[[models.img.guide.features]]
name = "camera angle"
examples = ["turn left"]
[models.img.install]
files = [{ repo = "org/repo", file = "m.safetensors", to = "diffusion_models" }]
size_gb = 7.8
tier = 1

[models.song]
provider = "command"
kind = "music"
command = ["python", "{adapter:levo2}", "{request}"]
cwd = "~/levo"
env = { A = "1" }
inputs = ["generate_type"]
lyrics_format = "levo"

[models.stt]
provider = "faster_whisper"
kind = "transcription"
[models.stt.capabilities]
word_timestamps = true
"""


def registry(tmp_path: Path, text: str = ENTRIES):
    path = tmp_path / "reg" / "models.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return load([path])


def test_generation_entries_load(tmp_path: Path) -> None:
    reg = registry(tmp_path)
    img, song, stt = reg.get("img"), reg.get("song"), reg.get("stt")
    assert img.workflow == str(tmp_path / "reg" / "flows" / "img.json")  # relative to the declaring file
    assert isinstance(img.inputs, dict)
    assert img.capabilities.max_references == 2
    dumped = img.model_dump()
    assert dumped["capabilities"]["price"]["per_image"] == 0.04
    assert dumped["prompt_inputs"]["camera_angle"]["place"] == "append"
    assert dumped["guide"]["checked"] == datetime.date(2026, 9, 29)
    assert dumped["install"]["files"][0]["to"] == "diffusion_models"
    assert (img.max_timeout_s, song.max_timeout_s, stt.max_timeout_s) == (600.0, 1800.0, 600.0)
    assert (img.local, song.local, stt.local) == (True, True, True)
    assert song.inputs == ["generate_type"]
    assert stt.capabilities.word_timestamps is True
    assert stt.capabilities.commercial_use is None  # unknown, not false


def test_timeout_default_by_kind_can_be_overridden(tmp_path: Path) -> None:
    text = (
        '[models.v]\nprovider = "comfyui"\nkind = "video"\n[models.w]\nprovider = "comfyui"\nkind = "video"\n'
    )
    reg = registry(tmp_path, text + "max_timeout_s = 60\n")
    assert (reg.get("v").max_timeout_s, reg.get("w").max_timeout_s) == (3600.0, 60.0)


def test_comfyui_url_and_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = (
        '[models.a]\nprovider = "comfyui"\nkind = "image"\n[models.b]\nprovider = "comfyui"\nkind = "image"\n'
    )
    reg = registry(tmp_path, text + 'base_url = "http://gpu-box:8188/"\n')
    assert reg.get("a").comfyui_url == "http://127.0.0.1:8188"
    assert (reg.get("b").comfyui_url, reg.get("b").local) == ("http://gpu-box:8188", False)
    monkeypatch.setenv("HONE_COMFYUI_URL", "http://127.0.0.1:9000")
    assert reg.get("a").comfyui_url == "http://127.0.0.1:9000"


def test_absolute_workflow_path_is_kept(tmp_path: Path) -> None:
    flow = tmp_path / "abs.json"
    reg = registry(tmp_path, f'[models.a]\nprovider = "comfyui"\nkind = "image"\nworkflow = "{flow}"\n')
    assert reg.get("a").workflow == str(flow)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ('provider = "comfyui"\nkind = "image"\ninputs = ["prompt"]', "table of workflow paths"),
        ('provider = "comfyui"\nkind = "image"\ninputs = { prompt = "six" }', "not a workflow path"),
        ('provider = "comfyui"\nkind = "image"\ninputs = { d = { path = "x" } }', "not a workflow path"),
        ('provider = "comfyui"\nkind = "image"\ncommand = ["x"]', "command is for command entries"),
        ('provider = "command"\nkind = "music"\ninputs = { a = "1.b" }', "list of input names"),
        ('provider = "command"\nkind = "music"\nworkflow = "w.json"', "workflow is for comfyui"),
        ('provider = "openai_compatible"\nkind = "image"\ncwd = "x"', "cwd is for command"),
        ('provider = "comfyui"\nkind = "image"\nlyrics_format = "rhymes"', "lyrics_format"),
        ('provider = "comfyui"\nkind = "image"\n[models.x.install]\ntier = 4', "tier"),
        ('provider = "comfyui"\nkind = "image"\n[models.x.guide]\nsumary = "typo"', "sumary"),
        ('provider = "comfyui"\nkind = "image"\n[models.x.prompt_inputs.a]\nchoices = {}', "choices"),
    ],
)
def test_bad_generation_entries(tmp_path: Path, entry: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        registry(tmp_path, f"[models.x]\n{entry}\n")
