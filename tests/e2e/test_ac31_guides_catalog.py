"""AC-31: guides, formats and the catalog - installed status from FakeOllama tags, a fake ComfyUI folder
and a fake HF cache; install commands printed, nothing downloaded; uninstalled and client-less models
refused before any job; lyrics converted per entry; prompt inputs; guides, `--feature` and
`require={"features": [...]}`; an entry without a guide."""

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import hone_models as mk
from hone_models.cli import app
from hone_models.errors import ConfigError
from hone_models.testing import FakeComfyUI, FakeMedia, FakeOllama
from media_fixtures import LeaseRecorder

pytestmark = pytest.mark.e2e
runner = CliRunner()
CATALOG = Path(__file__).parents[1] / "fixtures" / "catalog" / "models.toml"
LYRICS = (
    "[intro]\n[verse]\nNeon on the water\nWe run until the morning\n[chorus]\nHold on, hold on\n[outro]\n"
)
LEFT_45 = "<sks> front-left quarter view eye-level shot medium shot"


@pytest.fixture
def machine(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fake ComfyUI folder, HF cache and Ollama: `cat-image-edit`, the songs, `cat-whisper` and
    `cat-chat-here` are installed; `cat-missing-image`, `cat-hf-absent` and `cat-chat-absent` are not."""
    comfy = isolated / "ComfyUI" / "models"
    for folder, name in (
        ("diffusion_models", "cat_edit.safetensors"),
        ("diffusion_models", "song.safetensors"),
    ):
        (comfy / folder).mkdir(parents=True, exist_ok=True)
        (comfy / folder / name).write_bytes(b"weights")
    snapshot = isolated / "hf" / "models--org--whisper-x" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"weights")
    monkeypatch.setenv("HONE_COMFYUI_DIR", str(isolated / "ComfyUI"))
    monkeypatch.setenv("HF_HUB_CACHE", str(isolated / "hf"))
    monkeypatch.setattr(mk.gpu, "GPU", LeaseRecorder())

    def tags(path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        return {"models": [{"name": "hf.co/org/Here-GGUF:Q4_K_M"}]} if path == "/api/tags" else None

    with FakeOllama(tags):
        yield isolated


def rows(*options: str) -> dict[str, dict[str, Any]]:
    result = runner.invoke(app, ["models", "list", "--json", "--registry", str(CATALOG), *options])
    assert result.exit_code == 0, result.output
    return {r["id"]: r for r in json.loads(result.output) if r["id"].startswith("cat-")}


def test_ac31_installed_yes_no_unknown_as_the_fakes_say(machine: Path) -> None:
    status = {mid: row["installed"] for mid, row in rows().items()}
    assert status == {
        "cat-image-edit": "yes", "cat-song-sections": "yes", "cat-song-levo": "yes",  # ComfyUI folder
        "cat-missing-image": "no", "cat-bare": "unknown",  # no install table: never "no" by guess
        "cat-chat-here": "yes", "cat-chat-absent": "no",  # Ollama tags
        "cat-chat-remote": "unknown",  # its server does not answer
        "cat-whisper": "yes", "cat-hf-absent": "no",  # Hugging Face cache
        "cat-project": "unknown",  # HONE_CAT_PROJECT is not set
        "cat-scorer": "no",
    }  # fmt: skip
    installed, missing = rows("--installed"), rows("--missing")
    assert set(installed) == {
        "cat-image-edit",
        "cat-song-sections",
        "cat-song-levo",
        "cat-chat-here",
        "cat-whisper",
    }
    assert set(missing) == {"cat-missing-image", "cat-chat-absent", "cat-hf-absent", "cat-scorer"}
    assert set(rows("--tier", "3")) == {"cat-chat-absent"}
    assert set(rows("--kind", "transcription")) == {"cat-whisper", "cat-hf-absent"}


def test_ac31_install_prints_commands_and_downloads_nothing(
    machine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"ran {args}")

    monkeypatch.setattr(subprocess, "run", no_download)
    before = sorted(p for p in machine.rglob("*"))
    target = machine / "ComfyUI" / "models"

    def install(model_id: str) -> str:
        result = runner.invoke(app, ["models", "install", model_id, "--registry", str(CATALOG)])
        assert result.exit_code == 0, result.output
        return result.output

    ollama = install("cat-chat-absent")
    assert "ollama pull hf.co/org/Absent-GGUF:Q4_K_M" in ollama
    assert "size 5.0 GB; free on" in ollama
    comfy = install("cat-missing-image")
    assert (
        f"hf download org/missing split_files/diffusion_models/missing.safetensors --local-dir "
        f"{target / 'diffusion_models'}"
    ) in comfy
    assert f"mv {target / 'diffusion_models/split_files/diffusion_models/missing.safetensors'}" in comfy
    assert f"hf download org/missing vae.safetensors --local-dir {target / 'vae'}" in comfy
    assert "size 12.5 GB" in comfy
    project = install("cat-project")
    assert "git clone https://example.org/cat-project.git $HONE_CAT_PROJECT" in project
    assert "cd $HONE_CAT_PROJECT && .venv/bin/pip install -r requirements.txt" in project
    assert "size 3.0 GB" in project
    assert sorted(p for p in machine.rglob("*")) == before  # nothing downloaded or created


def test_ac31_uninstalled_model_and_clientless_kind_refused_before_any_job(machine: Path) -> None:
    reg = mk.registry.load([CATALOG])
    with FakeComfyUI() as server:
        img = mk.image("cat-missing-image", registry=reg)
        with pytest.raises(ConfigError, match="hone-models models install cat-missing-image"):
            img.generate("a lighthouse", out=machine / "x.png")
        assert server.requests == []  # before any job
    with pytest.raises(ConfigError, match="no client for kind 'scoring' yet"):
        mk.text("skywork-reward-v2-0.6b")
    with pytest.raises(ConfigError, match="no client for kind 'scoring' yet"):
        mk.image("cat-scorer", registry=reg)


def test_ac31_lyrics_in_the_common_format_reach_each_model_in_its_own_form(machine: Path) -> None:
    reg = mk.registry.load([CATALOG])
    sink = mk.records.MemorySink()
    with FakeComfyUI() as server:
        mk.music("cat-song-sections", registry=reg, sink=sink).generate(
            "pop", lyrics=LYRICS, out=machine / "a.wav"
        )
        mk.music("cat-song-levo", registry=reg, sink=sink).generate(
            "pop", lyrics=LYRICS, out=machine / "b.wav"
        )
    sections, levo = (job["94"]["inputs"]["lyrics"] for job in server.submitted)
    assert sections == LYRICS
    assert levo == (
        "[intro-short] ; [verse] Neon on the water. We run until the morning. ; "
        "[chorus] Hold on, hold on. ; [outro-short]"
    )
    assert sink.spans[1]["attributes"]["hone.models.media.inputs"]["lyrics"] == levo  # what the model got
    song = FakeMedia.like("songgeneration-v2-medium")  # the packaged LeVo entry
    song.generate("female, dark, hip hop", lyrics=LYRICS, out=machine / "c.wav")
    assert song.calls[0][1]["lyrics"] == levo


def test_ac31_prompt_input_known_and_unknown_choice(machine: Path) -> None:
    reg = mk.registry.load([CATALOG])
    img = mk.image("cat-image-edit", registry=reg)
    assert "camera_angle" in img.inputs
    with FakeComfyUI() as server:
        img.generate("the same girl", camera_angle="left_45", out=machine / "a.png")
        (job,) = server.submitted
        assert job["6"]["inputs"]["text"] == f"the same girl, {LEFT_45}"  # the phrase appended
        with pytest.raises(ConfigError, match=r"camera_angle='from_space'.*\['left_45', 'top_down'\]"):
            img.generate("the same girl", camera_angle="from_space", out=machine / "b.png")
        assert len(server.submitted) == 1


def test_ac31_guide_lists_inputs_features_and_limits(machine: Path) -> None:
    reg = mk.registry.load([CATALOG])
    g = mk.guide("cat-image-edit", registry=reg)
    assert (g.summary, g.prompt, g.kind, g.installed) == (
        "Edits up to two reference images.", "Plain sentences saying what to change.", "image", "yes",
    )  # fmt: skip
    assert set(g.inputs) == {"camera_angle", "references", "size", "steps"}  # every accepted input
    assert all(item.note for item in g.inputs.values())  # each with its note
    assert (g.inputs["steps"].note, g.inputs["steps"].origin) == ("4 to 8", "common")
    assert g.inputs["camera_angle"].origin == "prompt"
    assert set(g.inputs["camera_angle"].choices or {}) == {"left_45", "top_down"}
    (feature,) = g.features
    assert (feature.name, feature.input, feature.examples, feature.source) == (
        "camera angle", "camera_angle", [LEFT_45], "https://example.org/lora-card",
    )  # fmt: skip
    assert (g.license, g.commercial_use, g.max_references) == ("Apache-2.0", True, 2)
    text = g.as_text()
    for part in (
        "Edits up to two reference images.",
        "camera_angle",
        LEFT_45,
        "https://example.org/lora-card",
    ):
        assert part in text

    result = runner.invoke(app, ["models", "guide", "cat-image-edit", "--json", "--registry", str(CATALOG)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["inputs"]["camera_angle"]["choices"]["left_45"] == LEFT_45
    assert data["features"][0]["examples"] == [LEFT_45]
    assert data["checked"] == "2026-09-29"


def test_ac31_selection_by_feature_and_an_entry_without_a_guide(machine: Path) -> None:
    reg = mk.registry.load([CATALOG])
    found = mk.select({"features": ["camera angle"]}, kind="image", registry=reg)
    assert {m.id for m in found} == {"cat-image-edit", "qwen-image-edit-2511"}
    both = mk.select({"features": ["camera angle", "multi-image composition"]}, kind="image", registry=reg)
    assert [m.id for m in both] == ["qwen-image-edit-2511"]  # every listed feature, not any
    assert reg.select({"features": ["camera angle"]}, kind="image").id == "cat-image-edit"
    listed = rows("--feature", "camera angle")
    assert set(listed) == {"cat-image-edit"}
    assert all("camera angle" in r["capabilities"]["features"] for r in listed.values())

    bare = mk.guide("cat-bare", registry=reg)
    assert (bare.summary, bare.features, bare.installed) == (None, [], "unknown")
    assert set(bare.inputs) == {"size", "steps"}  # built from its inputs table
    assert bare.sizes == ["512x512", "1024x1024"]
    assert "cat-bare (image)" in bare.as_text()
