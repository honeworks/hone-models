"""`hone-models models check` for image, music and video entries (D-072): one tiny job through FakeComfyUI,
only the inputs the entry takes, the file measured, the peak GPU memory from a scripted reader."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hone_models as mk
from hone_models import _media_check, _media_files
from hone_models.cli import app
from hone_models.testing import FakeComfyUI, FakeMedia
from media_fixtures import COMFYUI_FIXTURES, LeaseRecorder

runner = CliRunner()
REGISTRY = ["--registry", str(COMFYUI_FIXTURES / "models.toml")]


@pytest.fixture(autouse=True)
def lease(monkeypatch: pytest.MonkeyPatch) -> LeaseRecorder:
    recorder = LeaseRecorder()
    monkeypatch.setattr(mk.gpu, "GPU", recorder)
    return recorder


def memory(monkeypatch: pytest.MonkeyPatch, used_mb: list[int] | None) -> None:
    """Script GPU 0's used memory: the first value, then the others as the sampler reads them."""
    readings = iter(used_mb or [])
    last = [0]

    def read() -> tuple[int, int] | None:
        if used_mb is None:
            return None
        last[0] = next(readings, last[0])
        return 8192, last[0]

    monkeypatch.setattr(_media_check, "read_memory", read)
    monkeypatch.setattr(_media_check, "SAMPLE_S", 0.01)


def test_image_check_runs_a_tiny_job_and_reports_the_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory(monkeypatch, [300, 2300, 6700, 300])
    with FakeComfyUI(run_s=0.2) as server:
        args = ["models", "check", "test-image", "--json", "--out", str(tmp_path), *REGISTRY]
        result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    found = json.loads(result.output)
    assert (found["id"], found["kind"], found["inputs"]) == ("test-image", "image", {"size": "256x256"})
    assert (found["width"], found["height"], found["mime"]) == (256, 256, "image/png")
    assert Path(found["path"]) == tmp_path / "test-image.png"
    assert found["peak_vram_gb"] == round((6700 - 300) / 1024, 2)
    assert found["error"] is None
    submitted = server.submitted[0]
    assert (submitted["13"]["inputs"]["width"], submitted["3"]["inputs"]["seed"]) == (256, 1)
    assert server.requests[-1]["path"] == "/free"  # the session freed ComfyUI at its end


def test_music_and_video_checks_pass_only_what_the_entry_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory(monkeypatch, None)  # no NVML and no nvidia-smi: the peak is unknown, never 0
    registry = mk.registry.load([COMFYUI_FIXTURES / "models.toml"])
    with FakeComfyUI() as server:
        song = _media_check.check(mk.music("test-music", registry=registry), tmp_path)
        clip = _media_check.check(mk.video("test-video", registry=registry), tmp_path)
    assert song["inputs"] == {"duration_s": "10", "lyrics": _media_check.LYRICS}
    assert (song["duration_s"], song["peak_vram_gb"]) == (10.0, None)
    assert set(clip["inputs"]) == {"size", "duration_s", "image"}
    assert server.submitted[1]["50"]["inputs"]["length"] == 17  # 1 s at 16 fps, + 1
    frame = tmp_path / "start-frame.png"
    assert _media_files.measure(frame).width == 256
    assert frame.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_music_check_sends_no_start_frame_to_an_entry_that_takes_every_shared_input(tmp_path: Path) -> None:
    song = FakeMedia.like("songgeneration-v2-medium")  # a command entry: it takes the whole vocabulary
    assert "image" in song.inputs
    found = _media_check.check(song, tmp_path)
    assert found["inputs"] == {"duration_s": "10", "lyrics": _media_check.LYRICS}
    assert "image" not in song.calls[0][1]
    assert not (tmp_path / "start-frame.png").exists()


def test_a_failed_tiny_job_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory(monkeypatch, [300])
    with FakeComfyUI() as server:
        server.queue("execution_error", message="CUDA out of memory")
        result = runner.invoke(app, ["models", "check", "test-image", "--out", str(tmp_path), *REGISTRY])
    assert result.exit_code == 1
    assert "test-image" in result.output
    assert "out of memory" in result.output.lower()


def test_default_out_folder_is_under_hone_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HONE_HOME", str(tmp_path / "home"))
    assert _media_check.default_out_dir() == tmp_path / "home" / "models" / "checks"
