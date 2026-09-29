"""AC-24: images, music and video through FakeComfyUI - inputs on the mapped nodes, a reference uploaded
once, outputs measured, one span each, a lease with the registry's vram_gb, /free after a plain call and
once after a session."""

import hashlib
from pathlib import Path

import pytest

import hone_models as mk
from hone_models.testing import FakeComfyUI
from media_fixtures import COMFYUI_FIXTURES, LeaseRecorder, png_file, registry

pytestmark = pytest.mark.e2e


@pytest.fixture
def lease(monkeypatch: pytest.MonkeyPatch) -> LeaseRecorder:
    recorder = LeaseRecorder()
    monkeypatch.setattr(mk.gpu, "GPU", recorder)
    return recorder


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ac24_image_with_a_reference_used_twice(tmp_path: Path, lease: LeaseRecorder) -> None:
    sink = mk.records.MemorySink()
    ref = png_file(tmp_path / "sheet" / "front.png")
    img = mk.image("test-image", registry=registry(), sink=sink)
    with FakeComfyUI() as server:
        r = img.generate(
            "a lighthouse", references=[ref], size="640x480", seed=7, out=tmp_path / "shots/01.png"
        )
        assert server.frees == 1  # a plain call frees ComfyUI afterwards
        img.generate("the same, from behind", references=[ref, ref], out=tmp_path / "shots/02.png")
        uploads = [q for q in server.requests if q["path"] == "/upload/image"]
    first, second = server.submitted
    name = f"hone/{sha256(ref)}.png"
    assert first["6"]["inputs"]["text"] == "a lighthouse"
    assert (first["3"]["inputs"]["seed"], first["3"]["inputs"]["steps"]) == (7, 8)  # steps: registry default
    assert (first["13"]["inputs"]["width"], first["13"]["inputs"]["height"]) == (640, 480)
    assert first["78"]["inputs"]["image"] == name
    assert "106" not in first  # the unused reference slot is removed with its link
    assert "image2" not in first["20"]["inputs"]
    assert second["78"]["inputs"]["image"] == second["106"]["inputs"]["image"] == name
    assert (second["13"]["inputs"]["width"], second["13"]["inputs"]["height"]) == (512, 512)  # default size
    assert len(uploads) == 1  # the reference is uploaded once, by hash
    assert list(server.uploads) == [name]

    assert r.error is None
    assert r.path == tmp_path / "shots" / "01.png"
    (out,) = r.files
    assert (out.width, out.height, out.mime) == (640, 480, "image/png")
    assert (out.sha256, out.bytes) == (sha256(out.path), out.path.stat().st_size)
    assert (r.seed, r.model, r.license, r.commercial_use) == (7, "test-image", "Apache-2.0", True)
    assert (r.cost_usd, r.cost_estimated) == (None, False)

    span = sink.spans[0]
    attrs = span["attributes"]
    assert (span["name"], span["span_id"], span["status"]["code"]) == ("hone.models.image", r.span_id, "ok")
    assert (attrs["gen_ai.operation.name"], attrs["gen_ai.provider.name"]) == ("image", "comfyui")
    assert attrs["gen_ai.request.seed"] == 7
    assert attrs["hone.models.media.prompt"] == "a lighthouse"
    assert attrs["hone.models.media.inputs"]["references"] == [
        {"path": str(ref), "sha256": sha256(ref), "bytes": ref.stat().st_size}
    ]
    assert attrs["hone.models.media.outputs"] == [out.record()]
    assert attrs["hone.models.media.workflow_sha256"] == sha256(COMFYUI_FIXTURES / "image.json")
    assert attrs["hone.models.media.job_id"] == r.job_id
    assert (attrs["hone.models.media.freed"], attrs["hone.models.media.session"]) == (True, False)
    assert lease.calls == [("test-image", 6.5), ("test-image", 6.5)]


def test_ac24_song_with_duration_on_two_nodes_in_a_session(tmp_path: Path, lease: LeaseRecorder) -> None:
    sink = mk.records.MemorySink()
    song = mk.music("test-music", registry=registry(), sink=sink)
    with FakeComfyUI() as server:
        with song.session():
            takes = [
                song.generate(
                    "dark trap",
                    lyrics="[verse]\nline",
                    duration_s=3,
                    bpm=95,
                    seed=i,
                    out=tmp_path / f"take_{i}",
                )
                for i in range(2)
            ]
            assert server.frees == 0  # the model stays loaded for the session
        assert server.frees == 1  # freed once at the end
    filled = server.submitted[0]
    assert (filled["94"]["inputs"]["duration"], filled["98"]["inputs"]["seconds"]) == (3, 3)
    assert (filled["94"]["inputs"]["tags"], filled["94"]["inputs"]["bpm"]) == ("dark trap", 95)
    assert filled["109"]["inputs"]["value"] == 0
    assert [t.path for t in takes] == [tmp_path / "take_0.wav", tmp_path / "take_1.wav"]  # ComfyUI's suffix
    assert [t.files[0].duration_s for t in takes] == [3.0, 3.0]
    assert lease.calls == [("test-music", 7.0)]  # one lease for the session
    loaded = [s["attributes"]["hone.models.media.loaded"] for s in sink.spans]
    assert loaded == [True, False]  # the second call found the model loaded
    assert all(s["attributes"]["hone.models.media.session"] for s in sink.spans)
    assert {s["name"] for s in sink.spans} == {"hone.models.music"}


def test_ac24_video_seconds_to_frames(tmp_path: Path, lease: LeaseRecorder) -> None:
    sink = mk.records.MemorySink()
    start = png_file(tmp_path / "shots" / "01.png", 32, 32)
    clip = mk.video("test-video", registry=registry(), sink=sink)
    with FakeComfyUI():
        r = clip.generate(
            "slow push-in", image=start, duration_s=5, size="832x480", out=tmp_path / "clips/01.mp4"
        )
    (span,) = sink.spans
    assert span["name"] == "hone.models.video"
    filled = span["attributes"]["hone.models.media.inputs"]
    assert filled["duration_s"] == 5
    assert filled["image"]["sha256"] == sha256(start)
    assert r.path == tmp_path / "clips" / "01.mp4"
    assert r.files[0].mime == "video/mp4"
    assert lease.calls == [("test-video", 7.5)]


def test_ac24_frames_on_the_workflow(tmp_path: Path, lease: LeaseRecorder) -> None:
    start = png_file(tmp_path / "start.png")
    with FakeComfyUI() as server:
        mk.video("test-video", registry=registry()).generate(
            "x", image=start, duration_s=5, out=tmp_path / "c.mp4"
        )
    assert server.submitted[0]["50"]["inputs"]["length"] == 81  # 5 s at 16 frames per second, plus 1
    assert server.submitted[0]["52"]["inputs"]["image"] == f"hone/{sha256(start)}.png"
