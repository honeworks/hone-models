"""AC-36 (change 0020): the packaged image models that work as shipped, guide notes per situation and pose
support, and cancelling a media call from another thread."""

import threading
import time
from pathlib import Path
from typing import Any

import pytest

import hone_models as mk
from hone_models.errors import Cancelled, ConfigError
from hone_models.providers import comfyui
from hone_models.testing import FakeComfyUI, FakeMedia
from media_fixtures import LeaseRecorder, png_file

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def lease(monkeypatch: pytest.MonkeyPatch) -> LeaseRecorder:
    recorder = LeaseRecorder()  # never the real GPU in the default suite
    monkeypatch.setattr(mk.gpu, "GPU", recorder)
    return recorder


def node_value(workflow: dict[str, Any], path: str) -> Any:
    node, field = path.split(".", 1)
    return workflow[node]["inputs"][field]


def slots(cfg: mk.registry.ModelConfig) -> list[str]:
    assert isinstance(cfg.inputs, dict)
    found = cfg.inputs.get("references", [])
    return [str(s) for s in found] if isinstance(found, list) else []


@pytest.mark.parametrize(
    "model_id", ["flux.2-klein-4b", "flux.2-klein-4b-text", "qwen-image-edit-2511", "qwen-image-2.1-turbo"]
)
def test_ac36_packaged_image_models_map_every_input(model_id: str, tmp_path: Path) -> None:
    cfg = mk.registry.load().get(model_id)
    refs = [png_file(tmp_path / f"ref{i}.png") for i in range(len(slots(cfg)))]
    for i, ref in enumerate(refs):
        ref.write_bytes(ref.read_bytes() + bytes([i]))  # each its own file, so each its own upload name
    img = mk.image(model_id, sink=mk.records.MemorySink())
    given: dict[str, Any] = {"references": refs} if refs else {}
    with FakeComfyUI() as server:
        r = img.generate("a lighthouse", size="1024x576", seed=11, steps=6, out=tmp_path / "a.png", **given)
        if refs:  # the first reference alone
            img.generate("a lighthouse", seed=12, references=refs[:1], out=tmp_path / "b.png")
    assert r.error is None
    assert r.path is not None
    assert (r.files[0].width, r.files[0].height) == (1024, 576)
    full = server.submitted[0]
    inputs = cfg.inputs
    assert isinstance(inputs, dict)
    for name, value in (("prompt", "a lighthouse"), ("seed", 11), ("steps", 6)):
        assert node_value(full, str(inputs[name])) == value
    widths = inputs["width"] if isinstance(inputs["width"], list) else [inputs["width"]]
    assert {node_value(full, str(w)) for w in widths} == {1024}
    uploaded = [node_value(full, slot) for slot in slots(cfg)]
    assert len(set(uploaded)) == len(refs)
    assert all(name.startswith("hone/") for name in uploaded)
    if not refs:
        return
    one = server.submitted[1]
    kept = [slot for slot in slots(cfg) if slot.split(".")[0] in one]
    if cfg.empty_slots == "repeat_last":  # FLUX.2 klein: the last reference fills the free slot
        assert [node_value(one, s) for s in kept] == [uploaded[0]] * len(slots(cfg))
    else:  # the others: free slots removed with their links
        assert [node_value(one, s) for s in kept] == [uploaded[0]]


def test_ac36_flux_klein_needs_a_reference(tmp_path: Path) -> None:
    with FakeComfyUI() as server, pytest.raises(mk.errors.CapabilityError, match="at least one reference"):
        mk.image("flux.2-klein-4b").generate("a cat", out=tmp_path / "cat.png")
    assert server.requests == []


def test_ac36_guide_notes_and_pose() -> None:
    g = mk.guide("qwen-image-edit-2511")
    faces = g.notes_about("faces")
    assert faces
    assert all(n.about == "faces" and n.says and n.source for n in faces)
    text = g.as_text()
    assert "Notes:" in text
    assert faces[0].says in text
    assert text.index("  general:") < text.index("  faces:") < text.index("  sizes:")
    assert "Pose: reference" in text
    assert g.pose == "reference"
    assert mk.guide("z-image-turbo").pose == "none"
    research = mk.guide("qwen-image-2.1-turbo")
    assert research.commercial_use is False
    assert "commercial use: not allowed" in research.as_text()
    with pytest.raises(ConfigError, match="choose one of: general, faces, hands, people"):
        g.notes_about("feet")


def test_ac36_cancel_stops_a_comfyui_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comfyui, "POLL_S", 30.0)  # a sleep would wait 30 s: only the event ends it in time
    sink = mk.records.MemorySink()
    img = mk.image("z-image-turbo", sink=sink)
    stop = threading.Event()
    with FakeComfyUI() as server:
        server.queue("hang")
        timer = threading.Timer(0.3, stop.set)
        timer.start()
        started = time.monotonic()
        with pytest.raises(Cancelled, match="cancelled"):
            img.generate("a lighthouse", size="512x512", cancel=stop, out=tmp_path / "x.png")
        waited = time.monotonic() - started
        cancels = [q["path"] for q in server.requests if q["path"].endswith("/cancel")]
        assert server.frees == 1
    assert waited < 1.0  # the set event ends the wait at once
    (span,) = sink.spans
    assert cancels == [f"/api/jobs/{span['attributes']['hone.models.media.job_id']}/cancel"]
    assert span["status"]["code"] == "error"
    assert span["attributes"]["hone.models.media.error_kind"] == "cancelled"
    assert [e["name"] for e in span["events"]] == ["cancelled"]


def test_ac36_a_late_cancel_keeps_the_result(tmp_path: Path) -> None:
    fake = FakeMedia.like("z-image-turbo")
    stop = threading.Event()
    fake.on_run = lambda job: stop.set()  # set while the job runs, after the provider's last check
    r = fake.generate("a lighthouse", size="512x512", cancel=stop, out=tmp_path / "a.png")
    assert r.error is None
    assert r.path == tmp_path / "a.png"
    with pytest.raises(Cancelled):  # already set: nothing runs
        fake.generate("a lighthouse", size="512x512", cancel=stop, out=tmp_path / "b.png")
    assert len(fake.calls) == 1
    late, refused = fake.sink.spans  # type: ignore[attr-defined]
    assert late["attributes"]["hone.models.media.cancel_late"] is True
    assert late["status"]["code"] == "ok"
    assert refused["attributes"]["hone.models.media.error_kind"] == "cancelled"


@pytest.mark.parametrize("kind", ["scheduler", "file_lock"])
def test_ac36_cancel_while_waiting_for_the_gpu(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mk.gpu, "POLL_S", 30.0)  # only the event can end the wait in time
    if kind == "scheduler":
        lease: Any = mk.gpu.GpuScheduler(tmp_path / "ledger.json", memory=lambda: (8192, 0), stall_s=600)
    else:
        lease = mk.gpu.FileLockGpuLease(tmp_path / "gpu.lock")
    held, done = threading.Event(), threading.Event()

    def other_job() -> None:
        with lease.lease("someone-else", 7.5):
            held.set()
            done.wait(10)

    holder = threading.Thread(target=other_job)
    holder.start()
    held.wait(5)
    img = mk.image("z-image-turbo", sink=mk.records.MemorySink())
    img.gpu = lease
    stop = threading.Event()
    try:
        with FakeComfyUI() as server:
            threading.Timer(0.3, stop.set).start()
            started = time.monotonic()
            with pytest.raises(Cancelled, match="GPU lease"):
                img.generate("a lighthouse", size="512x512", cancel=stop, out=tmp_path / "x.png")
            waited = time.monotonic() - started
    finally:
        done.set()
        holder.join()
    assert waited < 1.0
    assert server.submitted == []


def test_ac36_qwen_edit_own_inputs_and_a_single_reference(tmp_path: Path) -> None:
    ref = png_file(tmp_path / "ref.png")
    with FakeComfyUI() as server:
        mk.image("qwen-image-edit-2511").generate(
            "the same man", negative="blurry", lora_strength=1.0, lightning=0.0, references=ref,
            out=tmp_path / "a.png",
        )  # fmt: skip
        mk.image("flux.2-klein-4b").generate("the same man", references=ref, out=tmp_path / "b.png")
    qwen, flux = server.submitted
    assert node_value(qwen, "21.prompt") == "blurry"
    assert node_value(qwen, "2.strength_model") == 1.0
    assert node_value(qwen, "7.strength_model") == 0.0
    assert node_value(qwen, "11.image").startswith("hone/")  # one Path is a list of one reference
    assert node_value(flux, "11.image") == node_value(flux, "13.image")
