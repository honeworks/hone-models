"""AC-25: generation failures - an unknown input, node_errors, an out-of-memory execution error, a job that
never ends and an interrupt while waiting."""

import _thread
import threading
from pathlib import Path

import pytest

import hone_models as mk
from hone_models.errors import CapabilityError, ConfigError, ModelTimeout
from hone_models.testing import FakeComfyUI
from media_fixtures import LeaseRecorder, png_file, registry

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def lease(monkeypatch: pytest.MonkeyPatch) -> LeaseRecorder:
    recorder = LeaseRecorder()  # never the real GPU in the default suite
    monkeypatch.setattr(mk.gpu, "GPU", recorder)
    return recorder


@pytest.fixture
def img() -> mk.MediaClient:
    return mk.image("test-image", registry=registry(), sink=mk.records.MemorySink())


def spans(client: mk.MediaClient) -> list[dict]:
    return client.sink.spans  # type: ignore[attr-defined]


def cancels(server: FakeComfyUI) -> list[str]:
    return [q["path"] for q in server.requests if q["path"].endswith("/cancel")]


def test_ac25_unknown_input_before_any_request(img: mk.MediaClient, tmp_path: Path) -> None:
    with FakeComfyUI() as server:
        with pytest.raises(ConfigError, match=r"does not take the input\(s\) \['colour'\].*'references'"):
            img.generate("a lighthouse", colour="red", out=tmp_path / "x.png")
        with pytest.raises(ConfigError, match="does not exist"):
            img.generate("a lighthouse", references=[tmp_path / "missing.png"], out=tmp_path / "x.png")
        with pytest.raises(CapabilityError, match="at most 2 references"):
            img.generate(
                "a lighthouse", references=[png_file(tmp_path / "r.png")] * 3, out=tmp_path / "x.png"
            )
    assert server.requests == []
    assert spans(img) == []


def test_ac25_node_errors_name_the_node(img: mk.MediaClient, tmp_path: Path) -> None:
    with FakeComfyUI() as server:
        server.queue("node_errors", node="6", input="text", message="Value not in list")
        with pytest.raises(ConfigError, match=r"node 6 \(CLIPTextEncode\), input 'text': Value not in list"):
            img.generate("a lighthouse", out=tmp_path / "x.png")
    assert spans(img)[0]["status"]["code"] == "error"


def test_ac25_out_of_memory_is_a_result(img: mk.MediaClient, tmp_path: Path) -> None:
    with FakeComfyUI() as server:
        server.queue("execution_error", type="torch.OutOfMemoryError", message="CUDA out of memory.")
        r = img.generate("a lighthouse", out=tmp_path / "x.png")
        assert server.frees == 1  # freed after a failed call too
    assert (r.files, r.path, r.error_kind) == ([], None, "out_of_memory")
    assert "CUDA out of memory" in (r.error or "")
    assert r.job_id is not None
    (span,) = spans(img)
    assert span["status"]["code"] == "error"
    assert span["attributes"]["hone.models.media.error_kind"] == "out_of_memory"
    assert span["attributes"]["hone.models.media.job_id"] == r.job_id
    assert not (tmp_path / "x.png").exists()


def test_ac25_job_that_never_ends_is_cancelled(img: mk.MediaClient, tmp_path: Path) -> None:
    with FakeComfyUI() as server:
        server.queue("hang")
        with pytest.raises(ModelTimeout, match="cancelled"):
            img.generate("a lighthouse", timeout_s=0.3, out=tmp_path / "x.png")
        job_id = server.submitted and spans(img)[0]["attributes"]["hone.models.media.job_id"]
        assert cancels(server) == [f"/api/jobs/{job_id}/cancel"]
        assert server.frees == 1
    (span,) = spans(img)
    assert [e["name"] for e in span["events"]] == ["cancelled"]
    assert span["status"]["code"] == "error"


def test_ac25_interrupt_while_waiting_cancels_and_reraises(img: mk.MediaClient, tmp_path: Path) -> None:
    with FakeComfyUI() as server:
        server.queue("hang")
        timer = threading.Timer(0.3, _thread.interrupt_main)
        timer.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                img.generate("a lighthouse", out=tmp_path / "x.png")
        finally:
            timer.cancel()
        assert len(cancels(server)) == 1
