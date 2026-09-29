"""AC-34: `mk.machine.load(model_id)` warms an Ollama model up and says where it landed, reports "not
supported" for models only a job or a session can load; `GpuScheduler(if_busy="block")` leaves other
processes' models alone while another process holds a lease."""

from pathlib import Path

import httpx
import pytest
import respx

import hone_models as mk
from hone_models.testing import FakeOllama
from machine_fakes import OllamaState, lease_elsewhere, registry

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def no_real_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lease ledger's status reads no real GPU memory."""
    monkeypatch.setattr(mk.gpu.GPU, "memory", lambda: None)


@pytest.fixture
def machine(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[mk.machine.Machine, mk.records.MemorySink]:
    monkeypatch.setenv("HONE_GPU_LOCK", str(isolated / "gpu.lock"))
    monkeypatch.delenv("HONE_GPU_LOCK_HELD", raising=False)
    sink = mk.records.MemorySink()
    return mk.machine.Machine(registry=registry(), sink=sink, gpus=lambda: None), sink


def test_ac34_load_warms_an_ollama_model_up(machine) -> None:
    probe, sink = machine
    with FakeOllama(responder=OllamaState()) as server:
        result = probe.load("a")
    (warm,) = [r["body"] for r in server.requests if r["path"] == "/api/generate"]
    assert warm == {"model": "a", "prompt": "", "keep_alive": "5m", "stream": False}
    assert result["model_id"] == "a"
    assert result["loaded"] is True
    assert result["seconds"] is not None
    assert result["seconds"] >= 0
    assert (result["size_gb"], result["vram_gb"], result["error"]) == (9.1, 9.1, None)
    (span,) = sink.spans
    assert span["name"] == "hone.models.machine.load"
    attrs = span["attributes"]
    assert attrs["hone.models.machine.model_id"] == "a"
    assert attrs["hone.models.machine.loaded"] is True
    assert attrs["hone.models.machine.vram_gb"] == 9.1
    assert attrs["hone.models.machine.seconds"] == result["seconds"]


def test_ac34_load_reports_a_model_partly_on_the_cpu(machine) -> None:
    probe, _ = machine
    with FakeOllama(responder=OllamaState(offload={"b:7b": 2.2})):
        result = probe.load("b")
    assert result["loaded"] is True
    assert result["vram_gb"] < result["size_gb"]
    assert (result["size_gb"], result["vram_gb"]) == (9.1, 6.9)


def test_ac34_load_on_a_server_that_times_out(machine) -> None:
    probe, sink = machine
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/generate").mock(side_effect=httpx.ReadTimeout("slow"))
        result = probe.load("a")
    assert result["loaded"] is False
    assert "did not answer" in result["error"]
    assert sink.spans[0]["status"]["code"] == "error"


@pytest.mark.parametrize(
    ("model_id", "hint"), [("z-image-turbo", "run one small job"), ("kokoro-82m", "session")]
)
def test_ac34_load_is_not_supported_for_jobs_and_in_process_models(machine, model_id: str, hint: str) -> None:
    probe, sink = machine
    with respx.mock() as mock:  # no request at all: an unmatched one would fail the test
        result = probe.load(model_id)
    assert mock.calls.call_count == 0
    assert result["loaded"] is None
    assert result["error"].startswith("not supported for ")
    assert hint in result["error"]
    assert sink.spans[0]["status"]["code"] == "ok"


def test_ac34_short_lease_leaves_other_processes_models_alone(
    machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hone_models.gpu.POLL_S", 0.01)
    blocking = mk.gpu.GpuScheduler(
        memory=lambda: (8192, 7000), processes=dict, unload_others=True, if_busy="block"
    )
    unloading = mk.gpu.GpuScheduler(memory=lambda: (8192, 7000), processes=dict, unload_others=True)
    assert unloading.if_busy == "unload"  # the default keeps today's behaviour
    with FakeOllama(responder=OllamaState({"b:7b": 5.0})) as server, lease_elsewhere("gpu:tts", 1024):
        with pytest.raises(TimeoutError), blocking.lease("new", 4, timeout_s=0.2):
            pass  # waits and fails as leases do today
        assert [r for r in server.requests if r["path"] == "/api/generate"] == []
        with pytest.raises(TimeoutError), unloading.lease("new", 4, timeout_s=0.2):
            pass
        assert [r["body"]["model"] for r in server.requests if r["path"] == "/api/generate"] == ["b:7b"]
    with pytest.raises(mk.errors.ConfigError, match="if_busy"):
        mk.gpu.GpuScheduler(if_busy="wait")  # type: ignore[arg-type]  # an untyped caller
