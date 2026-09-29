"""AC-33: `mk.machine.prepare(needed)` unloads what is not needed, frees ComfyUI only when it holds
something not needed, stands back while another process uses the GPU (unless `if_busy="unload"`), and
reports failed unloads as errors (FakeOllama, ComfyUI on respx, a lease and a lock held by children)."""

from pathlib import Path

import pytest

import hone_models as mk
from hone_models.testing import FakeOllama
from machine_fakes import (
    OllamaState,
    comfyui_listed,
    comfyui_ran,
    fake_comfyui,
    lease_elsewhere,
    lock_held_elsewhere,
    registry,
)

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def no_real_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lease ledger's status reads no real GPU memory."""
    monkeypatch.setattr(mk.gpu.GPU, "memory", lambda: None)


@pytest.fixture
def lock(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = isolated / "gpu.lock"
    path.touch()
    monkeypatch.setenv("HONE_GPU_LOCK", str(path))
    monkeypatch.delenv("HONE_GPU_LOCK_HELD", raising=False)
    return path


@pytest.fixture
def machine(lock: Path) -> tuple[mk.machine.Machine, mk.records.MemorySink]:
    sink = mk.records.MemorySink()
    return mk.machine.Machine(registry=registry(), sink=sink, gpus=lambda: None), sink


def unloads(server: FakeOllama) -> list[str]:
    return [r["body"]["model"] for r in server.requests if r["body"].get("keep_alive") == 0]


def test_ac33_unloads_what_is_not_needed_and_frees_comfyui(machine) -> None:
    probe, sink = machine
    comfyui_ran("wan-video", "job-1")
    with FakeOllama(responder=OllamaState({"a:latest": 4.0, "b:7b": 5.0})) as server, fake_comfyui() as comfy:
        result = probe.prepare(["a"])
        assert unloads(server) == ["b:7b"]
        assert comfy.routes["free"].call_count == 1  # POST /free
    assert result["needed"] == ["a"]
    assert result["blocked_by"] is None
    assert result["unloaded"] == [{"server": "ollama", "name": "b:7b"}]
    assert result["released"] == ["comfyui"]
    assert result["if_busy"] == "block"
    assert result["errors"] == []
    assert result["missing"] == []
    assert [m["name"] for m in result["loaded_models"]] == ["a:latest"]
    assert result["need_gb"] == 4.0
    assert comfyui_listed() == []  # the loaded-models file is cleared after /free
    (span,) = sink.spans
    assert span["name"] == "hone.models.machine.prepare"
    assert span["kind"] == "internal"
    assert span["status"]["code"] == "ok"
    attrs = span["attributes"]
    assert attrs["hone.models.machine.needed"] == ["a"]
    assert attrs["hone.models.machine.if_busy"] == "block"
    assert attrs["hone.models.machine.unloaded"] == [{"server": "ollama", "name": "b:7b"}]
    assert attrs["hone.models.machine.released"] == ["comfyui"]
    assert attrs["hone.models.machine.blocked_by"] is None
    assert attrs["hone.models.machine.errors"] == []
    assert attrs["hone.models.machine.missing"] == []


def test_ac33_keeps_comfyui_when_everything_it_holds_is_needed(machine) -> None:
    probe, _ = machine
    comfyui_ran("z-image-turbo", "job-1")
    with FakeOllama(responder=OllamaState({"a:latest": 4.0})), fake_comfyui() as comfy:
        result = probe.prepare(["a", "z-image-turbo"])
        assert comfy.routes["free"].call_count == 0  # no /free
    assert result["released"] == []
    assert result["unloaded"] == []
    assert [m["model_id"] for m in result["loaded_models"]] == ["a", "z-image-turbo"]
    assert result["missing"] == []
    assert result["need_gb"] == 11.0  # 4 GB + ComfyUI's 7 GB
    assert comfyui_listed() == ["z-image-turbo"]


def test_ac33_another_process_lease_blocks_unless_asked_to_unload(machine) -> None:
    probe, sink = machine
    state = OllamaState({"a:latest": 4.0, "b:7b": 5.0})
    with FakeOllama(responder=state) as server, lease_elsewhere("gpu:tts", 5120) as pid:
        blocked = probe.prepare(["a"])
        assert unloads(server) == []  # no request sent
        forced = probe.prepare(["a"], if_busy="unload")
        assert unloads(server) == ["b:7b"]
    assert blocked["blocked_by"] == [f"lease 'gpu:tts' (pid {pid}, 5.0 GB)"]
    assert blocked["unloaded"] == []
    assert "b:7b" in [m["name"] for m in blocked["loaded_models"]]
    assert forced["blocked_by"] == blocked["blocked_by"]  # still reported
    assert forced["unloaded"] == [{"server": "ollama", "name": "b:7b"}]
    assert forced["if_busy"] == "unload"
    assert sink.spans[0]["attributes"]["hone.models.machine.blocked_by"] == blocked["blocked_by"]


def test_ac33_the_lock_held_by_another_process_blocks(machine, lock: Path) -> None:
    probe, _ = machine
    with (
        FakeOllama(responder=OllamaState({"a:latest": 4.0, "b:7b": 5.0})) as server,
        lock_held_elsewhere(lock),
    ):
        result = probe.prepare(["a"])
        assert unloads(server) == []
    assert result["blocked_by"] == [f"gpu lock {lock} held by hone-flow 91822 2026-09-29T09:58:12+02:00"]


def test_ac33_our_own_lock_and_leases_do_not_block(
    machine, lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe, _ = machine
    monkeypatch.setenv("HONE_GPU_LOCK_HELD", "1")
    with (
        FakeOllama(responder=OllamaState({"b:7b": 5.0})) as server,
        lock_held_elsewhere(lock),
        mk.gpu.lease("mine", 1),
    ):
        result = probe.prepare(["a"])
    assert result["blocked_by"] is None
    assert unloads(server) == ["b:7b"]
    assert result["missing"] == ["a"]  # needed but not loaded: normal before the first call


def test_ac33_a_failed_unload_is_an_error(machine, no_backoff) -> None:
    probe, sink = machine
    with FakeOllama(responder=OllamaState({"a:latest": 4.0, "b:7b": 5.0})) as server:
        server.queue("/api/generate", 500, 500, 500)
        result = probe.prepare(["a"])
    assert result["unloaded"] == []
    ((error),) = result["errors"]
    assert (error["server"], error["name"]) == ("ollama", "b:7b")
    assert "500" in error["error"]
    assert "b:7b" in [m["name"] for m in result["loaded_models"]]
    assert sink.spans[0]["status"]["code"] == "error"


def test_ac33_a_server_that_does_not_answer_is_an_error(machine) -> None:
    probe, _ = machine
    with FakeOllama() as server:
        server.queue("/api/ps", 503, 503)
        result = probe.prepare([])
    assert [(e["server"], e["name"]) for e in result["errors"]] == [("ollama", None)]
    assert result["need_gb"] == 0.0


def test_ac33_unknown_id_is_a_config_error(machine) -> None:
    probe, _ = machine
    with pytest.raises(mk.errors.ConfigError, match="unknown model"):
        probe.prepare(["no-such-model"])
    with pytest.raises(mk.errors.ConfigError, match="if_busy"):
        probe.prepare([], if_busy="wait")
