"""AC-32: `mk.machine.snapshot()` reports GPUs, servers, loaded models, the GPU lock and the leases, with
`None` for everything it cannot tell (fake NVML, FakeOllama, ComfyUI on respx, a lock held by a child)."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import hone_models as mk
from hone_models.testing import FakeOllama
from machine_fakes import (
    COMFY,
    GIB,
    OllamaState,
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


FAKE_NVML = Path(__file__).parents[1] / "fixtures" / "fake_nvml"


@pytest.fixture
def lock(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private lock file instead of the machine-wide one; not held by our own gpu-lock.sh."""
    path = isolated / "gpu.lock"
    path.touch()
    monkeypatch.setenv("HONE_GPU_LOCK", str(path))
    monkeypatch.delenv("HONE_GPU_LOCK_HELD", raising=False)
    return path


@pytest.fixture
def nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake `pynvml`: 8 GB, 5.61 GB used, 97 % busy, one foreign process holding 5.2 GB."""
    spec = importlib.util.spec_from_file_location("pynvml", FAKE_NVML / "pynvml.py")
    assert spec is not None
    assert spec.loader is not None
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    process = SimpleNamespace(pid=4242, usedGpuMemory=int(5.2 * GIB))
    monkeypatch.setattr(fake, "nvmlDeviceGetName", lambda handle: b"NVIDIA GeForce RTX 4060 Laptop GPU")
    monkeypatch.setattr(
        fake, "nvmlDeviceGetMemoryInfo", lambda h: SimpleNamespace(total=8 * GIB, used=int(5.61 * GIB))
    )
    monkeypatch.setattr(
        fake, "nvmlDeviceGetUtilizationRates", lambda handle: SimpleNamespace(gpu=97, memory=40)
    )
    monkeypatch.setattr(fake, "nvmlDeviceGetComputeRunningProcesses", lambda handle: [process])
    monkeypatch.setitem(sys.modules, "pynvml", fake)


def test_ac32_snapshot_reports_gpus_models_lock_and_leases(nvml, lock: Path) -> None:
    ollama = OllamaState({"gemma4-12b:latest": 9.1}, offload={"gemma4-12b:latest": 2.2})
    machine = mk.machine.Machine(registry=registry(comfyui=False))
    with FakeOllama(responder=ollama), lease_elsewhere("gpu:tts", 5120) as lease_pid:
        snap = machine.snapshot()
    json.dumps(snap)  # plain, JSON-able data
    assert snap["time"].endswith("Z")
    (gpu,) = snap["gpus"]
    assert gpu == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "memory_total_gb": 8.0,
        "memory_used_gb": 5.61,
        "memory_free_gb": 2.39,
        "utilization_pct": 97,
        "processes": [{"pid": 4242, "memory_gb": 5.2, "mine": False}],
    }
    assert [s["server"] for s in snap["servers"]] == ["ollama"]  # a registry without comfyui entries
    assert snap["servers"][0]["running"] is True
    assert snap["servers"][0]["error"] is None
    (model,) = snap["loaded_models"]
    assert model == {"server": "ollama", "name": "gemma4-12b:latest", "model_id": "gemma4-12b",
                     "size_gb": 9.1, "vram_gb": 6.9}  # fmt: skip
    assert model["vram_gb"] < model["size_gb"]  # partly on the CPU
    assert snap["gpu_lock"] == {"path": str(lock), "held": False, "mine": False, "holder": None}
    assert snap["leases"] == [{"name": "gpu:tts", "pid": lease_pid, "vram_gb": 5.0, "mine": False}]


def test_ac32_comfyui_models_named_after_a_job_and_dropped_after_free(lock: Path) -> None:
    machine = mk.machine.Machine(registry=registry(), gpus=lambda: None)
    comfyui_ran("z-image-turbo", "job-1")
    with FakeOllama(), fake_comfyui(holding_gb=6.8, newest_job="job-1"):
        snap = machine.snapshot()
        comfy = [s for s in snap["servers"] if s["server"] == "comfyui"]
        assert comfy == [{"server": "comfyui", "url": COMFY, "running": True, "error": None, "vram_gb": 6.8}]
        assert [m for m in snap["loaded_models"] if m["server"] == "comfyui"] == [
            {"server": "comfyui", "name": "z-image-turbo", "model_id": "z-image-turbo", "size_gb": None,
             "vram_gb": None}
        ]  # fmt: skip
        httpx.post(f"{COMFY}/free", json={"unload_models": True})  # freed by someone else: no torch memory
        snap = machine.snapshot()
        assert [m for m in snap["loaded_models"] if m["server"] == "comfyui"] == []
    with FakeOllama(), fake_comfyui(holding_gb=3.0, newest_job="job-from-the-ui"):
        models = machine.snapshot()["loaded_models"]  # a job hone-models did not run: unnamed
        assert [m["model_id"] for m in models if m["server"] == "comfyui"] == ["z-image-turbo", None]


def test_ac32_comfyui_refused_is_not_running(lock: Path) -> None:
    machine = mk.machine.Machine(registry=registry(), gpus=lambda: None)
    comfyui_ran("z-image-turbo", "job-1")
    with FakeOllama(), fake_comfyui(down=True):
        snap = machine.snapshot()
    comfy = [s for s in snap["servers"] if s["server"] == "comfyui"]
    assert comfy == [{"server": "comfyui", "url": COMFY, "running": False, "error": None, "vram_gb": None}]
    assert [m for m in snap["loaded_models"] if m["server"] == "comfyui"] == []


def test_ac32_unknown_is_none_never_zero(lock: Path, monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)  # no NVML ...
    monkeypatch.setenv("PATH", str(isolated / "empty"))  # ... and no nvidia-smi
    machine = mk.machine.Machine(registry=registry(comfyui=False))
    with FakeOllama() as server:
        server.queue("/api/ps", 500)  # the server answers with an error
        snap = machine.snapshot()
    assert snap["gpus"] is None
    (ollama,) = snap["servers"]
    assert ollama["running"] is None
    assert "500" in ollama["error"]
    assert snap["loaded_models"] == []
    assert snap["leases"] == []


def test_ac32_gpu_lock_held_by_another_process_by_us_and_free(
    lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    machine = mk.machine.Machine(registry=registry(comfyui=False), gpus=lambda: None)
    with FakeOllama(), lock_held_elsewhere(lock):
        state = machine.snapshot()["gpu_lock"]
        assert state == {"path": str(lock), "held": True, "mine": False,
                         "holder": "hone-flow 91822 2026-09-29T09:58:12+02:00"}  # fmt: skip
        monkeypatch.setenv("HONE_GPU_LOCK_HELD", "1")  # our own gpu-lock.sh holds it
        assert machine.snapshot()["gpu_lock"]["mine"] is True
    monkeypatch.delenv("HONE_GPU_LOCK_HELD")
    with FakeOllama():
        state = machine.snapshot()["gpu_lock"]
    assert (state["held"], state["mine"], state["holder"]) == (
        False,
        False,
        None,
    )  # a stale holder file is ignored
    assert os.environ["HONE_GPU_LOCK"] == str(lock)
