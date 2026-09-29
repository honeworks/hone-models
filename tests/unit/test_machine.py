"""Unit tests for `mk.machine` (change 0016): the readers, the tag rule, GB rounding, "unknown is None"."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

import hone_models as mk
from hone_models import _comfyui_loaded, _gpu_memory, _gpu_room, _machine_read
from hone_models._gpu_locks import machine_lock_path, probe_lock
from hone_models.registry import ModelConfig, Registry
from hone_models.testing import FakeOllama
from machine_fakes import COMFY, GIB, OllamaState, comfyui_ran, fake_comfyui, lock_held_elsewhere, registry


@pytest.fixture(autouse=True)
def private_lock(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = isolated / "gpu.lock"
    monkeypatch.setenv("HONE_GPU_LOCK", str(path))
    monkeypatch.delenv("HONE_GPU_LOCK_HELD", raising=False)
    monkeypatch.setattr(mk.gpu.GPU, "memory", lambda: None)
    return path


def test_tag_rule_and_gb_rounding() -> None:
    assert _machine_read.tagged("gemma4-12b") == "gemma4-12b:latest"
    assert _machine_read.tagged("qwen2.5vl:7b") == "qwen2.5vl:7b"
    assert _machine_read.tagged("hf.co/org/model") == "hf.co/org/model:latest"
    assert _machine_read.tagged("registry:5000/model") == "registry:5000/model:latest"
    assert _machine_read.gb(5744) == 5.61
    assert _machine_read.gb(0) == 0.0
    assert _machine_read.gb(None) is None
    assert _machine_read.gb_of_bytes(int(6.9 * GIB)) == 6.9
    assert _machine_read.gb_of_bytes(None) is None
    assert _machine_read.gb_of_bytes("7") is None


def fake_nvml(**overrides) -> SimpleNamespace:
    class NVMLError(Exception):
        pass

    def utilization(handle):
        raise NVMLError("not supported")

    base = {
        "NVMLError": NVMLError,
        "nvmlInit": lambda: None,
        "nvmlShutdown": lambda: None,
        "nvmlDeviceGetCount": lambda: 2,
        "nvmlDeviceGetHandleByIndex": lambda i: i,
        "nvmlDeviceGetName": lambda h: b"GPU %d" % h,
        "nvmlDeviceGetMemoryInfo": lambda h: SimpleNamespace(total=8 * GIB, used=GIB * (h + 1)),
        "nvmlDeviceGetUtilizationRates": utilization,
    }
    return SimpleNamespace(**(base | overrides))


def test_read_gpus_from_nvml_every_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml())
    gpus = _gpu_memory.read_gpus()
    assert gpus == [
        {
            "index": 0,
            "name": "GPU 0",
            "memory_total_mb": 8192,
            "memory_used_mb": 1024,
            "utilization_pct": None,
        },
        {
            "index": 1,
            "name": "GPU 1",
            "memory_total_mb": 8192,
            "memory_used_mb": 2048,
            "utilization_pct": None,
        },
    ]


def test_read_gpus_from_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    out = "0, NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 5610, 97\n1, Other, 4096, 0, [N/A]\n"
    monkeypatch.setattr(
        _gpu_memory.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, out)
    )
    assert _gpu_memory.read_gpus() == [
        {"index": 0, "name": "NVIDIA GeForce RTX 4060 Laptop GPU", "memory_total_mb": 8188,
         "memory_used_mb": 5610, "utilization_pct": 97},
        {"index": 1, "name": "Other", "memory_total_mb": 4096, "memory_used_mb": 0, "utilization_pct": None},
    ]  # fmt: skip
    for bad in ("0, x, 8188\n", "0, x, lots, 0, 1\n", ""):
        done = subprocess.CompletedProcess([], 0, bad)
        monkeypatch.setattr(_gpu_memory.subprocess, "run", lambda *a, _d=done, **k: _d)
        assert _gpu_memory.read_gpus() is None, bad


def test_no_compute_apps_is_an_empty_dict_not_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(_gpu_memory.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, ""))
    assert _gpu_memory.read_processes() == {}


def test_gpu_section_without_process_readings() -> None:
    gpu = {"index": 1, "name": "x", "memory_total_mb": 1024, "memory_used_mb": None, "utilization_pct": None}
    (out,) = _machine_read.gpu_section(lambda: [gpu], lambda: None) or []
    assert (out["memory_total_gb"], out["memory_used_gb"], out["memory_free_gb"]) == (1.0, None, None)
    assert out["processes"] is None


def test_probe_lock_states(private_lock: Path, isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert machine_lock_path() == private_lock
    monkeypatch.delenv("HONE_GPU_LOCK")
    assert str(machine_lock_path()) == "/tmp/honeworks-gpu.lock"  # noqa: S108 - only compared, never opened
    missing = probe_lock(private_lock)
    assert (missing["held"], missing["mine"], missing["holder"]) == (False, False, None)
    unreadable = probe_lock(isolated)  # a directory cannot be opened as a lock file
    assert (unreadable["held"], unreadable["mine"]) == (None, None)
    private_lock.touch()
    with lock_held_elsewhere(private_lock, holder=None):
        held = probe_lock(private_lock)
    assert (held["held"], held["mine"], held["holder"]) == (True, False, None)


def test_comfyui_loaded_file_add_read_clear() -> None:
    assert _comfyui_loaded.read() == {}
    _comfyui_loaded.add(COMFY + "/", "z-image-turbo", "job-1")
    _comfyui_loaded.add(COMFY, "wan-video", "job-2")
    (first, second) = _comfyui_loaded.read()[COMFY]
    assert (first["model_id"], first["job_id"], second["job_id"]) == ("z-image-turbo", "job-1", "job-2")
    assert set(first) == {"model_id", "job_id", "pid", "time"}
    _comfyui_loaded.clear(COMFY)
    assert _comfyui_loaded.read() == {}


def test_get_json_classifies_answers() -> None:
    url = "http://127.0.0.1:9/x"
    with respx.mock() as mock:
        route = mock.get(url)
        route.side_effect = httpx.ReadTimeout("slow")
        assert _machine_read.get_json(url)[0] is None
        route.side_effect = None
        route.return_value = httpx.Response(200, text="not json")
        assert _machine_read.get_json(url)[0] is None
        route.return_value = httpx.Response(200, json=[1, 2])
        running, data, error = _machine_read.get_json(url)
        assert (running, data) == (None, {})
        assert error is not None
        assert "no JSON object" in error


def test_comfyui_urls_from_registry_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _machine_read.comfyui_urls(registry(comfyui=False)) == []
    assert _machine_read.comfyui_urls(registry()) == [COMFY]
    monkeypatch.setenv("HONE_COMFYUI_URL", "http://gpu-box:8188/")
    assert _machine_read.comfyui_urls(registry(comfyui=False)) == ["http://gpu-box:8188"]
    no_url = ModelConfig.model_construct(id="flux", provider="comfyui", kind="image")
    assert _machine_read.comfyui_urls(Registry({"flux": no_url})) == ["http://gpu-box:8188"]


def test_the_packaged_catalog_checks_the_local_comfyui() -> None:
    # D-074: the packaged ComfyUI entries have workflows, so the default server is checked everywhere
    assert _machine_read.comfyui_urls(mk.registry.load()) == [_machine_read.COMFYUI_URL]


def test_comfyui_entry_is_named_by_its_workflow_file() -> None:
    cfg = ModelConfig.model_construct(id="z", provider="comfyui", kind="image")
    object.__setattr__(cfg, "workflow", "workflows/z-image-turbo.json")  # the 0015 entry key
    assert _machine_read.comfyui_entry("z", Registry({"z": cfg}))["name"] == "z-image-turbo.json"
    assert _machine_read.comfyui_entry(None, Registry({}))["name"] is None


def test_comfyui_memory_it_cannot_name_and_unknown_totals() -> None:
    reg = registry()
    with fake_comfyui(holding_gb=2.0, newest_job=None):
        server, models = _machine_read.read_comfyui(COMFY, reg, {})
    assert server["vram_gb"] == 2.0
    assert [m["model_id"] for m in models] == [None]  # memory held, nothing hone-models ran
    assert _machine_read.torch_vram_gb({"devices": [{"name": "cpu"}]}) is None
    assert _machine_read.torch_vram_gb({"devices": ["x"]}) is None
    assert _machine_read.torch_vram_gb({}) is None


def test_comfyui_history_failure_is_unknown() -> None:
    with respx.mock() as mock:
        mock.get(f"{COMFY}/system_stats").respond(json={"devices": []})
        mock.get(f"{COMFY}/history").respond(status_code=500)
        server, models = _machine_read.read_comfyui(COMFY, registry(), {})
        assert (server["running"], models) == (None, [])
        assert "500" in server["error"]
        mock.get(f"{COMFY}/history").mock(side_effect=httpx.ConnectError("refused"))
        server, _ = _machine_read.read_comfyui(COMFY, registry(), {})
        assert server["running"] is None
        assert "refused" in server["error"]


def test_ollama_answer_without_a_model_list_is_unknown() -> None:
    with FakeOllama(responder=lambda path, body: {"models": None} if path == "/api/ps" else None):
        server, models = _machine_read.read_ollama(registry())
    assert (server["running"], models) == (None, [])
    assert "no model list" in server["error"]


def test_prepare_reports_a_failed_comfyui_free() -> None:
    comfyui_ran("wan-video", "job-1")
    machine = mk.machine.Machine(registry=registry(), sink=mk.records.NullSink(), gpus=lambda: None)
    with FakeOllama(), fake_comfyui() as comfy:
        comfy.routes["free"].side_effect = None
        comfy.routes["free"].return_value = httpx.Response(500)
        result = machine.prepare(["z-image-turbo"])
    assert result["released"] == []
    assert [(e["server"], e["name"]) for e in result["errors"]] == [("comfyui", None)]
    assert _comfyui_loaded.read()[COMFY][0]["model_id"] == "wan-video"  # not cleared: nothing was freed
    assert result["missing"] == ["z-image-turbo"]


def test_need_gb_is_none_when_a_size_is_unknown() -> None:
    machine = mk.machine.Machine(registry=registry(), sink=mk.records.NullSink(), gpus=lambda: None)
    with FakeOllama(responder=OllamaState()), fake_comfyui(holding_gb=0):
        result = machine.prepare(["a", "wan-video", "gpt-4.1-mini"])
    assert result["need_gb"] is None  # wan-video has no vram_gb; the hosted model does not count
    assert result["missing"] == ["a", "wan-video"]


def test_load_not_supported_for_hosted_models() -> None:
    result = mk.machine.Machine(registry=registry(), sink=mk.records.NullSink()).load("gpt-4.1-mini")
    assert result["loaded"] is None
    assert result["error"] == "not supported for openai_compatible: run one small job in a session instead"


def test_load_when_ps_does_not_list_the_model() -> None:
    machine = mk.machine.Machine(registry=registry(), sink=mk.records.NullSink())
    with FakeOllama():  # answers the warm-up, lists nothing
        result = machine.load("a")
    assert (result["loaded"], result["size_gb"], result["vram_gb"]) == (True, None, None)


def test_scheduler_yields_to_a_lock_held_elsewhere(
    private_lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_lock.touch()
    assert _gpu_room.busy_elsewhere([]) is False
    with lock_held_elsewhere(private_lock):
        assert _gpu_room.busy_elsewhere([]) is True
        monkeypatch.setenv("HONE_GPU_LOCK_HELD", "1")
        assert _gpu_room.busy_elsewhere([]) is False
