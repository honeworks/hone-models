"""AC-32 / AC-33 / AC-34 [real]: `mk.machine` on the real machine, under scripts/gpu-lock.sh.

`snapshot()` reports the real card; a small model loaded with `load()` shows up in `loaded_models`, and
`prepare([])` leaves nothing loaded in Ollama afterwards."""

import os

import pytest

import hone_models as mk

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def ollama_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from conftest import OLLAMA_URL  # noqa: PLC0415 - the shared test setting

    monkeypatch.setenv("OLLAMA_HOST", OLLAMA_URL)
    monkeypatch.delenv("HONE_COMFYUI_URL", raising=False)  # this test frees nothing but Ollama


def test_ac32_snapshot_reports_the_real_card(gpu_lock) -> None:
    snap = mk.machine.snapshot()
    if snap["gpus"] is None:
        pytest.skip("no NVML / nvidia-smi on this machine")
    gpu = snap["gpus"][0]
    assert gpu["name"]
    assert gpu["memory_total_gb"] > 1
    assert 0 <= gpu["memory_used_gb"] <= gpu["memory_total_gb"]
    assert gpu["utilization_pct"] is None or 0 <= gpu["utilization_pct"] <= 100
    assert snap["gpu_lock"]["held"] is True  # the lock this run holds
    # mine: held by our own gpu-lock.sh; the conftest fixture's flock in this process does not say so
    assert snap["gpu_lock"]["mine"] is (os.environ.get("HONE_GPU_LOCK_HELD") == "1")


@pytest.mark.ollama
def test_ac33_load_then_prepare_leaves_nothing_loaded(ollama_model) -> None:
    name = ollama_model("HONE_TEST_THINKING_MODEL", "deepseek-r1:8b")  # a generate model, as in AC-19
    model_id = f"ollama:{name}"
    loaded = mk.machine.load(model_id)
    assert loaded["loaded"] is True, loaded["error"]
    assert loaded["seconds"] is not None
    names = [m["name"] for m in mk.machine.snapshot()["loaded_models"] if m["server"] == "ollama"]
    assert name in names
    result = mk.machine.prepare([])
    if result["blocked_by"]:
        pytest.skip(f"another process is using the GPU: {result['blocked_by']}")
    assert result["errors"] == []
    assert [m for m in result["loaded_models"] if m["server"] == "ollama"] == []
    assert {"server": "ollama", "name": name} in result["unloaded"]
