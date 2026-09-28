"""AC-19 [real]: GPU lease with NVML - status reports real memory; the lease/unload cycle works."""

import json
import time
import urllib.request

import pytest

import hone_models as mk

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def ollama_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from conftest import OLLAMA_URL  # noqa: PLC0415 - the shared test setting

    monkeypatch.setenv("OLLAMA_HOST", OLLAMA_URL)


def running_models() -> list[str]:
    from conftest import OLLAMA_URL  # noqa: PLC0415

    with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=5) as r:  # noqa: S310
        return [m["name"] for m in json.load(r)["models"]]


def memory() -> tuple[int, int, int]:
    """(total, used, free) MB from `mk.gpu.status()`, or skip without GPU information."""
    s = mk.gpu.status()
    if s.total_mb is None or s.used_mb is None or s.free_mb is None:
        pytest.skip("no NVML / nvidia-smi on this machine")
    return s.total_mb, s.used_mb, s.free_mb


def test_ac19_status_reports_real_memory(gpu_lock) -> None:
    total, used, free = memory()
    assert total > 1000
    assert 0 <= used <= total
    assert free == total - used  # no leases held yet


@pytest.mark.ollama
def test_ac19_lease_unloads_the_model_this_process_loaded(ollama_model) -> None:
    name = ollama_model("HONE_TEST_THINKING_MODEL", "deepseek-r1:8b")
    memory()
    llm = mk.text(f"ollama:{name}", sink=mk.records.NullSink())
    llm.complete([{"role": "user", "content": "Say OK."}], max_tokens=20)  # loads the model
    assert name in running_models()

    total, _, free = memory()
    need_mb = min(total - 256, free + 2048)
    if need_mb <= free:
        pytest.skip(f"{name} is too small to force an unload on this GPU")
    start = time.monotonic()
    with mk.gpu.lease("ac19", need_mb / 1024, timeout_s=120):  # only fits once the model is unloaded
        assert [e["name"] for e in mk.gpu.status().leases] == ["ac19"]
        assert name not in running_models()
    assert time.monotonic() - start < 120
    assert mk.gpu.status().leases == []
