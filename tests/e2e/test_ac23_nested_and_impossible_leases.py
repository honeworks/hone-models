"""AC-23: a lease inside another one in the same thread is part of it; a lease no one can grant fails
(change 0004), with fake memory readers."""

import os

import pytest

import hone_models as mk
from hone_models.testing import FakeOllama

pytestmark = pytest.mark.e2e


def test_ac23_nested_lease_never_waits_for_the_outer_one(isolated) -> None:
    ledger = isolated / "gpu-ledger.json"
    app = mk.gpu.GpuScheduler(ledger, memory=lambda: (8192, 0), processes=dict, unload_others=True)
    library = mk.gpu.GpuScheduler(ledger, memory=lambda: (8192, 0), processes=dict)
    with app.lease("gpu:tts", 5), library.lease("chatterbox", 5, timeout_s=1):  # 5 + 5 GB on 8 GB
        assert [(e["name"], e["vram_mb"]) for e in app.status().leases] == [("gpu:tts", 5120)]
    with app.lease("gpu:tts", 1), library.lease("chatterbox", 5, timeout_s=1):
        assert [e["vram_mb"] for e in app.status().leases] == [1024, 4096]
    assert app.status().leases == []


def test_ac23_a_lease_no_one_can_grant_fails_naming_the_holders(isolated) -> None:
    gpu = mk.gpu.GpuScheduler(
        isolated / "gpu-ledger.json",
        memory=lambda: (8192, 7000),
        processes=lambda: {os.getpid(): 200, 31337: 6800},
        stall_s=0.2,
    )
    with FakeOllama() as ollama:
        ollama.queue("/api/ps", {"models": [{"name": "qwen2.5vl:7b"}]})  # left loaded by another process
        with (
            pytest.raises(mk.errors.CapabilityError, match=r"pid 31337 \(6800 MB\)") as caught,
            gpu.lease("big", 4),
        ):
            pass
    assert "Ollama still holds ['qwen2.5vl:7b']" in str(caught.value)
    assert gpu.status().waiting == []
