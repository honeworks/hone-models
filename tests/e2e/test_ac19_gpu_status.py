"""AC-19 with a fake NVML (the real version is tests/gpu/test_ac19_real_gpu.py): status reports the
NVML memory; a lease that does not fit unloads the model this process loaded, waits, and times out."""

import sys
import threading
from pathlib import Path

import pytest

import hone_models as mk
from hone_models.testing import FakeOllama

pytestmark = pytest.mark.e2e
FAKE_NVML = Path(__file__).parents[1] / "fixtures" / "fake_nvml"


def test_ac19_status_and_lease_unload_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(FAKE_NVML))
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    import pynvml  # noqa: PLC0415 - the fake from FAKE_NVML

    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    status = mk.gpu.status()
    assert (status.total_mb, status.used_mb, status.free_mb, status.leases) == (8192, 0, 8192, [])

    with FakeOllama() as server:
        mk.text("gemma4-12b", sink=mk.records.NullSink()).complete([{"role": "user", "content": "hi"}])
        held, release = threading.Event(), threading.Event()

        def resident() -> None:  # another holder (a lease inside one in the same thread would be nested)
            with mk.gpu.lease("resident", 5, timeout_s=5):
                held.set()
                release.wait()

        holder = threading.Thread(target=resident)
        holder.start()
        held.wait()
        try:
            with pytest.raises(TimeoutError, match="3072 MB free"), mk.gpu.lease("other", 4, timeout_s=0.3):
                pass  # 5 + 4 GB do not fit on 8 GB: idle models are unloaded, then the lease waits
        finally:
            release.set()
            holder.join()
        with mk.gpu.lease("other", 4, timeout_s=0.3):  # granted once "resident" is released
            assert [e["name"] for e in mk.gpu.status().leases] == ["other"]
    unloads = [q["body"]["model"] for q in server.requests if q["path"] == "/api/generate"]
    assert unloads == ["gemma4-12b:latest"]
    assert mk.gpu.status().leases == []
