"""AC-12: two processes contend for a GPU lease (fake NVML reporting 8 GB).

The second waits until the first releases; a timeout raises `TimeoutError`; the ledger drops dead PIDs."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from hone_flow.testing.contracts import check_gpu_lease

import hone_models as mk

pytestmark = pytest.mark.e2e
FAKE_NVML = Path(__file__).parents[1] / "fixtures" / "fake_nvml"
HOLDER = """
import sys, time
from pathlib import Path
import hone_models as mk
with mk.gpu.lease("holder", vram_gb=6):
    Path(sys.argv[1]).write_text("held")
    while not Path(sys.argv[2]).exists():
        time.sleep(0.05)
"""


@pytest.fixture
def fake_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    """This process and its children see the fake `pynvml` (8 GB GPU)."""
    monkeypatch.syspath_prepend(str(FAKE_NVML))
    spec = importlib.util.spec_from_file_location("pynvml", FAKE_NVML / "pynvml.py")
    assert spec is not None
    assert spec.loader is not None
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    monkeypatch.setitem(sys.modules, "pynvml", fake)  # restored after the test
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(FAKE_NVML), os.environ.get("PYTHONPATH", "")]))


def start_holder(tmp_path: Path) -> tuple[subprocess.Popen[bytes], Path]:
    held, release = tmp_path / "held", tmp_path / "release"
    proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(held), str(release)])
    deadline = time.monotonic() + 30
    while not held.exists():
        assert proc.poll() is None, "holder process died"
        assert time.monotonic() < deadline, "holder never got its lease"
        time.sleep(0.05)
    return proc, release


def test_ac12_second_process_times_out_then_waits_for_release(fake_nvml, isolated) -> None:
    proc, release = start_holder(isolated)
    try:
        status = mk.gpu.status()
        assert (status.total_mb, status.used_mb, status.free_mb) == (8192, 0, 8192 - 6144)
        assert [(e["name"], e["pid"]) for e in status.leases] == [("holder", proc.pid)]

        start = time.monotonic()
        with (
            pytest.raises(TimeoutError, match="not granted within"),
            mk.gpu.lease("waiter", 6, timeout_s=0.6),
        ):
            pass
        assert time.monotonic() - start >= 0.6

        threading.Timer(0.3, release.touch).start()  # the holder releases while we wait
        start = time.monotonic()
        with mk.gpu.lease("waiter", 6, timeout_s=30):
            assert time.monotonic() - start >= 0.3
            assert [e["name"] for e in mk.gpu.status().leases] == ["waiter"]
        assert mk.gpu.status().leases == []
    finally:
        release.touch()
        proc.wait(timeout=30)
    assert proc.returncode == 0


def test_ac12_ledger_drops_dead_pids(fake_nvml, isolated) -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    ledger = isolated / ".hone" / "models" / "gpu-ledger.json"
    ledger.parent.mkdir(parents=True)
    stale = {"id": "0" * 16, "name": "crashed", "pid": dead.pid, "vram_mb": 8000, "since": "x"}
    ledger.write_text(json.dumps([stale]))

    assert mk.gpu.status().leases == []
    with mk.gpu.lease("after-crash", 6, timeout_s=0.1):  # granted at once: the dead entry no longer counts
        pass
    assert json.loads(ledger.read_text()) == []


def test_ac12_contract_checker_passes(fake_nvml) -> None:
    check_gpu_lease(mk.gpu.GPU)
    check_gpu_lease(mk.gpu.NullGpuLease())
