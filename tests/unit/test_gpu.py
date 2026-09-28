import os
import subprocess
import threading

import pytest
import respx
from hone_flow.testing.contracts import check_gpu_lease

import hone_models as mk
from hone_models import _gpu_memory, gpu
from hone_models._tracing import scope_attributes
from hone_models.errors import CapabilityError
from hone_models.providers import ollama

URL = "http://127.0.0.1:11434"


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "POLL_S", 0.01)


def scheduler(total: int | None = 8192, used: int = 0, **kw) -> gpu.GpuScheduler:
    memory = (lambda: None) if total is None else (lambda: (total, used))
    kw.setdefault("processes", lambda: {})
    return gpu.GpuScheduler(memory=memory, **kw)


def test_free_memory_counts_the_larger_of_used_and_reserved() -> None:
    assert gpu.free_mb(None, []) is None
    assert gpu.free_mb((8000, 1000), [{"vram_mb": 3000}]) == 5000
    assert gpu.free_mb((8000, 4000), [{"vram_mb": 3000}]) == 4000


def test_lease_reserves_then_releases(isolated) -> None:
    g = scheduler(used=500)
    with g.lease("a", 2):
        status = g.status()
        assert (status.total_mb, status.used_mb, status.free_mb) == (8192, 500, 8192 - 2048)
        assert [(e["name"], e["vram_mb"]) for e in status.leases] == [("a", 2048)]
    assert g.status().leases == []
    assert g.ledger_path == isolated / ".hone" / "models" / "gpu-ledger.json"


def test_lease_is_reentrant_per_name_and_thread() -> None:
    g = scheduler()
    with g.lease("a", 6), g.lease("a", 6), g.lease("a", 6):  # would not fit twice on 8 GB
        assert len(g.status().leases) == 1
    assert g.status().leases == []
    check_gpu_lease(g)


def test_same_name_in_another_thread_is_a_separate_lease() -> None:
    g = scheduler()
    errors: list[BaseException] = []

    def other() -> None:
        try:
            with g.lease("a", 6, timeout_s=0.2):
                pass
        except BaseException as exc:
            errors.append(exc)

    with g.lease("a", 6):
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert [type(e) for e in errors] == [TimeoutError]


def test_lease_released_when_the_block_raises() -> None:
    g = scheduler()
    with pytest.raises(ValueError, match="boom"), g.lease("a", 1):
        raise ValueError("boom")
    assert g.status().leases == []


def test_no_gpu_information_grants_at_once() -> None:
    g = scheduler(total=None)
    with g.lease("a", 100, timeout_s=0.05):
        status = g.status()
        assert (status.total_mb, status.used_mb, status.free_mb) == (None, None, None)
        assert len(status.leases) == 1


def test_lease_larger_than_the_gpu_fails_fast() -> None:
    with (
        pytest.raises(CapabilityError, match="needs 10240 MB but the GPU has 8192 MB"),
        scheduler().lease("huge", 10),
    ):
        pass


def test_spans_inside_a_lease_carry_gpu_attributes() -> None:
    sink = mk.records.MemorySink()
    reply = {"message": {"content": "hi"}, "done_reason": "stop", "model": "m"}
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/chat").respond(json=reply)
        mock.post("/api/show").respond(json={})
        llm = mk.text("ollama:m", sink=sink)
        with scheduler(used=1234).lease("llm", 1):
            llm.complete([{"role": "user", "content": "x"}])
        llm.complete([{"role": "user", "content": "x"}])
    inside, outside = (s["attributes"] for s in sink.spans)
    assert inside["hone.models.gpu.vram_before_mb"] == 1234
    assert inside["hone.models.gpu.unloaded"] == []
    assert inside["hone.models.gpu.lease_wait_ms"] >= 0
    assert not any(k.startswith("hone.models.gpu.") for k in outside)


def test_short_memory_unloads_models_this_process_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "loaded", {(URL, "big:latest")})
    used = [7000]
    with respx.mock(base_url=URL) as mock:

        def unloaded(request):
            used[0] = 1000
            return respx.MockResponse(json={"done": True})

        route = mock.post("/api/generate").mock(side_effect=unloaded)
        g = gpu.GpuScheduler(memory=lambda: (8192, used[0]), processes=dict)
        with g.lease("new", 4, timeout_s=5):
            attrs = scope_attributes.get() or {}
        assert route.call_count == 1
    assert attrs["hone.models.gpu.unloaded"] == ["big:latest"]
    assert attrs["hone.models.gpu.vram_before_mb"] == 1000
    assert ollama.loaded == set()


def test_unload_others_also_unloads_running_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "loaded", set())
    with respx.mock(base_url=URL) as mock:
        mock.get("/api/ps").respond(json={"models": [{"name": "theirs:7b"}]})
        route = mock.post("/api/generate").respond(json={})
        g = gpu.GpuScheduler(memory=lambda: (8192, 7000), unload_others=True, processes=dict)
        with pytest.raises(TimeoutError, match="1192 MB free"), g.lease("new", 4, timeout_s=0.05):
            pass
    assert [c.request.content for c in route.calls] == [b'{"model":"theirs:7b","keep_alive":0}']


def test_failed_unload_is_logged_and_the_lease_still_waits(monkeypatch, caplog) -> None:
    monkeypatch.setattr(ollama, "loaded", {(URL, "big:latest")})
    monkeypatch.setattr("hone_models._http.ATTEMPTS", 1)
    with respx.mock(base_url=URL) as mock:
        mock.post("/api/generate").respond(status_code=404, json={"error": "not found"})
        with pytest.raises(TimeoutError), scheduler(used=7000).lease("new", 4, timeout_s=0.05):
            pass
    assert "could not unload big:latest" in caplog.text


def test_nvidia_smi_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_gpu_memory.importlib, "import_module", _no_module)
    out = subprocess.CompletedProcess([], 0, stdout="8192, 1024\n")
    monkeypatch.setattr(_gpu_memory.subprocess, "run", lambda *a, **k: out)
    assert gpu.read_memory() == (8192, 1024)
    apps = subprocess.CompletedProcess([], 0, stdout="123, 500\n456, 3000\n")
    monkeypatch.setattr(_gpu_memory.subprocess, "run", lambda *a, **k: apps)
    assert gpu.read_processes() == {123: 500, 456: 3000}

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(_gpu_memory.subprocess, "run", missing)
    assert gpu.read_memory() is None
    assert gpu.read_processes() is None


def _no_module(name: str):
    raise ImportError(name)


def test_file_lock_lease_is_exclusive_and_reentrant(tmp_path) -> None:
    lock = gpu.FileLockGpuLease(tmp_path / "gpu.lock")
    check_gpu_lease(lock)
    errors: list[BaseException] = []

    def other() -> None:
        try:
            with gpu.FileLockGpuLease(tmp_path / "gpu.lock").lease("b", 1, timeout_s=0.2):
                pass
        except BaseException as exc:
            errors.append(exc)

    with lock.lease("a", 1):
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert [type(e) for e in errors] == [TimeoutError]
    with gpu.FileLockGpuLease(tmp_path / "gpu.lock").lease("b", 1, timeout_s=0.2):
        pass


def test_timeout_zero_does_not_wait_for_unloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "loaded", {(URL, "big:latest")})
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{URL}/api/generate").respond(json={})
        with pytest.raises(TimeoutError), scheduler(used=7000).lease("new", 4, timeout_s=0):
            pass
    assert route.call_count == 0


def test_malformed_ledger_entries_are_dropped() -> None:
    g = scheduler()
    g.ledger_path.parent.mkdir(parents=True)
    g.ledger_path.write_text('[{"name": "no pid"}, "junk"]')
    assert g.status().leases == []


def test_child_forked_inside_a_lease_takes_its_own() -> None:
    g = scheduler()
    with g.lease("a", 1):
        pid = os.fork()
        if pid == 0:  # child: the same name is not reentrant here, it reserves its own entry
            with g.lease("a", 1):
                code = 0 if len(g.status().leases) == 2 else 1
            os._exit(code)
        _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
