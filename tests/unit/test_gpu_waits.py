"""Leases that nest, leases that can never be granted, and long waits (change 0004)."""

import logging
import os
import threading
import time
from typing import Any

import pytest
import respx

from hone_models import gpu
from hone_models._tracing import scope_attributes
from hone_models.errors import CapabilityError

URL = "http://127.0.0.1:11434"


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "POLL_S", 0.01)


def scheduler(used: int = 0, processes: Any = dict, **kw) -> gpu.GpuScheduler:
    return gpu.GpuScheduler(memory=lambda: (8192, used), processes=processes, **kw)


def test_nested_lease_in_the_same_thread_reuses_the_outer_one() -> None:
    """The concept-shorts deadlock: a 5 GB batch lease around a speech client's own 5 GB lease."""
    g = scheduler()
    with g.lease("gpu:tts", 5):
        outer_attrs = scope_attributes.get()
        with g.lease("chatterbox", 5, timeout_s=0.2):  # 5 + 5 > 8 GB: waited forever before
            assert [e["name"] for e in g.status().leases] == ["gpu:tts"]
            assert scope_attributes.get() == outer_attrs
    assert g.status().leases == []


def test_nested_lease_reserves_only_what_the_outer_one_does_not_cover() -> None:
    g = scheduler()
    with g.lease("gpu:tts", 0), g.lease("chatterbox", 5, timeout_s=0.2):
        leases = g.status().leases
        assert [(e["name"], e["vram_mb"], e.get("nested_in")) for e in leases] == [
            ("gpu:tts", 0, None),
            ("chatterbox", 5120, "gpu:tts"),
        ]
        with g.lease("inner", 5, timeout_s=0.2):  # covered by 0 + 5 GB
            assert len(g.status().leases) == 2
    assert g.status().leases == []


def test_nesting_spans_schedulers_on_the_same_ledger() -> None:
    """An app's own `GpuScheduler(unload_others=True)` around the speech client's `mk.gpu.GPU`."""
    app, library = scheduler(unload_others=True), scheduler()
    with app.lease("gpu:tts", 5), library.lease("chatterbox", 5, timeout_s=0.2):
        assert len(library.status().leases) == 1


def test_a_lease_in_another_thread_is_not_nested() -> None:
    g = scheduler()
    errors: list[BaseException] = []

    def other() -> None:
        try:
            with g.lease("b", 5, timeout_s=0.1):
                pass
        except BaseException as exc:
            errors.append(exc)

    with g.lease("a", 5):
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert [type(e) for e in errors] == [TimeoutError]


def test_memory_this_process_uses_does_not_count_against_it() -> None:
    """The OneShotStudio case: 509 MB used, 160 MB of it this process's own CUDA context."""
    own = {os.getpid(): 160, 99999: 349}
    counted = gpu.GpuScheduler(memory=lambda: (8188, 509), processes=dict)
    with pytest.raises(TimeoutError, match="7679 MB free"), counted.lease("ollama", 7.5, timeout_s=0.1):
        pass  # counted as taken, it is 1 MB short
    g = gpu.GpuScheduler(memory=lambda: (8188, 509), processes=lambda: own)
    with g.lease("ollama", 7.5, timeout_s=0.1):
        assert g.status().leases[0]["vram_mb"] == 7680


def test_a_wait_no_one_can_end_fails_with_what_to_do() -> None:
    g = scheduler(used=6000, processes=lambda: {4242: 5800}, stall_s=0.05)
    with respx.mock(base_url=URL) as mock:
        mock.get("/api/ps").respond(json={"models": [{"name": "qwen2.5vl:7b"}]})
        with pytest.raises(CapabilityError) as caught, g.lease("ollama", 4):
            pass
    message = str(caught.value)
    assert "needs 4096 MB but only 2192 MB has been free" in message
    assert "no other lease held" in message
    assert "pid 4242 (5800 MB)" in message
    assert "Ollama still holds ['qwen2.5vl:7b']" in message
    assert "unload_others=True" in message
    assert g.status().waiting == []


def test_stall_message_without_process_or_ollama_information(no_backoff) -> None:
    g = scheduler(used=6000, processes=lambda: None, stall_s=0.05, unload_others=True)
    with respx.mock(base_url=URL) as mock:
        mock.get("/api/ps").respond(status_code=500)
        with pytest.raises(CapabilityError, match=r"held by other programs$"), g.lease("x", 4):
            pass


def hold(g: gpu.GpuScheduler, release: threading.Event) -> threading.Thread:
    """A thread holding a 6 GB lease until `release` is set; returns once the lease is held."""
    held = threading.Event()

    def holder() -> None:
        with g.lease("holder", 6):
            held.set()
            release.wait()

    t = threading.Thread(target=holder)
    t.start()
    held.wait()
    return t


def test_stall_does_not_count_while_another_lease_is_held() -> None:
    g = scheduler(stall_s=0.05)
    release = threading.Event()
    t = hold(g, release)
    threading.Timer(0.3, release.set).start()
    try:
        with g.lease("waiter", 6, timeout_s=10):  # waits past stall_s for a real holder, then is granted
            pass
    finally:
        release.set()
        t.join()


def test_a_long_wait_is_logged_and_listed(monkeypatch, caplog) -> None:
    monkeypatch.setattr(gpu, "WARN_AFTER_S", 0.0)
    g = scheduler()
    seen: list[list[dict]] = []
    release = threading.Event()
    t = hold(g, release)

    def look() -> None:  # once the waiter is listed, note the list and let the holder go
        while not g.status().waiting:
            time.sleep(0.01)
        seen.append(g.status().waiting)
        release.set()

    threading.Thread(target=look).start()
    with caplog.at_level(logging.WARNING, "hone_models"), g.lease("waiter", 6, timeout_s=10):
        pass
    t.join()
    assert [(w["name"], w["vram_mb"], w["pid"]) for w in seen[0]] == [("waiter", 6144, os.getpid())]
    assert g.status().waiting == []
    assert "GPU lease 'waiter' (6144 MB) has waited" in caplog.text
    assert "holder (6144 MB" in caplog.text


def test_a_nested_lease_that_cannot_fit_names_the_outer_one() -> None:
    """Change 0014 fallback: borrowing is not enough and nothing else holds a lease."""
    g = scheduler(used=6000, stall_s=0.05, unload_others=True)
    with respx.mock(base_url=URL) as mock:
        mock.get("/api/ps").respond(json={"models": []})
        with g.lease("gpu:tts", 1), pytest.raises(CapabilityError) as caught, g.lease("chatterbox", 5):
            pass
    assert "needs 4096 MB" in str(caught.value)
    assert "on top of what this process's lease 'gpu:tts' around it reserved" in str(caught.value)
