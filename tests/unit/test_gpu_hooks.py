"""Release hooks for GPU users other than Ollama (change 0005)."""

import sys
import types

import pytest
import respx

import hone_models as mk
from hone_models import gpu
from hone_models._tracing import scope_attributes

COMFY = "http://127.0.0.1:8188"


@pytest.fixture(autouse=True)
def no_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hone_models._gpu_room._HOOKS", {})
    monkeypatch.setattr(gpu, "POLL_S", 0.01)


def test_hooks_run_in_order_when_short_and_are_recorded(caplog) -> None:
    used = [7000]
    calls: list[str] = []

    def comfy() -> None:
        calls.append("comfyui")
        used[0] = 1000

    def broken() -> None:
        calls.append("broken")
        raise RuntimeError("server gone")

    mk.gpu.on_short("comfyui", comfy)
    mk.gpu.on_short("broken", broken)
    g = mk.gpu.GpuScheduler(memory=lambda: (8192, used[0]), processes=dict)
    with g.lease("ollama", 5, timeout_s=5):
        attrs = scope_attributes.get() or {}
    assert calls == ["comfyui", "broken"]
    assert attrs["hone.models.gpu.released"] == ["comfyui"]
    assert "GPU release hook 'broken' failed: server gone" in caplog.text


def test_hooks_do_not_run_when_memory_is_enough() -> None:
    calls: list[str] = []
    mk.gpu.on_short("x", lambda: calls.append("x"))
    with mk.gpu.GpuScheduler(memory=lambda: (8192, 0), processes=dict).lease("a", 1):
        assert scope_attributes.get()["hone.models.gpu.released"] == []  # type: ignore[index]
    assert calls == []


def test_registering_again_replaces_and_none_removes() -> None:
    calls: list[str] = []
    mk.gpu.on_short("x", lambda: calls.append("old"))
    mk.gpu.on_short("x", lambda: calls.append("new"))
    g = mk.gpu.GpuScheduler(memory=lambda: (8192, 7000), processes=dict)
    with pytest.raises(TimeoutError), g.lease("a", 4, timeout_s=0.05):
        pass
    mk.gpu.on_short("x", None)
    with pytest.raises(TimeoutError), g.lease("a", 4, timeout_s=0.05):
        pass
    assert calls == ["new"]


def test_comfyui_free_posts_to_free() -> None:
    with respx.mock(base_url=COMFY) as mock:
        route = mock.post("/free").respond(status_code=200)
        mk.gpu.comfyui_free(COMFY + "/")()
    assert route.calls.last.request.content == b'{"unload_models":true,"free_memory":true}'


def test_torch_empty_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    emptied: list[bool] = []
    cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: emptied.append(True))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    mk.gpu.torch_empty_cache()
    assert emptied == [True]
    monkeypatch.setitem(sys.modules, "torch", None)  # not installed: nothing to free
    mk.gpu.torch_empty_cache()
