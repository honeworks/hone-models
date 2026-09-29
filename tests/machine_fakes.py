"""Fakes for `mk.machine` tests (change 0016): a stateful Ollama `/api/ps`, a ComfyUI server on respx, a
registry with Ollama and ComfyUI entries, a lock held by another process and another process's lease."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import respx

import hone_models as mk
from hone_models.registry import Capabilities, ModelConfig, Registry

GIB = 1024**3
COMFY = "http://127.0.0.1:8188"
LOCK_HOLDER = """
import fcntl, sys, time
from pathlib import Path
with open(sys.argv[1], "a") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    Path(sys.argv[2]).write_text("held")
    while not Path(sys.argv[3]).exists():
        time.sleep(0.05)
"""
SLEEPER = "import sys, time\nfrom pathlib import Path\nwhile not Path(sys.argv[1]).exists(): time.sleep(0.05)"


def registry(*, comfyui: bool = True) -> Registry:
    """The packaged registry plus Ollama models `a` (4 GB) and `b` (`b:7b`, 5 GB) and, with `comfyui`,
    the ComfyUI entries `z-image-turbo` (7 GB) and `wan-video` (unknown size) on `COMFY`."""
    models = dict(mk.registry.load().models)
    models["a"] = ModelConfig(id="a", provider="ollama", capabilities=Capabilities(vram_gb=4.0))
    models["b"] = ModelConfig(id="b", provider="ollama", model="b:7b", capabilities=Capabilities(vram_gb=5.0))
    if comfyui:  # `comfyui` entries arrive with change 0015; built without validation until then
        for model_id, vram in (("z-image-turbo", 7.0), ("wan-video", None)):
            caps = Capabilities(vram_gb=vram)
            models[model_id] = ModelConfig.model_construct(
                id=model_id,
                provider="comfyui",
                kind="image",
                base_url=COMFY,
                capabilities=caps,
                workflow=model_id,  # runnable: the machine reads servers of entries with a workflow
            )
    return Registry(models)


class OllamaState:
    """A FakeOllama responder with the models it holds: `/api/ps` lists them, an empty `/api/generate`
    loads one (partly on the CPU when it is in `offload`), `keep_alive: 0` unloads it."""

    def __init__(
        self, models: dict[str, float] | None = None, offload: dict[str, float] | None = None
    ) -> None:
        self.models = dict(models or {})  # name: size GB
        self.offload = dict(offload or {})  # name: GB left on the CPU

    def __call__(self, path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        if path == "/api/ps":
            rows = [
                {"name": n, "size": int(gb * GIB), "size_vram": int((gb - self.offload.get(n, 0)) * GIB)}
                for n, gb in self.models.items()
            ]
            return {"models": rows}
        if path == "/api/generate":
            name = body["model"] if ":" in body["model"] else f"{body['model']}:latest"
            if body.get("keep_alive") == 0:
                self.models.pop(name, None)
            else:
                self.models[name] = 9.1
        return None


@contextmanager
def fake_comfyui(
    holding_gb: float = 6.8, newest_job: str | None = "job-1", *, down: bool = False
) -> Iterator[respx.MockRouter]:
    """ComfyUI at `COMFY` on respx: `/system_stats` reports `holding_gb` of torch memory until `/free`,
    `/history` has `newest_job` as its newest job; everything else passes through (FakeOllama)."""
    state = {"torch": int(holding_gb * GIB)}

    def stats(request: Any) -> respx.MockResponse:
        return respx.MockResponse(json={"devices": [{"name": "cuda:0", "torch_vram_total": state["torch"]}]})

    def free(request: Any) -> respx.MockResponse:
        state["torch"] = 0
        return respx.MockResponse(200)

    with respx.mock(assert_all_called=False) as mock:
        if down:
            import httpx  # noqa: PLC0415

            mock.route(host="127.0.0.1", port=8188).mock(side_effect=httpx.ConnectError("refused"))
        else:
            mock.get(f"{COMFY}/system_stats").mock(side_effect=stats)
            history = {newest_job: {"status": {"completed": True}}} if newest_job else {}
            mock.get(f"{COMFY}/history").respond(json=history)
            mock.post(f"{COMFY}/free", name="free").mock(side_effect=free)
        mock.route().pass_through()
        yield mock


def comfyui_ran(model_id: str, job_id: str, url: str = COMFY) -> None:
    """Record a ComfyUI job the way the comfyui provider does (`${HONE_HOME}/models/comfyui-loaded.json`)."""
    path = Path(os.environ["HONE_HOME"]) / "models" / "comfyui-loaded.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    entry = {"model_id": model_id, "job_id": job_id, "pid": os.getpid(), "time": "2026-09-29T10:00:00.000Z"}
    data.setdefault(url, []).append(entry)
    path.write_text(json.dumps(data))


def comfyui_listed(url: str = COMFY) -> list[str]:
    path = Path(os.environ["HONE_HOME"]) / "models" / "comfyui-loaded.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    return [e["model_id"] for e in data.get(url, [])]


def _wait_for(path: Path, proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 30
    while not path.exists():
        assert proc.poll() is None, "helper process died"
        assert time.monotonic() < deadline, "helper process never got ready"
        time.sleep(0.05)


@contextmanager
def lock_held_elsewhere(
    lock: Path, holder: str | None = "hone-flow 91822 2026-09-29T09:58:12+02:00"
) -> Iterator[int]:
    """Another process holds `lock` with `flock` (and a `.holder` file, as gpu-lock.sh writes)."""
    ready, release = lock.with_name("ready"), lock.with_name("release")
    proc = subprocess.Popen([sys.executable, "-c", LOCK_HOLDER, str(lock), str(ready), str(release)])
    try:
        _wait_for(ready, proc)
        if holder:
            Path(f"{lock}.holder").write_text(holder + "\n")
        yield proc.pid
    finally:
        release.touch()
        proc.wait(timeout=30)
        ready.unlink(missing_ok=True)
        release.unlink(missing_ok=True)


@contextmanager
def lease_elsewhere(name: str = "gpu:tts", vram_mb: int = 5120) -> Iterator[int]:
    """A live process with an entry in the lease ledger, as a lease in another process writes it."""
    release = Path(os.environ["HONE_HOME"]) / "release-lease"
    proc = subprocess.Popen([sys.executable, "-c", SLEEPER, str(release)])
    ledger = mk.gpu.GPU.ledger_path
    ledger.parent.mkdir(parents=True, exist_ok=True)
    entry = {"id": "f" * 16, "name": name, "pid": proc.pid, "vram_mb": vram_mb, "since": "x"}
    ledger.write_text(json.dumps([entry]))
    try:
        yield proc.pid
    finally:
        release.touch()
        proc.wait(timeout=30)
        release.unlink(missing_ok=True)
