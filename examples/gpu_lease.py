"""GPU leases: reserve GPU memory for any GPU work so processes on one machine take turns.

What: reserve memory around GPU work, read the GPU status, time out on memory others hold, nest a lease
    inside another, see idle Ollama models unloaded to make room, and let another GPU server go.

How: `with mk.gpu.lease(name, vram_gb=..., timeout_s=...):` reserves memory in a ledger shared by every
    process on the machine and releases it when the block ends. When memory is short, Ollama models this
    process loaded are unloaded, then the lease waits until `timeout_s` (`TimeoutError`).
    `mk.gpu.status()` shows total / used / free MB and the leases held. A lease taken inside another one in
    the same thread (a library's own lease inside an app's batch lease) is part of it and reserves only
    what the outer lease does not cover. `mk.gpu.on_short(name, release)` registers a hook the lease calls
    when memory is short, for GPU users it cannot unload itself (ComfyUI, torch). Calls made inside a lease
    carry `hone.models.gpu.*` attributes.

Why: two programs loading models at once run out of VRAM. Leases cover non-LLM GPU work too (Whisper,
    ComfyUI, torch). Pitfall: without GPU information (no NVML or `nvidia-smi`) leases are granted at once.

Runs offline: instead of the machine-wide `mk.gpu.lease` / `mk.gpu.status`, this uses its own
`mk.gpu.GpuScheduler` with a temporary ledger and a pretend 8 GB GPU.
"""

import tempfile
import threading
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)


def pretend_gpu() -> tuple[int, int]:
    """(total MB, used MB) of an 8 GB GPU: 1 GB used, 7.5 GB while an Ollama model is loaded."""
    paths = [req["path"] for req in server.requests]
    loaded = "/api/chat" in paths and "/api/generate" not in paths
    return 8192, 7500 if loaded else 1024


print("this machine:", mk.gpu.status())  # the real, machine-wide scheduler: safe to read without a GPU

gpu = mk.gpu.GpuScheduler(Path(tempfile.mkdtemp()) / "gpu-ledger.json", memory=pretend_gpu)
print("idle:", gpu.status())


def other_job(errors: list[Exception]) -> None:
    """Another job (here a thread; usually another process) that needs 6 GB."""
    try:
        with gpu.lease("sdxl", vram_gb=6, timeout_s=1):
            pass
    except TimeoutError as exc:
        errors.append(exc)


# 1. Hold a lease around GPU work. Another job that needs more than is left waits, then gives up.
with gpu.lease("whisper-turbo", vram_gb=4):
    status = gpu.status()
    print("held:", [(held["name"], held["vram_mb"]) for held in status.leases], "| free MB:", status.free_mb)
    assert [held["vram_mb"] for held in status.leases] == [4096]
    assert status.free_mb == 8192 - 4096  # total - max(used, reserved)
    errors: list[Exception] = []
    job = threading.Thread(target=other_job, args=(errors,))
    job.start()
    job.join()
    print("waited:", errors[0])
    assert isinstance(errors[0], TimeoutError)  # 6 GB cannot fit next to the 4 GB lease

    # A lease inside this one, in the same thread, is part of it: 4 GB are covered, 1 GB more is reserved.
    with gpu.lease("whisper-large", vram_gb=5, timeout_s=1):
        print("nested:", [(held["name"], held["vram_mb"]) for held in gpu.status().leases])
        assert [held["vram_mb"] for held in gpu.status().leases] == [4096, 1024]
assert gpu.status().leases == []  # released on exit

# 2. A model this process loaded fills the GPU: it is unloaded, then the lease is granted.
sink = mk.records.MemorySink()
mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": "warm up"}])  # "loads" gemma4-12b
with gpu.lease("image-model", vram_gb=5, timeout_s=5):
    mk.text("qwen2.5vl-7b", sink=sink).complete([{"role": "user", "content": "inside the lease"}])
attrs = sink.spans[-1]["attributes"]
print("unloaded:", attrs["hone.models.gpu.unloaded"], "| waited ms:", attrs["hone.models.gpu.lease_wait_ms"])
assert attrs["hone.models.gpu.unloaded"] == ["gemma4-12b:latest"]

# 3. Other GPU servers: register a release hook the lease calls when memory is short. For a real ComfyUI
#    server that is `mk.gpu.on_short("comfyui", mk.gpu.comfyui_free("http://127.0.0.1:8188"))`.
comfyui = {"holds_mb": 6000}  # a pretend image server keeping its models on the GPU


def free_comfyui() -> None:
    comfyui["holds_mb"] = 0


mk.gpu.on_short("comfyui", free_comfyui)
busy = mk.gpu.GpuScheduler(
    Path(tempfile.mkdtemp()) / "gpu-ledger.json", memory=lambda: (8192, comfyui["holds_mb"])
)
with busy.lease("llm", vram_gb=5, timeout_s=5):
    mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": "after ComfyUI let go"}])
print("released:", sink.spans[-1]["attributes"]["hone.models.gpu.released"])
assert sink.spans[-1]["attributes"]["hone.models.gpu.released"] == ["comfyui"]
mk.gpu.on_short("comfyui", None)  # hooks are per process; remove one that no longer applies

server.stop()
