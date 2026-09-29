"""Machine state: see which models the machine holds, and make it hold only the ones a run needs.

What: read the GPUs, the model servers, the loaded models, the machine-wide GPU lock and the leases in one
    call; warm a model up and see that it landed partly on the CPU; unload every other model before a
    run; and see `prepare` stand back while another process holds the GPU lock.

How: `mk.machine.Machine(lock_path=..., registry=...)` (or the default `mk.machine.MACHINE`) has
    `snapshot()`, `load(model_id)` and `prepare(needed, if_busy="block")`. `snapshot()` returns plain
    JSON-able data; `None` always means unknown, never 0; sizes are GB. `load` sends Ollama an empty
    `/api/generate` so the first real call does not pay for loading. `prepare` unloads the models not in
    `needed` unless another process holds the lock or a lease (`blocked_by`; `if_busy="unload"` unloads
    anyway) and reports `unloaded`, `errors`, `missing` and the state after. It never waits: whether to
    wait is the caller's policy.

Why: a model left loaded by an earlier run takes VRAM, so the model under test is partly offloaded to the
    CPU, runs slower and times out; results measured on a busy machine are wrong. Pitfall: `prepare`
    unloads models whoever loaded them, so call it inside your own lease or `gpu-lock.sh`, never while
    another run may be using them without a lease.

Runs offline: FakeOllama stands in for Ollama, the lock is a temporary file, and the GPU reading is a
pretend 8 GB card.
"""

import tempfile
import threading
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeOllama

GIB = 1024**3
held = {"gemma4-12b:latest": 9.1, "llama3.2:1b": 2.5}  # what the pretend Ollama holds, GB


def ollama(path: str, body: dict) -> dict | None:
    """Answer `/api/ps` from `held`; gemma does not fit next to the others: 2.2 GB stay on the CPU."""
    if path == "/api/ps":
        rows = [
            {"name": n, "size": int(gb * GIB), "size_vram": int((gb - 2.2) * GIB)} for n, gb in held.items()
        ]
        return {"models": rows}
    if path == "/api/generate" and body.get("keep_alive") == 0:
        held.pop(body["model"], None)  # an unload
    return None


def pretend_gpu() -> list[dict]:
    return [{"index": 0, "name": "Pretend GPU", "memory_total_mb": 8192, "memory_used_mb": 7800,
             "utilization_pct": 12}]  # fmt: skip


server = FakeOllama(responder=ollama).start()
lock = Path(tempfile.mkdtemp()) / "gpu.lock"  # the real one is HONE_GPU_LOCK or /tmp/honeworks-gpu.lock
machine = mk.machine.Machine(lock_path=lock, gpus=pretend_gpu, processes=dict, sink=mk.records.MemorySink())

# 1. What does the machine hold?
snap = machine.snapshot()
gpu = snap["gpus"][0]
print(
    f"{gpu['name']}: {gpu['memory_used_gb']} of {gpu['memory_total_gb']} GB used, {gpu['utilization_pct']} %"
)
for model in snap["loaded_models"]:
    print(f"  {model['server']}: {model['name']} (id {model['model_id']}) {model['vram_gb']} of "
          f"{model['size_gb']} GB on the GPU")  # fmt: skip
print("servers:", [(s["server"], s["running"]) for s in snap["servers"]], "| lock:", snap["gpu_lock"]["held"])
assert snap["servers"][0]["running"] is True  # True: answered; False: not running; None: unknown
assert [m["model_id"] for m in snap["loaded_models"]] == [
    "gemma4-12b",
    None,
]  # llama3.2:1b: not in the registry
assert snap["loaded_models"][0]["vram_gb"] < snap["loaded_models"][0]["size_gb"]  # partly on the CPU

# 2. Only the model the run needs: the leftover is unloaded, gemma is kept.
result = machine.prepare(["gemma4-12b"])
print("unloaded:", result["unloaded"], "| missing:", result["missing"], "| need GB:", result["need_gb"])
assert result["unloaded"] == [{"server": "ollama", "name": "llama3.2:1b"}]
assert result["blocked_by"] is None
assert [m["name"] for m in result["loaded_models"]] == ["gemma4-12b:latest"]

# 3. Warm-up: load before the first measured call. `vram_gb < size_gb` warns about the CPU offload now.
warm = machine.load("gemma4-12b")
print(
    f"load: loaded={warm['loaded']} in {warm['seconds']} s, {warm['vram_gb']} of {warm['size_gb']} GB on GPU"
)
assert warm["loaded"] is True
assert machine.load("kokoro-82m")["loaded"] is None  # loads in-process, inside a session: "not supported"

# 4. Another process holds the GPU lock: nothing is unloaded, and prepare says who holds it.
held["llama3.2:1b"] = 2.5
taken, release = threading.Event(), threading.Event()


def other_run() -> None:  # stands in for another program running under gpu-lock.sh
    with mk.gpu.FileLockGpuLease(lock).lease("other-run", 0):
        taken.set()
        release.wait()


threading.Thread(target=other_run).start()
taken.wait()
blocked = machine.prepare(["gemma4-12b"])
print("blocked by:", blocked["blocked_by"], "| unloaded:", blocked["unloaded"])
assert blocked["blocked_by"] is not None
assert blocked["unloaded"] == []
release.set()

server.stop()
