# GPU leases and local sessions

## Leases
Processes on one machine share the GPU by leasing memory through a small ledger
(`${HONE_HOME:-.hone}/models/gpu-ledger.json`, guarded by `flock`):

```python
import hone_models as mk

with mk.gpu.lease("whisper-turbo", vram_gb=1, timeout_s=300):
    pass  # any GPU work: an LLM call, Whisper, a diffusion model ...
print(mk.gpu.status())
```

- Free memory is `total - max(used, reserved)`: *used* from NVML (`gpu` extra; fallback `nvidia-smi`),
  *reserved* from the leases held. Without GPU information leases are granted at once.
- When memory is short, Ollama models this process loaded are unloaded once, then the lease waits
  (polling every 0.5 s) until `timeout_s` (`TimeoutError`). `mk.gpu.GpuScheduler(unload_others=True)`
  also unloads models other processes loaded.
- **Release hooks** for GPU users the lease cannot unload itself: `mk.gpu.on_short(name, release)`
  registers `release()`, called once per lease when memory is short, after the Ollama unloads and before
  waiting, in registration order. `mk.gpu.comfyui_free(url)` makes a hook that asks a ComfyUI server to
  unload its models (`POST /free`); `mk.gpu.torch_empty_cache` returns torch's cached memory in this
  process. Registering a name again replaces it, `on_short(name, None)` removes it; a hook that raises is
  logged and skipped. Spans inside the lease list the hooks that ran in `hone.models.gpu.released`.

  ```python no-run
  mk.gpu.on_short("comfyui", mk.gpu.comfyui_free("http://127.0.0.1:8188"))
  mk.gpu.on_short("torch", mk.gpu.torch_empty_cache)
  ```
- Memory the leasing process itself uses (its CUDA context, torch's cache) does not count against its own
  lease; per-process usage comes from NVML or `nvidia-smi`.
- **A wait that cannot end fails.** When no other lease is held and free memory has not changed for
  `stall_s` (default 120 s), the lease raises `CapabilityError` naming the processes that hold GPU memory
  (and any Ollama models still loaded) instead of waiting forever. A wait longer than 30 s is logged as a
  warning (again every 5 minutes) and listed in `mk.gpu.status().waiting`.
- **Nested leases.** A lease taken inside another one in the same thread is part of it: it reserves only
  what the outer leases do not cover (a 5 GB lease inside a 5 GB batch lease reserves nothing more; inside
  a 0 GB one it reserves 5 GB). This holds across schedulers on the same ledger, so a library's own lease
  inside an app's batch lease never waits for the app. Leases in other threads or processes are separate.
- `mk.gpu.lease` / `mk.gpu.status` belong to the machine-wide scheduler `mk.gpu.GPU`. A separate
  `mk.gpu.GpuScheduler(ledger_path, memory=reader, processes=reader, stall_s=120)` uses its own ledger,
  reads memory from `memory()` (returning `(total MB, used MB)` or `None`) and per-process usage from
  `processes()` (`{pid: MB}` or `None`), e.g. for tests.
- A lease larger than the GPU raises `CapabilityError` at once. Entries of dead processes are dropped.
- Leases are reentrant per process, thread and name. Spans made inside carry
  `hone.models.gpu.lease_wait_ms`, `.vram_before_mb`, `.unloaded` and `.released`.
- `mk.gpu.NullGpuLease()` (never waits) and `mk.gpu.FileLockGpuLease(path)` (one user at a time) implement
  the same `GpuLease` port.

Model calls do not take a lease on their own ([design decision D-009](../design/decisions.md#d-009-no-automatic-gpu-lease-around-model-calls)): wrap them when you need scheduling.

## Sessions and unloading

```python no-run
with mk.session("ollama") as url:  # use the running server, else start `ollama serve`
    mk.text("gemma4-12b").complete([{"role": "user", "content": "hi"}])
mk.unload("gemma4-12b")  # free its memory now (keep_alive: 0)
```

A server the session started runs with a clean environment (no `VIRTUAL_ENV`, `LD_LIBRARY_PATH`,
`PYTHONPATH`, `PYTHONHOME`) and is stopped on exit; one that was already running is left alone. A remote
`OLLAMA_HOST` is never started.

`mk.session("comfyui")` does the same for the ComfyUI server at `HONE_COMFYUI_URL` (default
`http://127.0.0.1:8188`), started with the command line in `HONE_COMFYUI_START` in its own process group
and stopped on exit. `mk.unload(model_id)` for a ComfyUI entry sends `POST /free`, which frees every
model that server holds. Image, music and video calls take their GPU lease by themselves
([generation.md](generation.md#gpu-and-sessions)).

**Runnable examples:** [gpu_lease.py](../examples/gpu_lease.py), [sessions.py](../examples/sessions.py).
