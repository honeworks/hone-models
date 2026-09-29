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
- `mk.gpu.GpuScheduler(unload_others=True, if_busy="block")` unloads other processes' models only while
  no other process holds a lease or the machine-wide GPU lock; the default `if_busy="unload"` unloads
  them anyway, as before.

Model calls do not take a lease on their own ([design decision D-009](../design/decisions.md#d-009-no-automatic-gpu-lease-around-model-calls)): wrap them when you need scheduling.

## Machine state: what is loaded, and only what is needed

`mk.machine` answers "what does this machine hold right now?" and "make it hold only what this run
needs" in one call each. A model left loaded by an earlier run takes VRAM, so the model you measure is
partly offloaded to the CPU, runs slower and times out; `mk.machine` makes that visible and fixes it
without breaking another run.

```python no-run
snap = mk.machine.snapshot()  # plain, JSON-able data
result = mk.machine.prepare(["gemma4-12b"])  # unload everything else, unless another run is using the GPU
warm = mk.machine.load("gemma4-12b")  # load now, so the first measured call does not pay for it
```

**`snapshot()`** reads, never changes anything, and never raises for a missing reader or a dead server.
Sizes are GB with two decimals; `None` always means unknown, never 0.

| Key | What |
|---|---|
| `time` | when it was read (UTC) |
| `gpus` | every GPU: `index`, `name`, `memory_total_gb`, `memory_used_gb`, `memory_free_gb`, `utilization_pct`, `processes` (`[{pid, memory_gb, mine}]`, GPU 0 only); `None` when neither NVML nor `nvidia-smi` answers |
| `servers` | `{server, url, running, error}` for Ollama (`OLLAMA_HOST`) and each ComfyUI server (with `vram_gb`, its torch memory). `running` is `True` (answered), `False` (not running, so it holds nothing) or `None` (timed out or failed: unknown, `error` says why) |
| `loaded_models` | `{server, name, model_id, size_gb, vram_gb}`; `model_id` is the registry id (else `None`); `vram_gb < size_gb` means partly on the CPU. Complete only when no server has `running: None` |
| `gpu_lock` | the machine-wide lock of `scripts/gpu-lock.sh`: `{path, held, mine, holder}`; `mine` when our own `gpu-lock.sh` holds it (`HONE_GPU_LOCK_HELD=1`) |
| `leases` | the GPU leases on this machine: `{name, pid, vram_gb, mine}` |

ComfyUI servers are checked only when the registry names one (the `base_url` of its `comfyui` entries)
or `HONE_COMFYUI_URL` is set. ComfyUI cannot name the models it holds, so they are the registry ids
hone-models ran there since the last `/free`, with an entry whose `model_id` is `None` for a job or memory
it cannot name.

**`prepare(needed, *, if_busy="block")`** takes registry ids and unloads every Ollama model not needed
(whoever loaded it) and frees a ComfyUI server that holds anything not needed (ComfyUI can only free
everything). When another process holds the GPU lock or a lease, it is doing GPU work right now: with
`if_busy="block"` nothing is unloaded and `blocked_by` names the holders; `if_busy="unload"` unloads
anyway. Your own leases and your own `gpu-lock.sh` do not block. It never loads and never waits: whether
to wait is your policy. It returns `needed`, `blocked_by`, `unloaded`, `released`, `if_busy`, `errors`
(failed unloads and servers that did not answer, never raised), `missing` (needed but not loaded),
`loaded_models` (read again afterwards) and `need_gb` (the needed models' registry `vram_gb`, `None` if
one is unknown), and records a `hone.models.machine.prepare` span.

**`load(model_id)`** warms an Ollama model up (an empty `/api/generate` with the entry's
`defaults.keep_alive`, default `5m`) and returns `{model_id, loaded, seconds, size_gb, vram_gb, error}`.
For ComfyUI, command and hosted models loading means running a job, and speech models load in-process
inside a session, so `load` returns `loaded: None` with the reason. It records a
`hone.models.machine.load` span.

Call `prepare` and `load` inside your own lease or `scripts/gpu-lock.sh`. `mk.machine.Machine(lock_path=,
registry=, sink=)` makes a probe with its own settings; `mk.machine.MACHINE` is the default one, and it
has the shape of hone-select's `MachineProbe` (entry point `hone.machine_probes`).

## Sessions and unloading

```python no-run
with mk.session("ollama") as url:  # use the running server, else start `ollama serve`
    mk.text("gemma4-12b").complete([{"role": "user", "content": "hi"}])
mk.unload("gemma4-12b")  # free its memory now (keep_alive: 0)
```

A server the session started runs with a clean environment (no `VIRTUAL_ENV`, `LD_LIBRARY_PATH`,
`PYTHONPATH`, `PYTHONHOME`) and is stopped on exit; one that was already running is left alone. A remote
`OLLAMA_HOST` is never started.

**Runnable examples:** [gpu_lease.py](../examples/gpu_lease.py), [sessions.py](../examples/sessions.py),
[machine_state.py](../examples/machine_state.py).
