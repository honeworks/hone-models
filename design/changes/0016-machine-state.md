# 0016: Knowing and controlling which models the machine holds

## Status

`proposed`

## Context

Experiments in hone-select ([hone-select 0009](https://github.com/honeworks/hone-select/blob/main/design/changes/0009-experiments.md))
and pipelines in hone-flow run on one shared machine: an RTX 4060 Laptop GPU with 8 GB, 24 CPU cores and
30 GB RAM. On this machine a busy environment corrupts results:

- a model left loaded in Ollama by an earlier run (or by another program) takes VRAM, so the model under
  test is partly offloaded to the CPU; it runs slower, can hit timeouts, and the timeouts become errors
  in the results;
- a ComfyUI server keeps its models in VRAM after a render, with the same effect;
- another GPU job, or heavy CPU work, distorts speed measurements.

The owner split the responsibility. hone-select experiments own the **policy**: conditions in
`experiment.toml`, checks before the start and before each sample or model group, what to do when a
check fails (wait, stop, or only record), and a per-sample snapshot of the environment. That is a
separate change record in hone-select. hone-models owns **knowing and controlling the machine's model
state**: which models are loaded in which server, how much GPU memory is used and free, making sure only
the needed models are loaded, and the machine-wide GPU lock and leases. hone-select measures CPU load and
free RAM itself with the standard library.

hone-select's core cannot import hone-models, so hone-select defines a port, `MachineProbe`, and reaches
hone-models through its optional extra `hone-select[models]` and an entry point, the same way it loads
judges (`hone.decision_clients`, hone-select D-002).

Most of the pieces already exist in hone-models, but only as parts of the lease:

- `_gpu_memory.read_memory()` / `read_processes()`: total and used memory of GPU 0 and memory per process,
  from NVML (extra `gpu`), else `nvidia-smi`, else `None`;
- `providers/ollama.running()` (`GET /api/ps`) and `ollama.unload(url, model)` (`keep_alive: 0`);
  `mk.unload(model_id)` for one registry id;
- `mk.gpu.comfyui_free(url)` (`POST /free`, change [0005](0005-release-hooks-for-other-gpu-servers.md));
- the lease ledger (`mk.gpu.status().leases`: name, pid, MB of every live lease on the machine);
- `scripts/gpu-lock.sh`: the machine-wide lock file `/tmp/honeworks-gpu.lock` (`HONE_GPU_LOCK`), held with
  `flock` around every real-model run in every repository; it writes `<lock>.holder` ("repo pid time")
  and sets `HONE_GPU_LOCK_HELD=1` for the command it runs.

## Problem

- There is no public way to ask "what is loaded on this machine right now, and how much GPU memory is
  free?" in one call. `mk.gpu.status()` gives memory and leases but not the models in each server, not
  the GPU's utilization, and not the machine-wide lock.
- There is no public way to say "only these models should be loaded". `mk.unload` frees one model you
  name; `GpuScheduler(unload_others=True)` unloads everything, and only when a lease is short of memory.
  Neither checks whether another process is in the middle of GPU work, so a caller who wants a quiet
  machine must either leave leftovers in place or risk yanking a model another run is using.
- Nothing reports the case that matters most for experiments: a model that is loaded but partly on the
  CPU. Ollama's `/api/ps` reports it (`size_vram` smaller than `size`), but hone-models drops that field.
- The facts that exist are read in different places with different "unknown" handling; a consumer
  cannot tell "no model is loaded" from "the server did not answer".

## Options

1. **Leave it to consumers.** hone-select (and each app) calls `/api/ps`, `nvidia-smi`, ComfyUI and the
   lock file itself. Repeats code hone-models already has, in every consumer, and each copy decides
   differently what "unknown" and "someone else's model" mean.
2. **Extend `mk.gpu`** with `mk.gpu.snapshot()` and `mk.gpu.only_loaded(ids)`. Keeps one module, but
   `mk.gpu` is about leasing GPU memory and already exports 13 names; model servers (Ollama, ComfyUI) and
   the lock file are not GPU memory, and the probe object hone-select needs would be a second kind of
   thing in the same module.
3. **A new module `mk.machine`**, built on the existing readers, with one object that has exactly the
   port's shape (`snapshot()`, `prepare(needed)`) and a default instance, the same pattern as
   `mk.gpu.GPU` / `mk.gpu.lease`. Unload rules that respect the lock and other processes' leases.
4. **A daemon** that watches the machine and serves state and requests to every process. Accurate and
   race-free, but a long-running service is far more than the problem needs, and AGENTS.md rule 7 says we
   never start or stop services on the shared machine.

## Decision

Proposed: option 3. It is the smallest thing that gives hone-select (and hone-flow, and apps) one
answer, reuses the readers the lease already trusts, and keeps the policy out of hone-models: hone-models
says what is true and does what it is asked when that is safe; it never waits, retries or decides that a
run should stop.

### Public API

```python
import hone_models as mk

mk.machine.snapshot() -> dict              # what the machine holds now (shape below)
mk.machine.prepare(needed) -> dict         # make sure only `needed` models are loaded (registry ids)
mk.machine.Machine(*, comfyui_url=..., lock_path=None, registry=None, sink=None)   # the probe object
mk.machine.MACHINE                         # the default instance; snapshot / prepare are its methods
```

`Machine` is a plain class whose two methods are the `MachineProbe` port of hone-select; hone-models does
not import that port (rule 2), it has the same shape, like every other shape in design/current.md §10.
`snapshot` and `prepare` at module level are `MACHINE.snapshot` and `MACHINE.prepare`, as `mk.gpu.lease`
is `GPU.lease`. The name `prepare` is the port's; `only_loaded` was considered and dropped so that there
is one name for one operation.

Constructor arguments, all optional:

- `comfyui_url`: the ComfyUI server to check; default `HONE_COMFYUI_URL`, else `http://127.0.0.1:8188`;
  `None` means "this machine runs no ComfyUI, do not check" (and the snapshot says so, see below).
- `lock_path`: the machine-wide lock file; default `HONE_GPU_LOCK`, else `/tmp/honeworks-gpu.lock`, the
  same rule as `scripts/gpu-lock.sh`.
- `registry`, `sink`: as for the clients.
- For tests, the readers are injectable like `GpuScheduler`'s (`gpus=`, `processes=`, `scheduler=`).

The Ollama server is `ollama.base_url()` (`OLLAMA_HOST`), the same one `ollama.running()` and the lease
use. One Ollama server and one ComfyUI server per machine is what exists today; lists can come later.

### `snapshot()`

One call, no side effects, returns plain JSON-able data (so hone-select can store it per sample as is).
Sizes are GB with two decimals, as in the port; `None` always means unknown, never 0.

```python
{
  "time": "2026-09-29T10:00:00.000Z",
  "gpus": [                                   # None when neither NVML nor nvidia-smi answers
    {"index": 0, "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
     "memory_total_gb": 8.0, "memory_used_gb": 5.61, "memory_free_gb": 2.39,
     "utilization_pct": 97,                   # None when the reader cannot tell
     "processes": [{"pid": 4242, "memory_gb": 5.2, "mine": False}]}   # None when unknown
  ],
  "servers": [
    {"server": "ollama",  "url": "http://127.0.0.1:11434", "running": True,  "error": None},
    {"server": "comfyui", "url": "http://127.0.0.1:8188",  "running": False, "error": None}
  ],
  "loaded_models": [
    {"server": "ollama", "name": "gemma4-12b:latest", "model_id": "gemma4-12b",
     "size_gb": 9.1, "vram_gb": 6.9},        # vram_gb < size_gb: partly on the CPU
    {"server": "comfyui", "name": None, "model_id": None, "size_gb": None, "vram_gb": 3.4}
  ],
  "gpu_lock": {"path": "/tmp/honeworks-gpu.lock", "held": True, "mine": False,
               "holder": "hone-flow 91822 2026-09-29T09:58:12+02:00"},
  "leases": [{"name": "gpu:tts", "pid": 91830, "vram_gb": 5.0, "mine": False}]
}
```

Refinements of the shape agreed with hone-select (the port's three keys keep their meaning; the rest is
added):

- `gpus[]`: `utilization` became `utilization_pct` (0-100, NVML's unit), plus `memory_free_gb` and
  `processes` (who holds GPU memory: the "other GPU job" case). `mine` marks this process.
- `servers[]` is new. It is how a consumer tells "nothing is loaded" from "we could not find out":
  `running: True` (answered), `False` (connection refused: the server is not running, so it holds
  nothing), `None` (timed out or answered with an error: unknown, `error` says why). A server whose
  `running` is not `True` contributes no entries to `loaded_models`, and `loaded_models` is complete only
  when no server has `running: None`.
- `loaded_models[]` gains `model_id` (the registry id whose provider name matches, else `None`) so the
  caller can compare with the ids it passed to `prepare`.
- `gpu_lock` gains `path` and `mine`; `leases` is new (the ledger, as in `mk.gpu.status()`, in GB).

How each fact is read:

| Fact | Source | Unknown when |
|---|---|---|
| GPUs, memory, utilization | NVML (extra `gpu`): device count, name, `MemoryInfo`, `UtilizationRates`; else one `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu` | both fail: `gpus` is `None` (not `[]`: without a reader we cannot tell "no GPU" from "unknown") |
| GPU memory per process | `read_processes()` (existing) | it returns `None`: `processes` is `None` |
| Ollama models | `GET /api/ps`: `name`, `size`, `size_vram` (bytes) | timeout or error status: `running: None` |
| ComfyUI memory | `GET /system_stats`: `devices[].torch_vram_total` is what its torch holds; it does not list its models, so one entry with `name: None` when that is above 0 | as for Ollama; `comfyui_url=None`: the server is left out of `servers` entirely, which the snapshot shows |
| lock | open the lock file and try `flock(LOCK_EX \| LOCK_NB)`, releasing at once; `holder` from `<lock>.holder`, read only when held (the file can be stale after a killed run); `mine` from `HONE_GPU_LOCK_HELD=1` | the file cannot be opened: `held: None` |
| leases | the lease ledger (existing) | never (an empty ledger is a known empty) |

`nvidia-smi` reports every GPU; NVML is read for every device too. The lease itself still accounts
GPU 0 only (§7); that does not change. Server requests use a short timeout (2 s), so a snapshot costs
well under a second when everything answers and at most a few seconds when something hangs. Nothing is
cached: each call reads the machine now.

`snapshot()` never raises for a missing reader or a dead server; those are facts it reports. It raises
only for programming errors.

### `prepare(needed)`

`needed` is a sequence of registry ids. `prepare` makes the model servers hold only those models, if it
is safe to do so, and reports what it did and what is true afterwards. It does not load the needed
models (the first call does, and whether to warm up is hone-select's policy), and it never waits.

1. Resolve each id through the registry (`ConfigError` for an unknown id, as everywhere; a call problem
   is an exception, rule 3). For Ollama models, the provider name with `:latest` added when it has no
   tag, which is how `/api/ps` names them.
2. **Check that nobody else is using the GPU.** If the machine-wide lock is held and not `mine`, or the
   ledger has a live lease of another process, `prepare` unloads nothing and returns
   `blocked_by` naming the holder(s). The reason: a lease or the lock means some run is doing GPU work
   right now, and the only models we could unload are the ones it may be using. Unloading them would
   corrupt that run; waiting is the right answer, and whether and how long to wait is the caller's
   policy. This process's own leases and a lock held by our own `gpu-lock.sh` do not block.
3. Otherwise unload every loaded Ollama model that is not needed (`keep_alive: 0`, the existing
   `ollama.unload`), whoever loaded it: with no lease and no lock held elsewhere, a loaded model is a
   leftover, which is exactly the problem. If ComfyUI holds memory and is not needed, call its `/free`
   (the existing `comfyui_free`). A ComfyUI server is kept by putting the name `"comfyui"` in `needed`
   (it has no registry ids for its models; see the open questions).
4. Read `/api/ps` and ComfyUI again and report the state afterwards.

Returned (JSON-able):

```python
{
  "needed": ["gemma4-12b"],
  "blocked_by": None,              # or ["gpu lock held by hone-flow 91822 ...", "lease 'gpu:tts' (pid 91830, 5.0 GB)"]
  "unloaded": [{"server": "ollama", "name": "qwen2.5vl:7b"}],
  "released": ["comfyui"],         # servers asked to free their memory
  "errors": [],                    # [{"server", "name", "error"}]: an unload that failed, a server that did not answer
  "missing": ["gemma4-12b"],       # needed but not loaded (normal before the first call)
  "loaded_models": [...],          # as in snapshot(), read after the unloads
  "need_gb": 7.5                   # sum of the needed models' registry vram_gb; None if any is unknown
}
```

A failed unload or an unreachable server is an entry in `errors`, never skipped silently and never an
exception: the caller decides what an incomplete cleanup means. A server with `running: None` also
appears in `errors`, because `prepare` cannot know what it holds. `need_gb` lets the caller compare with
`memory_total_gb`: models that together do not fit will be offloaded whatever `prepare` does.

There is a race `prepare` cannot close: between its read and its unloads another process can load a
model or take a lease. The state after is read again and returned, and the next `snapshot()` shows any
change; a daemon (option 4) would be the only full fix and is not worth it.

### Relation to the lease and the lock

- `prepare` takes no lease and no lock of its own. A caller that wants the GPU for a model group holds
  a lease (hone-select gets it through `hone.gpu_leases`) or runs under `gpu-lock.sh`, and calls
  `prepare` inside it; its own lease and lock do not block it.
- The lease keeps its current behaviour (unload this process's models when short, all with
  `unload_others=True`, then the hooks). `prepare` is the explicit, lease-aware version for callers who
  want the machine clean before they start, not only when memory runs out.
- The rule "do not unload while another process holds a lease or the lock" is new and applies to
  `prepare` only. Whether `GpuScheduler(unload_others=True)` should follow it too is an open question.

### The adapter for hone-select

Two registrations, the same pair as for decision clients today:

- hone-models registers its own name in its `pyproject.toml`, as for its other shapes (§10 table):
  `[project.entry-points."hone.machine_probes"] hone_models = "hone_models.machine:Machine"`
  (a factory; called with no arguments it gives the default probe).
- hone-select registers `"hone_models:machine" = "hone_select.adapters.hone_models:machine"` (its D-002
  rule: names are exact entry-point names), a function that imports `hone_models` only when called and
  returns `hone_models.machine.Machine(**options)`; without the extra it raises hone-select's
  `ConfigError` saying to install `hone-select[models]`. That code lives in hone-select and is part of
  its change record; this record only promises the shape it relies on.

design/current.md §10 gains a row: `mk.machine.Machine()`, `mk.machine.MACHINE`: `snapshot() -> dict`,
`prepare(needed) -> dict`, with the keys above. `mk.PORTS_VERSION` stays `"1"`: the change is additive.

### Records

- `prepare` emits one span, `hone.models.machine.prepare` (kind `internal`), with
  `hone.models.machine.{needed,unloaded,released,blocked_by,errors,missing}`; status `error` when
  `errors` is not empty. A model yanked from under someone is then visible in the call store, next to
  the calls that follow it. It carries the trace context as every span does (`hone.run_id`, ...).
- `snapshot` emits no span: it is a read, called before every sample, and hone-select stores the result
  in its own sample record.
- The existing warning log style is used when `prepare` is blocked or an unload fails.
- No change to model-call spans. (Recording `vram_gb < size_gb` on every chat span, "this call ran
  partly on the CPU", would be useful but costs an `/api/ps` per call; left for later.)

### Acceptance cases

| AC | Scenario | Expected |
|---|---|---|
| AC-31 | `snapshot()` with fake readers: NVML fake reporting 8 GB and one foreign process; FakeOllama `/api/ps` scripted with a model partly on the CPU; ComfyUI refused; then no NVML and no `nvidia-smi`; then Ollama answering 500 | GB values and `vram_gb < size_gb` as scripted; ComfyUI `running: False` with no entries; `gpus` is `None` (not `[]`, no zeros); Ollama `running: None` with an error and no entries; the lock `held` / `mine` / `holder` correct for a lock held by another process, by this one (`HONE_GPU_LOCK_HELD=1`) and free |
| AC-32 | `prepare(["a"])` with FakeOllama running `a` and `b`, and ComfyUI (fake) holding memory; again with another process's lease in the ledger; again with the lock held by another process; again with the unload of `b` failing | first: `b` unloaded, ComfyUI `/free` called, span recorded; with a foreign lease or the lock: nothing unloaded, no request sent, `blocked_by` names the holder; failing unload: listed in `errors`, span status `error`, `b` still in `loaded_models` |

AC-17 (contract checks) adds `mk.machine.MACHINE` against the shape in §10.

### Tests

- Everything offline with existing tools: FakeOllama's `queue("/api/ps", ...)` and its request log
  (the unloads are `/api/generate` with `keep_alive: 0`), respx for ComfyUI's `/system_stats` and
  `/free` (a public `FakeComfyUI` is not needed), a fake `pynvml` module and a fake `nvidia-smi`
  as the existing `_gpu_memory` tests do, a temporary lock file held by a child process for the lock
  cases, and a second process's ledger entry as in AC-12.
- Unit tests for the tag rule (`name` vs `name:latest`), the GB rounding, and "unknown is `None`" for
  every field.
- A `gpu` test (AC-19 style, under `scripts/gpu-lock.sh`): `snapshot()` on the real machine reports the
  real card and, after the test loads and unloads a small model, `prepare([])` leaves nothing loaded.
- Docs: a section in docs/gpu-and-sessions.md and an offline `examples/machine_state.py` (AC-20).

## Consequences

- Experiments (and hone-flow, and apps) get one call that says what the machine holds, including the
  partial CPU offload that caused the timeouts, and one call that clears leftovers without breaking
  another run.
- "Unknown" is explicit everywhere (`None`, `running: None`, `errors`), so a policy can refuse to start
  on an unknown state instead of reading it as idle.
- Cost: one module of about 150 lines on top of the existing readers, one new reader (`read_gpus`,
  all devices with names and utilization) next to `read_memory`, one span name, two acceptance cases.
  No new dependency.
- Limits: the lock check sees only `flock` users of the same file; a process that uses the GPU with no
  lease and no lock is visible only as a GPU process in `gpus[].processes`, and `prepare` does not stop
  it (hone-models never kills processes). ComfyUI's models are not named. The race above remains.
- `FileLockGpuLease(path)` does not write a `.holder` file, so a Python holder of the machine lock shows
  as `holder: None`; making it write one is a small follow-up if wanted.

## Migration and compatibility

Additive. New module `hone_models.machine`, new name `machine` in `hone_models.__all__`, a new
entry-point group, a new span name and attributes in design/current.md §8. Existing APIs, the lease, the
ledger format, `gpu-lock.sh` and the lock file are unchanged. hone-select's adapter needs this version
of hone-models; hone-select's `models` extra gets the new minimum when both are released.

## Open questions for the owner

1. **Blocked, or unload anyway?** The proposal makes `prepare` unload nothing while another process holds
   a lease or the lock (and report it). The alternative is to unload the models no lease names and keep
   only those a lease names; lease names are often not model names (`gpu:tts`), so that would guess.
   Is "blocked, the caller waits" right?
2. **ComfyUI in `needed`.** ComfyUI's models have no registry ids, so the proposal uses the plain name
   `"comfyui"` in `needed` to keep that server's memory. Fine, or should the registry get entries for
   ComfyUI models (a `comfyui` provider with `vram_gb`)?
3. **Check ComfyUI when not configured?** The proposal checks `http://127.0.0.1:8188` by default because
   this machine runs ComfyUI; on other machines that is a refused connection, reported as
   `running: False`. Or check it only when `HONE_COMFYUI_URL` is set?
4. **`unload_others=True`.** Should the lease's own "unload everything" also stop at another process's
   lease or lock, like `prepare`? Safer, but it changes existing behaviour (a separate small change).
5. **Warm-up.** `prepare` does not load the needed models. Should hone-models offer `mk.machine.load(id)`
   (a zero-token request with `keep_alive`) so the first sample's time does not include loading, or is
   warm-up hone-select's job through a normal call?
