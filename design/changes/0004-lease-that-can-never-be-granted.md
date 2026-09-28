# 0004: A GPU lease that can never be granted should fail, not wait forever

## Status

`implemented in 0.1.0` (options 2, 3 and 4, plus nested leases: see "Implementation" below)

## Context

Found while building OneShotStudio (a demo app built on the honeworks packages), a hone-flow workflow that runs with
`fk.Workflow(gpu="hone_models")` on an 8 GB RTX 4060 Laptop. Its Ollama steps asked the lease for
`vram_gb=7.5` (the registry says `gemma4-12b` needs 7.4 GB). The same process had earlier run torch
models in-process (Whisper, Audiobox, SongEval) and released them, but a CUDA context of ~160 MB stays
for the life of the process, and the driver reserves ~300 MB. NVML then reported `used = 509 MB`, so
`free = 8188 - 509 = 7679 MB`, one MB short of the 7680 MB asked for.

hone-flow calls the lease with `timeout_s=None`, so `GpuScheduler._acquire` unloaded the idle Ollama
models (none were loaded), then polled every 0.5 s forever. The run sat in `running` with the next step
`pending` and no log line for 15 minutes, until it was killed and resumed with a smaller `vram_gb`.

A second case in the next run: the previous `oneshot` process had ended with `qwen2.5vl:7b` still
loaded in Ollama (5.4 GB, keep-alive 30 min). The new process asked for its first `gpu:ollama` lease;
the default scheduler (`unload_others=False`, which is what `gpu="hone_models"` instantiates) unloads
only models *this* process loaded, so it waited, silently, until Ollama's keep-alive would expire. The
app now passes `mk.gpu.GpuScheduler(unload_others=True)` and unloads its models at exit.

## Problem

- A lease whose request is larger than what can ever be free (because the memory is held by the
  requesting process itself, by the driver, or by a program that is not a lease holder and never exits)
  waits forever when no timeout is given. Nothing distinguishes "waiting for another holder" from
  "waiting for memory nobody will release".
- The wait is silent: no log line, no span event, nothing in `mk.gpu.status()` that says who is waiting.
- Through the `hone.gpu_leases` entry point a consumer always gets `unload_others=False`, so an idle
  Ollama model left by an earlier process blocks every lease until its keep-alive expires, even though
  unloading it is harmless. Proposed with the options below: log the idle models that are in the way,
  and consider `unload_others=True` for models idle longer than a few seconds.
- `CapabilityError` is raised only when the request exceeds the GPU's *total* memory, not when it
  exceeds what is achievable with no lease held.

## Options

1. **Document it**: callers must pass a timeout. hone-flow (and every consumer) would need to choose one;
   long waits for real holders are legitimate (hours on a shared machine).
2. **Fail fast when no one holds a lease**: if the ledger has no other live entries, nothing was
   unloaded or could be unloaded, and `free < need` has not changed for N polls, raise
   `CapabilityError("... needs 7680 MB; 7679 MB is free with no leases held: lower vram_gb or free the
   memory held by <pids from NVML>")`.
3. **Count the requesting process's own usage as available**: read per-process usage from NVML
   (`nvmlDeviceGetComputeRunningProcesses`) and treat memory held by this pid as grantable to it.
4. **Log while waiting**: a warning after 30 s and then every few minutes, naming the shortfall, and a
   `waiting` list in `GpuStatus`.

## Decision

Proposed: 2 + 3 + 4. Option 3 removes the self-inflicted case (a process's own CUDA context or cached
torch memory is not a reason to wait for itself); option 2 turns the remaining impossible waits into an
explicit error that says what to do; option 4 makes any long wait visible. Option 1 alone keeps the trap.

## Consequences

- Workflows fail with an actionable message instead of hanging; hone-flow records the failure and
  `resume()` continues after the fix.
- Per-process NVML readings need `nvidia-ml-py` (the `gpu` extra); without it option 3 is skipped and
  option 2 still applies.
- A legitimate wait behind a non-lease GPU user that will exit soon (for example a ComfyUI server being
  stopped) could fail too early under option 2; "unchanged for N polls" with N covering ~2 minutes keeps
  that rare, and the error names the pids so the user sees why.

## Migration and compatibility

No API change. Behaviour change: a lease that previously hung now raises `CapabilityError` (a
`HoneModelsError`) after the stall period, or is granted when the shortfall is the caller's own memory.
Callers that relied on waiting forever for memory held by a non-lease program must pass their own
`timeout_s` loop. New optional field `GpuStatus.waiting`.

## Implementation

Implemented as decided, plus one case found in concept-shorts after this record was written: a hone-flow
`gpu:tts` batch lease of 5 GB held around the Chatterbox speech client's own 5 GB lease, in the same
process and thread, on an 8 GB card. The inner lease waited for memory that only the outer one (its own
caller) could free, forever. A lease taken inside other leases of the same process and thread on the same
ledger (any `GpuScheduler` instance) is now part of them: it reserves only what they do not cover
(nothing, or a ledger entry marked `nested_in` for the difference). Leases in other threads stay separate.

- Option 2: `GpuScheduler(stall_s=120)`; the error names the other processes holding GPU memory and the
  Ollama models still loaded (with the `unload_others=True` hint). It applies with or without `timeout_s`.
- Option 3: `GpuScheduler(processes=read_processes)`, NVML `nvmlDeviceGetComputeRunningProcesses`, else
  `nvidia-smi --query-compute-apps`; read only when memory is short.
- Option 4: a warning after 30 s, then every 5 minutes; `GpuStatus.waiting` (from `gpu-waiting.json` next
  to the ledger, so the ledger format and the free-memory sum stay as they were).
- The `unload_others` default through the `hone.gpu_leases` entry point is unchanged; the stall error
  points at it.
