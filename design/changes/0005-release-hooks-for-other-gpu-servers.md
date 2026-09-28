# 0005: Release hooks for GPU servers other than Ollama

## Status

`implemented in 0.1.0`

## Context

Found while building OneShotStudio (a demo app built on the honeworks packages), which alternates on one 8 GB GPU between
Ollama (writing and judging), a ComfyUI server (ACE-Step songs, z-image pictures) and in-process torch
models (Whisper, Audiobox, SongEval). hone-flow batches the steps by `gpu:` tag and holds hone-models'
lease around each batch. When memory is short, the lease unloads the Ollama models this process loaded,
which covers the Ollama → ComfyUI switch. The other switches the app handles by hand:

- ComfyUI → torch scorers: the app calls ComfyUI's `POST /free` after each render;
- torch scorers → ComfyUI: the app calls each scorer's `close()` and `torch.cuda.empty_cache()`;
- ComfyUI → Ollama: the app stops the ComfyUI server at the end of its batch (and pays a restart on the
  next batch), because a running ComfyUI keeps its models in VRAM and the lease cannot ask it to let go.

## Problem

The lease knows how to make room only in Ollama. Every other GPU user must be released by application
code placed exactly right, and a mistake shows up as an out-of-memory error in a model call far away
from the cause, or (with change 0004's problem) as a lease that waits forever.

## Options

1. **Keep it in applications** (today).
2. **Release hooks**: `mk.gpu.on_short(name, release: Callable[[], None])` registers a function the
   lease calls, in registration order, when memory is short (after unloading Ollama models, before
   waiting). Hooks for common servers ship as helpers: `mk.gpu.comfyui_free(url)` (POST `/free` with
   `unload_models`), `mk.gpu.torch_empty_cache()`.
3. **Leases for servers**: model long-running servers as lease holders that yield on request. Needs a
   protocol those servers do not speak.

## Decision

Proposed: option 2. It keeps the lease the single place that makes room, works with servers as they are
(ComfyUI already has `/free`), and costs one small registry. Option 3 is out of reach.

## Consequences

- Applications register a hook once (`mk.gpu.on_short("comfyui", mk.gpu.comfyui_free(url))`) instead of
  sprinkling frees; a ComfyUI server can stay up across batches when memory allows, saving restarts.
- Hooks run in the leasing process; a hook that raises is logged and skipped.
- Spans record which hooks ran (`hone.models.gpu.released`), next to `hone.models.gpu.unloaded`.

## Migration and compatibility

Additive. With no hooks registered the lease behaves as today.

## Implementation

As decided (option 2). Hooks are per process, not per scheduler, so an app's own
`GpuScheduler(unload_others=True)` and the library's `mk.gpu.GPU` share them. `on_short(name, None)`
removes a hook. `mk.gpu.torch_empty_cache` is itself the hook (no factory); it is a no-op without torch.
Hooks run once per lease, together with the Ollama unloads.
