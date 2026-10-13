# 0020: Image models that work as shipped, guides per situation, pose support and cancel

## Status

`accepted` 2026-10-13 (owner, for hone-frame change 0008: "unlike hone-music-video, where we added things
manually, it's not allowed here"; "all model knowledge moves to hone-models"; "even add the research only
ones ... but show a warning"; "pause will stop the current image as well").

## Context

hone-frame (and hone-music-video) ship their own registry entries and ComfyUI workflows for the image
models they use, because the catalog entries for `flux.2-klein-4b`, `qwen-image-edit-2511` and
`qwen-image-2.1` have no workflow and cannot be called. Each library also keeps its own notes on how the
models behave (faces at small sizes, hands, several people, reference order, pose pictures). hone-frame
0008 gives the person full control: they write every prompt themselves, from one complete guide built
from what the models can and cannot do. That knowledge has to live with the models.

hone-frame 0008 also pauses and stops a run in the middle of an image; a media call can only be stopped
today by its timeout or by an exception in the calling thread.

## Problem

1. Four image models a person would pick are not callable from the catalog.
2. A guide has a summary, a prompt line and features; it cannot say how a model behaves for faces, hands,
   several people, poses, objects, places, text or references, which is what a prompt writer needs.
3. Nothing says whether a model takes a pose skeleton as a control input, only as an ordinary picture, or
   not at all.
4. A caller cannot cancel a running media call from another thread.

## Options

- Notes as free text in `guide.prompt`: unstructured, a tool cannot show the notes for one situation.
- Notes per situation as a list (chosen): `about` from a fixed vocabulary, so hone-frame can show the
  face notes on a face item and the scene notes on a scene.
- Cancel through `max_timeout_s` only: the run cannot stop now.

## Decision

1. **Catalog entries with workflows** (`data/workflows/`): `flux.2-klein-4b` (two reference slots, the
   last reference repeated into an empty slot, D-017 of hone-frame), `flux.2-klein-4b-text` (no
   references), `qwen-image-edit-2511` (Q4 GGUF, Multiple-Angles LoRA, the 8-step Lightning LoRA, three
   slots), `qwen-image-2.1-turbo` (int8, the Turbo LoRA, up to ten slots, 8 steps; Qwen research
   licence, `commercial_use = false`). Workflows and input maps are the ones proven on the 8 GB reference
   machine in hone-frame and hone-music-video. `z-image-turbo` gains `sizes`. Every image entry declares
   `license`, `commercial_use`, `max_references`, `sizes` and `vram_gb`.
2. **Guide notes per situation**: `[[models.<id>.guide.notes]]` with `about` (one of `general`, `faces`,
   `hands`, `people`, `poses`, `objects`, `places`, `scenes`, `text`, `references`, `sizes`), `says` (one
   or two plain sentences), optional `source` and `checked`. `ModelGuide.notes`, `notes_about(about)`,
   and `as_text()` / `as_dict()` include them grouped by situation. Unknown `about` values are refused at
   load with the allowed list.
3. **Pose support**: `capabilities.pose` = `"none"` (pose pictures are not useful), `"reference"` (a
   skeleton or mannequin goes in as an ordinary reference picture; works less exactly than a control),
   `"control"` (a pose skeleton is a control input, named by the entry's `pose` input). Shown in the guide.
   The shipped image models are `"reference"` except `z-image-turbo` (`"none"`: no references).
4. **Cancel**: `MediaClient` calls take `cancel: threading.Event | None`. The ComfyUI provider checks it
   while waiting; when set it cancels the job (the existing `_cancel`) and raises `Cancelled` (a new
   `HoneModelsError`), recorded as outcome `cancelled`. Providers that cannot cancel finish the call and
   then raise `Cancelled`.

## Consequences

- hone-frame and hone-music-video can delete their own entries and workflows for these models.
- Notes are documentation (as the guide is): never used by a call.
- More catalog entries to keep proven; each workflow is checked by the existing workflow-path tests.

## Migration and compatibility

Additive. Existing guides without notes are valid; `pose` unknown is `None`. A caller that does not pass
`cancel` behaves as before.

## Acceptance

- The four entries load, are callable against the fake ComfyUI server, and map prompt, seed, size, steps
  and every reference slot.
- `mk.guide("qwen-image-edit-2511").notes_about("faces")` returns its face notes; `as_text()` shows notes
  grouped by situation and the pose support.
- An unknown `about` is a `ConfigError` naming the allowed values.
- Setting `cancel` during a fake ComfyUI job cancels the job (the cancel endpoint is called) and raises
  `Cancelled` within one poll interval.
