# 0006: Local text-to-speech (`mk.speech`)

## Status

`implemented in 0.1.0`

## Context

The owner asked for local text-to-speech for the showcase apps (in the plan for the showcase apps, §8): concept
shorts, mini-courses, faceless explainers and a teacher persona all need narration, some of it minutes
long. They run on one 8 GB GPU next to Ollama and ComfyUI, record every model call, and must be testable
offline. hone-models already gives chat, decisions and embeddings one interface with records and GPU
leases; speech is the next model kind those apps need.

## Problem

- No speech model kind: each app would wire a TTS library by hand, outside the GPU lease and without
  records.
- Apps must start now, before a real backend is proven on this machine, so they need a fake.
- TTS models synthesize a few hundred characters well; narration is minutes long.
- There is no sudo on the machine, so a backend needing system packages (espeak-ng) may not install;
  non-commercial models (e.g. XTTS-v2) are out.

## Options

1. **Leave TTS to applications.** Duplicated code, no records, no lease.
2. **One backend-agnostic speech client in hone-models**: `mk.speech(model_id)` with one method,
   `synthesize`, backed by a local model installed as an optional extra and downloaded on first use,
   plus a `FakeSpeech` for tests.
3. **A TTS server protocol** (an OpenAI-compatible `/audio/speech` endpoint). Needs a server running
   beside Ollama and ComfyUI; nothing on the machine serves it today.

## Decision

Option 2, kept as small as §8 of the plan asks:

- `mk.speech(model_id)` returns a `SpeechClient` with
  `synthesize(text, *, voice=None, speed=1.0, out, trace=None) -> SpeechResult`
  (`path`, `duration_s`, `sample_rate`, `voice`, `model`, `span_id`). The WAV is written to `out`.
- Registry entries of `kind = "speech"` and a new provider `kokoro`: **Kokoro-82M** (Apache-2.0,
  82 M parameters, under 1 GB of VRAM). Its voices are listed in `capabilities.voices`; the first is the
  default. The `speech` extra installs the `kokoro` package; the weights come from Hugging Face on first
  use into its cache (`HF_HOME`, default `~/.cache/huggingface`).
- Long text is split into sentences, packed into chunks of at most 400 characters, synthesized chunk by
  chunk and concatenated with a short pause, so minutes-long narration works.
- Each call records one `hone.models.speech` span (`gen_ai.operation.name = "speech"`). The text goes in
  the content attribute `hone.models.speech.input`, so content capture rules apply to it; voice, speed,
  character count, chunk count, duration and sample rate are plain attributes.
- Synthesis runs inside `mk.gpu.lease(model_id, vram_gb)`; the model is freed (`close()`, CUDA cache
  emptied) after each call so it never lingers on the shared GPU. Loading takes about a second.
- `hone_models.testing.FakeSpeech` writes a silent WAV of about `words / 2.5` seconds and records the
  same span. (This record first said `mk.speech("fake")` returns one; there is no `fake` registry id:
  construct `FakeSpeech()` directly, see [0009](0009-speech-sessions-and-timings.md).)

## Consequences

- Apps narrate with three lines and see each synthesis in their call records.
- The `speech` extra pulls torch and spaCy (large); the core stays small and never imports it.
- Kokoro supports Python < 3.13 only; the extra is marked accordingly.
- Other backends (Piper, Chatterbox) can be added as further `provider` values without API changes.

## Migration and compatibility

Additive: new function, result type, registry kind and provider, extra and fake. Existing registries
and spans are unchanged.
