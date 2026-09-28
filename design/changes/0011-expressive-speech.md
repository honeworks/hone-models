# 0011: Expressive speech (Chatterbox, emotion and intensity, paragraph chunks)

## Status

`implemented in 0.1.0` (the owner asked for it in the plan for the soft-skills course builder, step 1b)

## Context

The owner listened to course-builder's narration (Kokoro-82M through `mk.speech`, change 0006) and it
sounds like someone flatly reading text. Courses need a teacher's voice with feeling: encouraging here,
curious there, serious for a warning. Course-builder's scripts will label each line's intent and narrate
a paragraph at a time, so the model hears whole thoughts and the prosody flows across sentences.

Constraints from 0006 still hold: one 8 GB GPU shared with Ollama and ComfyUI, no sudo, the torch build
pinned in the uv cache (`>=2.6,<2.11`), records for every call, a fake for offline tests, and no
non-commercial weights (not F5-TTS, not XTTS-v2). No real person's voice is cloned (the owner's own
voice comes later, not now).

## Problem

- Kokoro has no expression control: every line comes out with the same even, reading prosody.
- `synthesize` has no way to ask for an emotion or its strength.
- The chunker packs sentences across paragraph boundaries and cuts a long paragraph greedily, so a
  chunk can start in the middle of one thought and end in the middle of the next, or end with a
  one-sentence stub. Each chunk is synthesized on its own, so the melody restarts at every cut.

## Options

1. **Chatterbox** (Resemble AI, MIT, `chatterbox-tts`): 0.5 B Llama backbone plus a flow-matching
   vocoder, English, zero-shot voice from a reference clip, and two expression controls:
   `exaggeration` (emotion intensity, 0.25 to 2, default 0.5) and `cfg_weight` (pacing and adherence,
   lower is slower and more deliberate). About 3 GB of fp32 weights; fits 8 GB. Its wheels pin
   `torch==2.6.0`, `torchaudio==2.6.0` and `numpy<2` exactly, but its code runs on the machine's torch
   2.10 (checked), so the pins can be overridden.
2. **CosyVoice 2** (Apache-2.0): instruction-controlled emotion, but installed from a git checkout with
   a long list of pinned dependencies (and optional system packages); no pip package.
3. **Orpheus-3B** (Apache-2.0): emotion through inline tags (`<laugh>`, `<sigh>`), 3 B parameters (needs a
   4-bit build or vLLM to fit 8 GB next to anything else), and a SNAC decoder.
4. Prompt-level tricks on Kokoro (punctuation, speed changes): no real change in prosody.

API shapes considered: a free-form instruction string (only CosyVoice understands it); provider-specific
knobs (`exaggeration=`, `cfg_weight=`) on the common API (leaks one backend into every app); **a small
shared vocabulary of emotions plus an intensity**, mapped by each provider to what it can do.

## Decision

Option 1, Chatterbox, as a second speech provider, with the shared-vocabulary API:

- `synthesize(text, *, voice=None, speed=1.0, emotion=None, intensity=None, out, trace=None)`.
  `emotion` is one of `tts.emotions` (`hone_models.speech.EMOTIONS`): `neutral`, `calm`, `warm`,
  `serious`, `curious`, `encouraging`, `playful`, `excited`. `intensity` is 0.0 to 1.0 (how strongly;
  `None` takes the emotion's own default). Both are validated for every model (`ConfigError`), so a
  typo fails in tests with the fake.
- A new capability `expressive` (registry `capabilities.expressive`, `SpeechClient.expressive`) says
  whether a model applies them. Kokoro declares `expressive = false` and **ignores** emotion and
  intensity; Chatterbox declares `true`.
- **Chatterbox mapping:** `exaggeration = 0.25 + intensity`, so intensity 0 to 1 spans 0.25 to 1.25
  (0.25 is Chatterbox's neutral default of 0.5). Each emotion has a default intensity and a
  `cfg_weight` (calm 0.1 / 0.5 ... excited 0.75 / 0.3); lower `cfg_weight` slows the delivery that
  higher exaggeration speeds up, as Chatterbox's authors recommend. The table lives in
  `providers/chatterbox.py`.
- **Paragraph chunks** (all providers): a chunk never crosses a paragraph (a blank line). A paragraph
  that fits the model's limit is one chunk; a longer one is split on sentences into chunks of about
  equal length. Chunks are joined with 0.25 s of silence inside a paragraph and 0.6 s between
  paragraphs. The limit is `defaults.max_chunk_chars` in the registry (400 for both packaged models;
  Chatterbox makes at most 1000 speech tokens, about 40 s, per chunk). So one call per paragraph, or
  one call for several paragraphs with the same emotion, synthesizes whole paragraphs.
- **Records:** the speech span adds `hone.models.speech.emotion` and `hone.models.speech.intensity`
  (only when given; a missing value is absent, never 0) and `hone.models.speech.expressive` (whether
  the model applied them).
- **Registry:** `chatterbox` (provider `chatterbox`, extra `expressive`, `vram_gb = 5`, MIT, 24 kHz,
  seed from `defaults.seed`, default 0).
- **Voices:** Chatterbox takes a voice from a reference clip. The packaged voices are clips rendered
  once by Kokoro's synthetic voices (`scripts/make-expressive-voices.py`, about 10 s each, FLAC in
  `hone_models/data/voices/`): `warm_female` (Kokoro `af_heart`, the default), `warm_male`
  (`am_michael`), `british_female` (`bf_emma`), `british_male` (`bm_george`), plus
  `chatterbox_default`, the voice bundled with the Chatterbox weights. No recording of a real person is
  used; `voice` does not accept file paths (cloning comes later, when the owner records a voice).
- **Speed:** Chatterbox has no speed control; `speed != 1.0` time-stretches the result (librosa).
- **GPU:** as for Kokoro: the model is loaded per call inside `mk.gpu.lease("chatterbox", 5)` and freed
  afterwards (CUDA cache and cuBLAS workspaces emptied).
- **Install:** extra `expressive` = `chatterbox-tts>=0.1.7` (Python < 3.13) and `setuptools<81` (its
  watermarker imports `pkg_resources`). The exact pins are replaced with
  `[tool.uv] override-dependencies = ["torch>=2.6,<2.11", "torchaudio>=2.6,<2.11", "numpy>=1.24"]`;
  apps that install the extra copy these lines (overrides only apply in the root project).
- **FakeSpeech** takes `emotion` and `intensity`, validates them the same way, and keeps them in `calls`
  (`(text, voice, speed, emotion, intensity)`); `FakeSpeech(expressive=True)` stands in for Chatterbox.

- **Sessions** (added after apps reported Chatterbox reloading about 3 GB per call; this implements
  option 1 of [0009](0009-speech-sessions-and-timings.md)): `with tts.session():` takes the GPU lease
  once, loads the model on the first `synthesize` call, keeps it for every call in the block and frees
  it once at the end (also on an exception). Outside a session a call still loads and frees the model
  itself. Providers become small engines: `SPEECH[provider](config)` loads the model and returns an
  engine, `engine(chunks, voice, speed, (emotion, intensity))` speaks, `engine.close()` frees. The
  Chatterbox engine prepares a voice's reference clip only when the voice changes and seeds every call,
  so a line sounds the same in or out of a session. Spans add `hone.models.speech.session` and
  `hone.models.speech.loaded` (whether the call loaded the model). `FakeSpeech` counts `loads` and
  `frees`. Sessions do not nest (`ConfigError`).

## Consequences

- Apps get expressive narration by changing the model id and passing `emotion=` per paragraph.
- The `expressive` extra is large (Chatterbox pulls transformers, diffusers, librosa and gradio) and
  depends on overriding its pins; a Chatterbox release that really needs torch 2.6 would break it (the
  `gpu` test would show it).
- Chatterbox loads about 3 GB per call (seconds from a warm disk cache); a session pays it once per
  batch but holds the GPU lease (5 GB) for the whole block, so apps keep sessions to one batch.
- Chatterbox outputs carry Resemble's inaudible Perth watermark (left on).
- Kokoro chunks change slightly: paragraphs are no longer merged into one chunk, and a paragraph pause
  is longer (0.6 s instead of 0.25 s).

## Migration and compatibility

Additive: the new parameters are keyword-only with defaults, and existing calls behave as before apart
from paragraph boundaries (above). `Render` provider functions take `(config, chunks, voice, speed,
emotion, intensity)`. `FakeSpeech.calls` entries grow from three to five items; tests that compare whole
tuples add `None, None`.
