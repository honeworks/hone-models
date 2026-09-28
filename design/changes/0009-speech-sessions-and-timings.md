# 0009: Speech sessions and sentence timings

## Status

`implemented in 0.1.0`: option 1 (sessions) with [0011](0011-expressive-speech.md), option 2 (timings)
as `SpeechResult.segments` and `synthesize(..., by_sentence=True)`.

## Context

Found while building explainer-channel (a demo app built on the honeworks packages). A 10-minute explainer is about 110
narrated sentences. Captions and picture cuts must sit on sentence boundaries for the whole episode, so
the app synthesizes each sentence with its own `synthesize()` call and joins the WAVs itself (the join
points are the timings; Whisper on the result agrees within about 0.05 s).

Two costs follow from the current API (0006):

- The Kokoro provider loads the model and builds a `KPipeline` (spaCy, phonemizer) on every call and frees
  it afterwards. Measured on this machine on CPU: 10-15 s per sentence, almost all of it loading; the
  synthesis itself is under 2 s for a 5 s sentence. On an episode that is 110 loads.
- `SpeechResult` has no timings, although the provider already splits the text into sentences and
  concatenates their audio, so it knows where each sentence starts.

Also: 0006 says `mk.speech("fake")` returns a `FakeSpeech`; the registry has no `fake` id, so it raises
`ConfigError` (apps construct `hone_models.testing.FakeSpeech()` instead).

## Problem

- Many short syntheses in a row pay the model load each time.
- Apps that need timings must call per sentence (and pay the load) instead of once per text.

## Options

1. **A session**: `with tts.session(): ...` keeps the model and pipeline loaded (inside one GPU lease)
   for the calls in the block and frees them at exit; outside a session the behaviour stays as today.
2. **Timings in the result**: `SpeechResult.segments: list[(text, start_s, end_s)]`, one per sentence
   chunk the provider synthesized (the join points it already has). One call per chapter then gives the
   app everything it needs.
3. Leave it to apps (today: per-sentence calls, about 2 minutes of GPU time per episode on loads alone).

## Decision

Proposed: 1 and 2 (both additive). Split chunks on sentence boundaries only (never inside a sentence), so
`segments` line up with sentences. Fix the `"fake"` id or the 0006 text.

## Consequences

- A session holds VRAM (under 1 GB) for its duration; the lease covers it.
- Spans: one per `synthesize()` call as today; the session adds a `hone.models.speech.session` attribute.

## Implementation

`SpeechResult.segments` is a list of `SpeechSegment(text, start_s, end_s)`, one per synthesized chunk,
computed from the PCM lengths and the pauses the join inserts. Since 0011 a paragraph that fits is one
chunk (prosody across sentences), so a segment is one or more whole sentences; `by_sentence=True` makes
each sentence its own chunk (an over-long sentence is still cut at spaces) and the segments are then
sentence timings. The `"fake"` id: the 0006 text is corrected (there is no such registry id; construct
`FakeSpeech()`).
