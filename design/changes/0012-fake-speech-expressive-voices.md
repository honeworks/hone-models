# 0012: FakeSpeech(expressive=True) accepts the Chatterbox voices

## Status

`implemented in 0.1.0`: option 3 (`FakeSpeech.like(model_id)`); option 2 in a compatible form:
`FakeSpeech(expressive=True)` without `voices=` accepts Kokoro's voices (still first, so the default and
existing tests that pass `af_heart` keep working, as in how-it-works-3d) plus the `chatterbox` entry's.

## Context

Found in data-stories (a demo app built on the honeworks packages). The app narrates with `mk.speech("chatterbox")` and the
voice `warm_male`, and tests on `FakeSpeech(expressive=True)`, which the docs present as the stand-in for
`chatterbox`. The narration step failed in the test with `ConfigError: unknown voice 'warm_male' for
'fake-speech'`: the fake's default voices are Kokoro's, whatever `expressive` says. The app now passes
`voices=("warm_male", ...)` explicitly.

## Problem

A test written from the docs ("`FakeSpeech(expressive=True)` stands in for `chatterbox`") fails on the
first real voice name, and the fix (a `voices=` tuple copied from the registry) can drift from the real
model's list.

## Options

1. Keep it; document `voices=` next to `expressive=True`.
2. When `expressive=True` and `voices` is not given, default to the `chatterbox` registry entry's voices.
3. `FakeSpeech.like("chatterbox")`: a fake with the capabilities (voices, expressive, max chunk) of a
   registry entry.

## Recommendation

Option 3 (it also covers Kokoro and any project registry entry), with option 2 as its shortcut. Small:
read the entry through the registry, copy `capabilities`, keep the fake provider.
