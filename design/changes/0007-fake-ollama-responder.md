# 0007: A responder hook for FakeOllama

## Status

`implemented in 0.1.0` (option 1; the responder's dict is merged over the default reply, like
`queue`, and queued answers come first)

## Context

Found while building course-builder (a demo app built on the honeworks packages). Its tests run the whole pipeline on
`FakeOllama`. With a JSON schema, `FakeOllama` answers a minimal sample (one-element arrays, `"ok"`
strings). This app's gates are strict on purpose (a curriculum has exactly N lessons, a lesson deck has
6-11 slides and 350-1200 narration words, a quiz question has 4 distinct options and evidence found in
the lesson, a blind judge must name the key's letter), so sample answers are rejected and the pipeline
cannot complete on fakes. `queue()` scripts answers in order, which breaks as soon as selections make a
variable number of calls.

The app replaces the private module function `hone_models.testing.fake_ollama.reply` with a wrapper that
answers by schema title (`monkeypatch.setattr(fake_ollama, "reply", ...)`). It works but depends on a
private name.

## Options

1. **`FakeOllama(responder=callable)`**: `responder(path, body) -> dict | None`; `None` falls back to the
   default reply. Public, one argument, covers "answer by schema", "answer by prompt name" and
   "look at the question and answer correctly".
2. Schema-title fixtures: `FakeOllama(answers={"Curriculum": {...}})`. Simpler but cannot compute
   answers from the request (the blind-judge case).

## Decision

Proposed: option 1 (option 2 can be written on top of it in a few lines).
