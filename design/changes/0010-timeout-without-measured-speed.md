# 0010: A timeout that grows with max_tokens before speed is measured

## Status

`implemented in 0.1.0` (option 1: 10 tokens/s for local models without `speed_tok_s`; option 2 is
not built)

## Context

Found in concept-shorts (a demo app built on the honeworks packages). Its script step asks `gemma4-12b` (7.4 GB, partly
offloaded on an 8 GB card, ~12-20 tokens/s) for one structured answer with `max_tokens=7000`: a whole
scene script as JSON. Both drafts failed with `ModelTimeout ... did not answer within 120s; set
capabilities.speed_tok_s or max_timeout_s`, and the step failed. `timeout_for()` only grows beyond 120 s
when `capabilities.speed_tok_s` is known, which on a fresh machine it is not unless someone ran
`hone-models models check`. The app fixed it with a project registry entry (`speed_tok_s = 12`).

## Problem

The caller already says how long an answer may be (`max_tokens`); for a local model with unknown speed,
a flat 120 s turns a legitimate long request into a failure, and the message is only discoverable after
losing the call.

## Options

1. **Assume a conservative speed for local models** without a measurement (e.g. 10 tokens/s for
   `local` providers): `expected = 2 * max_tokens / 10`, still capped by `max_timeout_s`.
2. **Measure on first use**: time the first call per model and store `speed_tok_s` like
   `models check` does, then use it for later calls (the first long call can still fail).
3. **Keep 120 s** and document it more prominently.

## Decision

Proposed: option 1, with option 2 as a later refinement.

## Consequences

- Long structured answers from slow local models no longer fail on a fresh machine.
- A truly hung server is detected later (at most `max_timeout_s`, 600 s by default).
