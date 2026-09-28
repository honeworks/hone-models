# 0003: Examples set and a scriptable FakeOllama

## Status

`implemented in 0.1.0`

## Context

After the first build, the few examples covered only some of the public concepts. A reader needs one
runnable, explained example per concept, and those examples must run offline and deterministically so
the test suite can execute them. Writing them showed that the packaged `FakeOllama` test server could
only give fixed answers: it could not produce retried, repaired or failed structured output,
thinking-only replies, server errors, OpenAI-compatible servers or Jev replies.

## Problem

- No complete, tested set of examples, and no rule for what an example must contain.
- The public test double was too rigid for the failure cases the package exists to handle, so examples
  (and users' own tests) would have had to patch internals.
- Some names the examples need (reading spans back, the default store path, registering extra secrets,
  the `calls` queries, a scheduler with its own ledger) were used but not documented as public.

## Options

1. Examples that monkeypatch internals.
2. A separate scripted fake per provider.
3. Extend the public `FakeOllama` so it can script any answer, and make every name the examples use
   public and documented.

## Decision

Option 3:

- `FakeOllama.queue(path, *answers)` scripts the next answers for one path (a dict is merged over the
  default reply, an int is an HTTP error status); `start()` / `stop()` for scripts; Ollama's
  OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings`; request `headers` in `server.requests`.
  Jev is scripted by queueing its `/decide` reply (D-017).
- Public, documented names beyond the core API: `mk.records.read_spans`, `default_store`, `add_secret`;
  the `hone_models.calls` module (`find_calls`, `call_stats`); `mk.gpu.GpuScheduler(ledger, memory=...)`
  (D-018).
- One example per public concept, each opening with What / How / Why, asserting its key facts, using
  only the public API, and indexed in `examples/README.md`. A new guarantee, AC-20, checks all of this.
  The test runs every example with `OLLAMA_HOST` on a closed port, so a real local Ollama is never
  reached by accident.

## Consequences

- Better: every concept has a runnable reference; users get the same scriptable fake for their own
  tests; the examples found and fixed two bugs (cost not recorded for replies that omit a zero-priced
  token count, span ids shortened in narrow terminals).
- Costs: `FakeOllama` is now a larger public surface to keep stable.

## Migration and compatibility

Additive. Existing uses of `FakeOllama` as a `with` block with default answers are unchanged.
