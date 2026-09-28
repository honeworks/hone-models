# 0002: Replay entry point and spans from other recorders

## Status

`implemented in 0.1.0`

## Context

Found while testing hone-models together with a trace analyzer that reads recorded calls from many
tools and replays them with one prompt section changed, to check whether a suspected cause of a problem
really is the cause. The analyzer loads a replayer by name and hands it spans as it stored them.

## Problem

- The text client, decision client, embedder and GPU lease had entry points, but the replayer did not:
  tools that load components by name had no way to find it.
- Replay accepted only spans exactly as hone-models writes them. Spans read back through the
  OpenTelemetry encoding carry list and object attributes as JSON strings, and prompt-section records
  written by other recorders have no `role`, so such spans could not be replayed.
- Calls made during an analyzer's replay could not be told apart from ordinary calls: the analyzer's
  finding id in the trace context was not copied onto the spans.
- The contract checks for the shapes hone-models provides were copies inside hone-models' own tests,
  which could drift from the checks of the code that consumes them.

## Options

1. Leave replay to the analyzer: it would need its own model-calling code, duplicating hone-models.
2. Ask other recorders to write exactly hone-models' span format.
3. Make the replayer loadable by name and tolerant of the standard encodings, and carry the finding id
   through the trace context like the other correlation keys.

## Decision

Option 3:

- Entry point `hone.replayers`: `hone_models = hone_models.replay:Replayer`, created without
  arguments.
- `Replayer.replay_call` accepts list and object attributes encoded as JSON strings and section records
  without a `role` (they take the role of the message their `start` falls in). Without
  `hone.models.model_id`, the model comes from `gen_ai.provider.name` + `gen_ai.request.model` as an
  ad-hoc id (D-014).
- `hone.lens.finding_id` joins the correlation keys copied from the trace context onto every span.
- The contract tests import the checks from the packages that consume these shapes (a `dev` extra)
  instead of keeping a copy (D-016).

## Consequences

- Better: any tool can load hone-models' replayer by name; spans from other
  recorders can be replayed; replayed calls are traceable to the finding that caused them.
- Costs: developing hone-models now needs the consuming packages installed for its contract tests; until
  they are published this is done through local path sources (D-016, awaiting owner review).

## Migration and compatibility

Additive. Spans written before this change replay as before; existing callers are unaffected.
