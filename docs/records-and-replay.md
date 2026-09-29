# Records and replay

Every call writes one span to the sink passed as `sink=` (default: SQLite at
`${HONE_HOME:-.hone}/models/spans.db`). Sinks: `SqliteSpanSink(path)`, `JsonlSpanSink(path)`,
`MemorySink()`, `NullSink()`; any object with `emit(span)`, `flush()` and `close()` works.

```python
import hone_models as mk

sink = mk.records.MemorySink()
mk.text("gemma4-12b", sink=sink).complete([{"role": "user", "content": "hi"}], temperature=0.3)
span = sink.spans[0]
print(
    span["name"], span["attributes"]["gen_ai.request.model"], span["attributes"]["gen_ai.usage.output_tokens"]
)
```

Main attributes (full list in [design/current.md §8.3](../design/current.md#83-attributes)): `gen_ai.operation.name`, `gen_ai.provider.name`,
`gen_ai.request.{model,temperature,top_p,max_tokens,seed}`, `gen_ai.response.{model,finish_reasons}`,
`gen_ai.usage.{input_tokens,output_tokens}`, `gen_ai.input.messages`, `gen_ai.output.messages`,
`hone.models.model_id`, `hone.models.model_digest`, `hone.models.prompt.*`,
`hone.models.structured.{path,attempts,schema}`, `hone.models.context.{limit,estimated_prompt_tokens}`,
`hone.models.cost_usd`, `hone.models.request.params`, `hone.models.gpu.*`, `hone.models.replay_of`,
`hone.models.decision.{state,questions,answers}`, and for images, music and video
`hone.models.media.{prompt,inputs,outputs,job_id,workflow_sha256,session,loaded,freed,server_started,queue_wait_ms,error,error_kind,license,commercial_use,cost_estimated}`
([generation.md](generation.md#records)). Generated files are recorded by path and hash, never their
bytes. `mk.machine.prepare` and `mk.machine.load` record
`hone.models.machine.prepare` / `hone.models.machine.load` spans (kind `internal`) with
`hone.models.machine.*` attributes. A cancelled job is a `cancelled` event. HTTP retries are `retry` events; structured-output
retries are `structured_retry` events. LiteLLM models retry inside LiteLLM, without events.

**Content capture.** `HONE_CAPTURE_CONTENT=0` or `SqliteSpanSink(path, capture_content=False)` stores
messages, outputs, variables, params, decision content, and a generation call's prompt, error and text
inputs (lyrics, for example; numbers and file records stay readable) only as `{"sha256", "len"}`.

**Secrets.** Headers are never recorded; the value of every `api_key_env` in use, and anything shaped like
a bearer token or `sk-...` key, is replaced with `***` before writing. Register other values with
`mk.records.add_secret(value)`.

**Reading records.** `mk.records.read_spans(path)` returns every span of a SQLite store, oldest first;
`mk.records.default_store()` is the default store's path. `hone_models.calls.find_calls(spans, since="1d",
model=...)` and `call_stats(calls, by="model" | "provider" | "tag")` are the queries behind
`hone-models calls` ([cli.md](cli.md)).

## Replay

```python
import hone_models as mk

sink = mk.records.MemorySink()
llm = mk.text("gemma4-12b", sink=sink)
llm.complete(mk.Prompt("p", "1", {"rules": "Be brief.", "brief": "Name a fish."}))
new = mk.replay.Replayer(sink=sink).replay_call(sink.spans[0], {"prompt.sections": {"rules": None}})
print(new["attributes"]["hone.models.replay_of"] == sink.spans[0]["span_id"])
```

Overrides: `"prompt.sections"` (`{id: new_text | None}`), `"model"`, `"params"` (merged) or `"messages"`
(full replacement; not together with `prompt.sections`). Everything else is taken from the span, which
must have been recorded with content capture on.

Spans from other recorders work too: list and object attributes may be JSON strings (the OpenTelemetry
encoding), and section records without a `role` get the role of the message their `start` falls in.
`Replayer` is registered in the entry-point group `hone.replayers` as `hone_models` (created without
arguments), so `hone-lens test F-0001 --replayer hone_models` replays through it.

**Runnable examples:** [records.py](../examples/records.py), [prompt_sections_and_replay.py](../examples/prompt_sections_and_replay.py).
