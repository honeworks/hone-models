"""Records: every call as a span, joined to the caller's trace, with content capture and secrets handled.

What: choose where spans go, read one back, link calls to a run through the trace context, turn content
    capture off, and see API keys stripped from what is stored.

How: every client takes `sink=`: `mk.records.SqliteSpanSink(path)` (the default, at
    `${HONE_HOME:-.hone}/models/spans.db`), `JsonlSpanSink(path)`, `MemorySink()` (tests) or `NullSink()`.
    Spans follow the shared honeworks records format (OpenTelemetry-style: `gen_ai.*` and `hone.*`
    attributes). Pass `trace={"traceparent": "00-<trace id>-<parent span id>-01", "hone.run_id": ...}`
    to `complete` / `decide` / `embed` to make the call a child of your span and copy the shared `hone.*`
    keys onto it. `capture_content=False` (or env `HONE_CAPTURE_CONTENT=0`) stores hashes and lengths
    instead of prompts and outputs. Values of `api_key_env` variables, `Bearer ...` tokens, `sk-...` keys
    and anything passed to `mk.records.add_secret` are replaced by `***` before a span is stored.

Why: records answer "what exactly did the model see and say, and what did it cost" long after the run,
    and hone-lens reads them next to the other packages' records; calls_cli.py queries them. Pitfalls:
    with capture off a call cannot be replayed (prompt_sections_and_replay.py); sinks never raise into
    your code, they log once and count `sink.failures`.
"""

import json
import tempfile
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
folder = Path(tempfile.mkdtemp())

# 1. A JSON-lines sink: one span per line, easy to grep or ship elsewhere.
jsonl = mk.records.JsonlSpanSink(folder / "spans.jsonl")
r = mk.text("gemma4-12b", sink=jsonl).complete([{"role": "user", "content": "Name a sea shanty."}])
span = json.loads((folder / "spans.jsonl").read_text().splitlines()[-1])
attrs = span["attributes"]
print(
    span["name"],
    span["status"]["code"],
    attrs["gen_ai.request.model"],
    attrs["gen_ai.usage.output_tokens"],
    "tokens",
)
print("  input:", attrs["gen_ai.input.messages"], "\n  output:", attrs["gen_ai.output.messages"])
assert span["span_id"] == r.span_id

# 2. Trace context: the call becomes a child of the caller's span and carries its run id.
trace_id, parent_id = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
sink = mk.records.MemorySink()
mk.text("gemma4-12b", sink=sink).complete(
    [{"role": "user", "content": "hi"}],
    trace={"traceparent": f"00-{trace_id}-{parent_id}-01", "hone.run_id": "run-42", "hone.step": "draft"},
)
child = sink.spans[-1]
print(
    "trace:",
    child["trace_id"],
    "parent:",
    child["parent_span_id"],
    "run:",
    child["attributes"]["hone.run_id"],
)
assert (child["trace_id"], child["parent_span_id"]) == (trace_id, parent_id)
assert child["attributes"]["hone.step"] == "draft"

# 3. Content capture off: prompts and outputs are stored as hashes and lengths only.
private = mk.records.MemorySink(capture_content=False)
mk.text("gemma4-12b", sink=private).complete([{"role": "user", "content": "patient record 1234"}])
stored = private.spans[-1]["attributes"]["gen_ai.input.messages"]
print("capture off:", stored)
assert set(stored) == {"sha256", "len"}
assert "patient" not in json.dumps(private.spans)

# 4. Secrets: key-like values never reach a sink, even if they end up in a prompt.
mk.records.add_secret("hunter2-internal-token")  # your own secrets (API keys from api_key_env are automatic)
sink = mk.records.MemorySink()
mk.text("gemma4-12b", sink=sink).complete(
    [{"role": "user", "content": "keys: sk-live0123456789abcdefXYZ and hunter2-internal-token"}]
)
stored_text = json.dumps(sink.spans[-1])
print("stripped:", sink.spans[-1]["attributes"]["gen_ai.input.messages"][0]["content"])
assert "sk-live" not in stored_text
assert "hunter2" not in stored_text

# 5. The SQLite store (the default sink) and reading it back.
db = mk.records.SqliteSpanSink(folder / "spans.db")
mk.text("gemma4-12b", sink=db).complete([{"role": "user", "content": "one more"}])
stored_spans = mk.records.read_spans(folder / "spans.db")
print("sqlite spans:", [s["name"] for s in stored_spans])
assert len(stored_spans) == 1

server.stop()
