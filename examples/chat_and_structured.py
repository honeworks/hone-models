"""Structured output: a Pydantic model or JSON Schema in, a validated object out, and how it was obtained.

What: ask for JSON matching a schema and walk through every `structured_path` the result can report:
    constrained, parsed, retried, repaired and failed.

How: pass `schema=` (a Pydantic class or a JSON Schema dict) to `llm.complete(...)`. The pipeline
    (1) constrains decoding when the model declares `json_schema` (Ollama `format`, OpenAI
    `response_format`), (2) parses, stripping markdown fences and prose, (3) validates, (4) retries up to
    2 times sending the validation error back, (5) repairs truncated JSON as a last resort. `r.parsed` is
    the object (or `None`), `r.structured_path` says which step produced it, `r.attempts` counts calls.

Why: models return fenced, chatty, invalid or cut-off JSON. This never hides that: `r.parsed is None`
    means `r.error` says why, and a `repaired` result may have lost items at the end (the reply was cut
    off at `max_tokens`), so decide whether that is acceptable for your use. Retries are recorded as
    `structured_retry` span events.
"""

import json

from pydantic import BaseModel, Field

import hone_models as mk
from hone_models.testing import FakeOllama


class Verdict(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str


class Idea(BaseModel):
    title: str
    hook: str


class Ideas(BaseModel):
    ideas: list[Idea]


server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
sink = mk.records.MemorySink()
llm = mk.text("gemma4-12b", sink=sink)
ask = [{"role": "user", "content": "Rate this pitch 1-5: a shanty about spreadsheets."}]

# constrained: the model supports JSON Schema decoding and returned clean JSON.
r = llm.complete(ask, schema=Verdict)
print(f"{r.structured_path:<11} {r.parsed!r} after {r.attempts} attempt(s)")
assert isinstance(r.parsed, Verdict)
assert r.structured_path == "constrained"
assert "format" in server.requests[-1]["body"]  # the schema went to Ollama at the top level

# parsed: the JSON was wrapped in prose and a markdown fence and had to be cleaned up.
fenced = 'Sure!\n```json\n{"score": 4, "rationale": "catchy"}\n```'
server.queue("/api/chat", {"message": {"role": "assistant", "content": fenced}})
r = llm.complete(ask, schema=Verdict)
print(f"{r.structured_path:<11} {r.parsed!r}")
assert r.structured_path == "parsed"
assert r.parsed == Verdict(score=4, rationale="catchy")

# retried: the first reply failed validation (score 9 > 5); the error was sent back and the model fixed it.
server.queue("/api/chat", {"message": {"role": "assistant", "content": '{"score": 9, "rationale": "wow"}'}})
r = llm.complete(ask, schema=Verdict)
feedback = server.requests[-1]["body"]["messages"][-1]["content"]
print(f"{r.structured_path:<11} {r.parsed!r} after {r.attempts} attempts; sent back: {feedback[:60]!r}")
assert (r.structured_path, r.attempts) == ("retried", 2)
assert [e["name"] for e in sink.spans[-1]["events"]] == ["structured_retry"]

# repaired: the reply was cut off at max_tokens; complete elements were kept, the rest dropped.
cut = '{"ideas": [{"title": "Cargo Hold Chaos", "hook": "lost luggage"}, {"title": "Bilge Bl'
server.queue("/api/chat", {"message": {"role": "assistant", "content": cut}, "done_reason": "length"})
r = llm.complete([{"role": "user", "content": "Three song ideas."}], schema=Ideas, max_tokens=40)
print(f"{r.structured_path:<11} {len(r.parsed.ideas)} idea(s) kept, finish_reason={r.finish_reason}")
assert r.structured_path == "repaired"
assert [idea.title for idea in r.parsed.ideas] == ["Cargo Hold Chaos"]
assert r.finish_reason == "length"

# failed: no usable JSON after 3 attempts. Nothing raises: parsed is None and error says why.
prose = {"message": {"role": "assistant", "content": "I'd rather sing it."}}
server.queue("/api/chat", prose, prose, prose)
r = llm.complete(ask, schema=Verdict)
print(f"{r.structured_path:<11} parsed={r.parsed} error={r.error[:70]!r}...")
assert r.parsed is None
assert r.error
assert sink.spans[-1]["status"]["code"] == "error"

# A JSON Schema dict works too; the result is a plain dict.
tags = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["tags"],
}
r = llm.complete([{"role": "user", "content": "Tag this song."}], schema=tags)
print("dict schema:", json.dumps(r.parsed), r.structured_path)
assert r.parsed == {"tags": ["ok"]}

server.stop()
