# Testing without models

`hone_models.testing.FakeOllama` is a small local HTTP server with the Ollama API. It sets `OLLAMA_HOST`
while it runs, so code under test needs no change:

```python
import hone_models as mk
from hone_models.testing import FakeOllama

with FakeOllama() as server:
    r = mk.text("ollama:any-model", sink=mk.records.NullSink()).complete([{"role": "user", "content": "hi"}])
    assert r.text == "echo: hi"
    print(server.requests[-1]["path"])  # /api/chat
```

Chat replies echo the last user message; with a JSON Schema the reply is a sample object matching it
(first enum value, middle of a number range, `"ok"`, `true`), so decisions and structured output work.
Embeddings are stable 8-dimensional hash vectors.
It also serves Ollama's OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings`, for registry
entries with `provider = "openai_compatible"` and `base_url = server.url + "/v1"`.

`server.queue(path, *answers)` scripts the next answers for one path, then the default replies resume: a
dict is merged over the default reply, an int is an HTTP error status. Use it to test truncation,
thinking-only replies, invalid JSON and server errors:

```python
import hone_models as mk
from hone_models.testing import FakeOllama

with FakeOllama() as server:
    server.queue(
        "/api/chat", {"message": {"role": "assistant", "content": "cut"}, "done_reason": "length"}, 503
    )
    llm = mk.text("gemma4-12b", sink=mk.records.NullSink())
    assert llm.complete([{"role": "user", "content": "hi"}]).finish_reason == "length"
    assert llm.complete([{"role": "user", "content": "hi"}]).text == "echo: hi"  # 503 was retried
```

When the number or order of calls depends on the code under test, a queue breaks. `FakeOllama(responder=fn)`
answers from the request instead: `fn(path, body)` returns a dict merged over the default reply (as with
`queue`), or `None` to keep the default. Queued answers still come first.

```python
import json

import hone_models as mk
from hone_models.testing import FakeOllama


def by_schema(path: str, body: dict) -> dict | None:
    if path == "/api/chat" and (body.get("format") or {}).get("title") == "Verdict":
        return {"message": {"role": "assistant", "content": json.dumps({"score": 5})}}
    return None


schema = {"title": "Verdict", "type": "object", "properties": {"score": {"type": "integer"}}}
with FakeOllama(responder=by_schema):
    llm = mk.text("gemma4-12b", sink=mk.records.NullSink())
    assert llm.complete([{"role": "user", "content": "rate"}], schema=schema).parsed == {"score": 5}
```

`server.requests` holds each request's `path` (without the query string), JSON `body` and `headers`.
`FakeOllama().start()` / `.stop()` do the same as the `with` block, for scripts.

`hone_models.testing.check_record_sink(sink, read_back)` is the contract check for a `RecordSink`
implementation ([design/current.md §8.2](../design/current.md#82-sinks-and-the-sqlite-store)).

## Bring your own client
Code written against the honeworks ports accepts anything with the same shape. A text client only needs
`complete(messages, *, schema=None, trace=None, **params)` returning an object with `text`, `parsed`,
`error`, `model`, `finish_reason`, `usage` and `span_id`; the exact shapes are in
[design/current.md §10](../design/current.md#10-shapes-other-code-can-rely-on).

**Runnable examples:** [testing_with_fake_ollama.py](../examples/testing_with_fake_ollama.py).
