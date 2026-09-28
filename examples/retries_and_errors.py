"""Retries and errors: what is retried, what raises, and what comes back as `result.error`.

What: see a transient server error retried and recorded, a client error raised at once, the typed
    exception hierarchy, and a model-quality problem reported on the result instead of raised.

How: every HTTP request is tried up to 3 times when the failure is transient (connection errors, HTTP
    429, HTTP 5xx), waiting up to 0.5 s, then up to 1 s (jittered); each retry is a `retry` event on the
    call's span. When the attempts run out, `ProviderError` is raised (`.status` is None). Other HTTP 4xx
    raise `ProviderError` at once, with `.status`; a read timeout raises `ModelTimeout` at once. All
    exceptions derive from `mk.errors.HoneModelsError` (a `RuntimeError`):
        ConfigError       bad registry entry, unknown model id, missing API key, bad argument
        CapabilityError   the model cannot do this (vision, required capability); raised before calling
          ContextOverflow prompt + max_tokens does not fit the context window
        ProviderError     the server failed (unreachable, HTTP error, malformed reply); `.status`
          ModelTimeout    no answer within the timeout
        ValidationFailed  output does not match a schema (used inside the structured pipeline)
    Problems with the model's answer (empty reply, JSON that never validates) are not exceptions: the
    call returns with `r.error` set and its span has status "error".

Why: callers can catch exactly what they can handle: fix configuration (`ConfigError`), fall back to
    another model (`CapabilityError`), retry later (`ProviderError`), or skip one item (`r.error`)
    without stopping a batch. Pitfall: do not wrap calls in your own retry loop; it multiplies the
    built-in retries and hides them from the records.
"""

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
sink = mk.records.MemorySink()
llm = mk.text("gemma4-12b", sink=sink)
hello = [{"role": "user", "content": "hello"}]

# 1. Transient errors (503, then 429) are retried; the call succeeds and the retries are recorded.
server.queue("/api/chat", 503, 429)
r = llm.complete(hello)
retries = [e["attributes"] for e in sink.spans[-1]["events"] if e["name"] == "retry"]
print("after retries:", r.text, "| retry events:", [(e["attempt"], e["reason"]) for e in retries])
assert r.error is None
assert [e["reason"] for e in retries] == ["http 503", "http 429"]

# 2. A client error (400) is not retried: ProviderError with the status, at once.
server.queue("/api/chat", 400)
sent_before = len(server.requests)
try:
    llm.complete(hello)
except mk.errors.ProviderError as exc:
    print("raised:", type(exc).__name__, "status", exc.status)
    status = exc.status
else:
    raise AssertionError("HTTP 400 must raise")
assert status == 400
assert len(server.requests) == sent_before + 1  # one request: no retry

# 3. Configuration mistakes are ConfigError. Catch HoneModelsError to handle every hone-models error.
try:
    mk.text("no-such-model")
except mk.errors.HoneModelsError as exc:
    print("raised:", type(exc).__name__, "-", str(exc)[:60], "...")
    raised = exc
else:
    raise AssertionError("an unknown model id must raise")
assert isinstance(raised, mk.errors.ConfigError)

# 4. The model answered, but with nothing: not an exception; the result says what went wrong.
server.queue("/api/chat", {"message": {"role": "assistant", "content": "  "}})
r = llm.complete(hello)
print("result error:", r.error, "| span status:", sink.spans[-1]["status"]["code"])
assert r.error == "model returned an empty response"
assert sink.spans[-1]["status"]["code"] == "error"

server.stop()
