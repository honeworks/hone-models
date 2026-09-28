# Concepts

**Clients.** `mk.text(...)`, `mk.decision(...)` and `mk.embedder(...)` return small client objects bound to
one model. They are the honeworks `TextClient`, `DecisionClient` and `Embedder` ports, so other packages
can take them without importing hone-models.

**The registry** says, for every model id, which provider serves it, the provider's name for it and what
it can do (vision, thinking, JSON schema, logprobs, context size, VRAM, price). Checks happen **before**
a request is sent: an image to a model declared without vision raises `CapabilityError`, a prompt that
cannot fit raises `ContextOverflow` with the numbers.

**Results vs. exceptions.** Problems with the model's *answer* (empty reply, invalid JSON after retries,
a truncated reply) are reported on the result: `r.error` is set, nothing is silently empty. Problems with
the *call* (unreachable server, HTTP 4xx, unknown model, missing capability, missing API key) raise typed
exceptions from `mk.errors`, all subclasses of `HoneModelsError` (itself a `RuntimeError`).

```python
import hone_models as mk

try:
    mk.text("no-such-model")
except mk.errors.ConfigError as exc:
    print(exc)  # lists the registered ids and the ad-hoc id format
```

**Records.** Every call becomes one span (`hone.models.chat`, `hone.models.decide`, `hone.models.embed`)
in `${HONE_HOME:-.hone}/models/spans.db`, using OpenTelemetry GenAI attribute names where they exist. See
[records-and-replay.md](records-and-replay.md).

**Trace context.** Pass `trace={"traceparent": "00-<trace id>-<parent span id>-01", ...}` to any call to
join a caller's trace; `mk.current_trace()` returns the active context to hand to other packages.

**Runnable examples:** [quickstart.py](../examples/quickstart.py), [retries_and_errors.py](../examples/retries_and_errors.py), [records.py](../examples/records.py).
