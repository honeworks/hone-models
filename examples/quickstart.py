"""Quickstart: one chat call and what comes back.

What: call a chat model by its registry id and read the result: the text, why it stopped, the token
    usage and the id of the span that recorded the call.

How: `mk.text(model_id)` returns a `TextClient`; `llm.complete(messages, **params)` sends one request and
    returns a `TextResult`. Messages are `{"role": ..., "content": ...}` dicts, as in every chat API;
    params are `temperature`, `top_p`, `seed`, `max_tokens`, `stop`, `think`, `logprobs`.

Why: every other feature (structured output, prompts, decisions) goes through this call. Problems with
    the model's answer (empty reply, invalid JSON) come back as `r.error` instead of raising, so check
    it; configuration and transport problems raise typed errors (retries_and_errors.py). Every call is
    recorded in `${HONE_HOME:-.hone}/models/spans.db` unless you pass `sink=` (records.py).
"""

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)

llm = mk.text("gemma4-12b")  # a registry id; an ad-hoc "ollama:llama3.2:1b" works too (registry.py)
r = llm.complete(
    [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Write a haiku about rain."},
    ],
    temperature=0.2,
    seed=7,
    max_tokens=100,
)
if r.error:
    raise SystemExit(f"the model failed: {r.error}")
print("text:  ", r.text)
print("finish:", r.finish_reason, "| usage:", r.usage, "| model:", r.model, "| span:", r.span_id)

assert r.text == "echo: Write a haiku about rain."  # FakeOllama echoes the last user message
assert r.finish_reason == "stop"  # "length" would mean the reply was cut off at max_tokens
assert set(r.usage) == {"input_tokens", "output_tokens"}

server.stop()
