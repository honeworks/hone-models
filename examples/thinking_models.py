"""Thinking models: reasoning is off unless you ask for it, and a reply with only "thinking" is an error.

What: call a model that can reason before answering (deepseek-r1, gemma4, qwen3 ...) with reasoning off
    (the default) and on, and see how a reply that holds only reasoning text is reported.

How: the registry marks such models with `capabilities.thinking = true`; for them hone-models sends
    Ollama `think: false` unless you pass `think=True` to `complete(...)`. Models without the flag get no
    `think` field at all. When a reply has reasoning but no content, `r.text` is empty and `r.error`
    says so; the result is never a silent empty string.

Why: thinking models spend the output budget on reasoning and can return an empty `content`, which
    broke downstream JSON parsing in earlier projects. Turn reasoning on only where it improves the
    answer, and give it a larger `max_tokens`. Pitfall: `think` is an Ollama option; other providers
    ignore it.
"""

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
sink = mk.records.MemorySink()
question = [{"role": "user", "content": "Is 1001 prime?"}]

llm = mk.text("deepseek-r1-8b", sink=sink)
print("declared thinking:", llm.config.capabilities.thinking)

# 1. Default: reasoning off.
llm.complete(question)
print("default   -> think =", server.requests[-1]["body"]["think"])
assert server.requests[-1]["body"]["think"] is False

# 2. Ask for reasoning (with room for it).
llm.complete(question, think=True, max_tokens=4000)
print("think=True -> think =", server.requests[-1]["body"]["think"])
assert server.requests[-1]["body"]["think"] is True

# 3. The model used its whole budget on reasoning: an explicit error, not an empty answer.
thinking_only = {"role": "assistant", "content": "", "thinking": "1001 = 7 * 11 * 13, so ..."}
server.queue("/api/chat", {"message": thinking_only, "done_reason": "length"})
r = llm.complete(question, think=True, max_tokens=20)
print("thinking only ->", repr(r.text), "| error:", r.error)
assert r.text == ""
assert "thinking" in r.error

# 4. A model not declared as thinking gets no think field.
mk.text("qwen2.5vl-7b", sink=sink).complete(question)
assert "think" not in server.requests[-1]["body"]

server.stop()
