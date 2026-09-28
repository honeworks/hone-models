"""Context budget: prompts that do not fit fail before calling, and cut-off replies are always visible.

What: see how the prompt size and `max_tokens` are checked against the model's context window, what is
    sent to Ollama (`num_ctx`, `num_predict`), and how to detect a reply truncated at `max_tokens`.

How: before each request hone-models estimates the prompt tokens (characters / 3.5) and adds
    `max_tokens` (default: the model's `max_output_tokens`, else 2048). If that exceeds the registry's
    `max_input_tokens`, `mk.errors.ContextOverflow` is raised with the numbers and no request is sent.
    Otherwise Ollama gets `num_ctx` sized to the request, so the server never truncates the prompt at its
    own default. A reply cut off at `max_tokens` has `r.finish_reason == "length"`.

Why: Ollama silently truncates prompts longer than its default context, and a long JSON answer cut off at
    the output limit looks like a model failure. Here both are explicit. Fixes, in order: shorten the
    prompt, lower `max_tokens`, or pick a larger model with `require={"min_context": n}` (registry.py).
    Pitfall: the estimate is rough; leave headroom rather than aiming at the exact limit.
"""

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
sink = mk.records.MemorySink()
llm = mk.text("gemma4-12b", sink=sink)  # max_input_tokens = 32768, max_output_tokens = 8192
long_text = "The cargo hold is full of lost luggage. " * 2500  # 100,000 characters, ~28,600 tokens

# 1. Prompt + the default output budget (8192) does not fit: raised before any request.
try:
    llm.complete([{"role": "user", "content": long_text}])
except mk.errors.ContextOverflow as exc:
    print("refused:", exc)
else:
    raise AssertionError("the prompt cannot fit with the default output budget")
assert not server.requests  # nothing was sent

# 2. A smaller output budget fits; num_ctx is sized to the request and sent with it.
llm.complete([{"role": "user", "content": long_text}], max_tokens=2000)
options = server.requests[-1]["body"]["options"]
attrs = sink.spans[-1]["attributes"]
print("sent: num_ctx =", options["num_ctx"], "num_predict =", options["num_predict"])
print(
    "recorded: ~",
    attrs["hone.models.context.estimated_prompt_tokens"],
    "prompt tokens of",
    attrs["hone.models.context.limit"],
)
assert options["num_predict"] == 2000
assert attrs["hone.models.context.estimated_prompt_tokens"] + 2000 <= options["num_ctx"] <= 32768

# 3. A reply cut off at max_tokens: plain text calls report it in finish_reason; check it.
server.queue(
    "/api/chat",
    {"message": {"role": "assistant", "content": "Verse one: the cargo"}, "done_reason": "length"},
)
r = llm.complete([{"role": "user", "content": "Write a long shanty."}], max_tokens=5)
if r.finish_reason == "length":
    print("truncated:", repr(r.text), "-> raise max_tokens or ask for less")
assert r.finish_reason == "length"

# With a schema a cut-off reply is never "constrained": it is "repaired" or "failed" (chat_and_structured.py).

server.stop()
