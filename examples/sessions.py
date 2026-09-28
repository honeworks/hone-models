"""Local server lifecycle: `mk.session` makes sure Ollama runs; `mk.unload` frees a model's memory now.

What: wrap a batch of local model calls in a session that starts the Ollama server only if it is not
    already running, and unload a model as soon as you are done with it.

How: `with mk.session("ollama") as url:` checks the server at `OLLAMA_HOST` (default
    `http://127.0.0.1:11434`); if it answers, it is used and left running; if not, and it is local,
    `ollama serve` is started (with a clean environment) and stopped when the block ends.
    `mk.unload(model_id)` asks Ollama to drop the model from memory (`keep_alive: 0`) instead of waiting
    for its keep-alive timeout.

Why: scripts and CI jobs should not need "start Ollama first" instructions, and must not stop a server
    someone else is using. Unloading right after a batch frees GPU memory for the next job (see
    gpu_lease.py, which also unloads idle models automatically when a lease needs room). Pitfalls: a
    remote `OLLAMA_HOST` that does not answer raises `ProviderError` (it is never started from here);
    only Ollama models can be unloaded (`ConfigError` otherwise).

Runs offline: `FakeOllama` is "already running", so the session uses it and starts nothing.
"""

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
sink = mk.records.MemorySink()

with mk.session("ollama") as url:
    print("using the Ollama server at", url)
    llm = mk.text("gemma4-12b", sink=sink)
    for line in ["first verse", "second verse"]:
        print(" ", llm.complete([{"role": "user", "content": line}]).text)
assert url == server.url  # it was running, so it was used (and is still running)

mk.unload("gemma4-12b")  # free its memory now
unload = server.requests[-1]
print("unload request:", unload["path"], unload["body"])
assert unload["body"] == {"model": "gemma4-12b:latest", "keep_alive": 0}

# Hosted models have nothing to unload.
try:
    mk.unload("gpt-4.1-mini")
except mk.errors.ConfigError as exc:
    print("refused:", exc)
else:
    raise AssertionError("only Ollama models can be unloaded")

server.stop()
