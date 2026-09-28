"""OpenAI-compatible servers: llama.cpp, vLLM, LM Studio, Ollama's /v1 and OpenAI itself, one entry each.

What: register a model served by any OpenAI-compatible HTTP server and use it exactly like an Ollama
    model: chat, structured output and embeddings.

How: a registry entry with `provider = "openai_compatible"`, the server's `base_url` (ending in `/v1`)
    and, if the server needs a key, `api_key_env` naming the environment variable that holds it (the key
    itself never goes in the file, and is stripped from records). Requests go to
    `{base_url}/chat/completions` and `{base_url}/embeddings`; JSON Schema output uses `response_format`
    when the entry declares `json_schema = true`; `logprobs=True` fills `r.logprobs` (decisions use it
    for calibrated yes/no probabilities).

Why: the same code runs against a local llama.cpp server, a vLLM box or a hosted API by changing one
    registry entry. For OpenAI itself the packaged entry `gpt-4.1-mini` or an ad-hoc id such as
    `openai:gpt-4.1` works once `OPENAI_API_KEY` is set. Pitfall: declare `max_input_tokens` so prompts
    that do not fit are refused before calling (context_budget.py).

Runs offline: `FakeOllama` also serves Ollama's OpenAI-compatible `/v1` endpoints.
"""

import os
import tempfile
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for the server
os.environ["LOCAL_LLM_KEY"] = "local-example-key-123"  # normally set in your shell, never in code

registry_file = Path(tempfile.mkdtemp()) / "models.toml"
registry_file.write_text(
    f"""
[models.local-llama]
provider = "openai_compatible"
model = "llama-3.2-3b-instruct"            # the name the server knows
base_url = "{server.url}/v1"               # e.g. http://localhost:8080/v1 for llama.cpp
api_key_env = "LOCAL_LLM_KEY"
[models.local-llama.capabilities]
json_schema = true
max_input_tokens = 8192

[models.local-embed]
provider = "openai_compatible"
model = "bge-small"
kind = "embedding"
base_url = "{server.url}/v1"
""",
    encoding="utf-8",
)
reg = mk.registry.load(registry_file)
sink = mk.records.MemorySink()

# 1. Chat: same call as for Ollama.
llm = mk.text("local-llama", registry=reg, sink=sink)
r = llm.complete([{"role": "user", "content": "Name a sea shanty."}], temperature=0.2)
print(
    "chat:",
    r.text,
    "| local:",
    llm.config.local,
    "| provider:",
    sink.spans[-1]["attributes"]["gen_ai.provider.name"],
)
assert server.requests[-1]["path"] == "/v1/chat/completions"

# 2. Structured output: the schema goes out as response_format.
r = llm.complete(
    [{"role": "user", "content": "Rate it 1-5."}],
    schema={"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}}},
)
print("structured:", r.parsed, r.structured_path)
assert server.requests[-1]["body"]["response_format"]["type"] == "json_schema"
assert r.structured_path == "constrained"

# 3. Embeddings from the same server.
vectors = mk.embedder("local-embed", registry=reg, sink=sink).embed(["rain", "drizzle"])
print("embeddings:", len(vectors), "x", len(vectors[0]))
assert len(vectors) == 2
assert server.requests[-1]["path"] == "/v1/embeddings"

# The key was sent as a bearer token, and never reaches the records.
assert server.requests[0]["headers"]["Authorization"] == "Bearer local-example-key-123"
assert "local-example-key-123" not in str(sink.spans)

server.stop()
