"""LiteLLM provider (extra `litellm`): any hosted model LiteLLM knows, behind the same `mk.text` call.

What: call a model through LiteLLM from a registry entry, with structured output, and see how the ad-hoc
    `litellm:<provider>/<model>` ids work for hosted models.

How: `pip install "hone-models[litellm]"`. Either use an ad-hoc id such as
    `mk.text("litellm:anthropic/claude-sonnet-5")` (LiteLLM reads the provider's key from its usual
    environment variable, e.g. `ANTHROPIC_API_KEY`, and the capabilities come from LiteLLM's model map),
    or register the model with `provider = "litellm"`, `model = "<litellm model name>"` and optionally
    `base_url` (sent as `api_base`) and `api_key_env`. Requests and replies use the OpenAI shape, so
    everything else (schemas, records, errors) is the same as for the other providers.

Why: one dependency gives access to dozens of hosted APIs without a provider module each. Use the
    `openai_compatible` provider instead when the server already speaks the OpenAI API (fewer moving
    parts). Pitfalls: LiteLLM retries transient errors itself, so they are not `retry` span events; set
    `LITELLM_LOCAL_MODEL_COST_MAP=True` to stop LiteLLM fetching its model map from the internet at import.

Runs offline: LiteLLM's "openai/" route is pointed at `FakeOllama`'s OpenAI-compatible `/v1` endpoints.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # before LiteLLM is imported

import hone_models as mk
from hone_models.testing import FakeOllama

try:
    import litellm  # noqa: F401 - only checking that the extra is installed
except ImportError:
    raise SystemExit("this example needs the extra: pip install 'hone-models[litellm]'") from None

server = FakeOllama().start()  # offline stand-in for a hosted API
os.environ["EXAMPLE_API_KEY"] = "sk-example-0123456789abcdef"  # normally set in your shell

registry_file = Path(tempfile.mkdtemp()) / "models.toml"
registry_file.write_text(
    f"""
[models.hosted-mini]
provider = "litellm"
model = "openai/gpt-4.1-mini"      # a LiteLLM model name: "<provider>/<model>"
base_url = "{server.url}/v1"       # only for self-hosted or proxied endpoints; omit for the real API
api_key_env = "EXAMPLE_API_KEY"
[models.hosted-mini.capabilities]
json_schema = true
max_input_tokens = 128000
""",
    encoding="utf-8",
)
reg = mk.registry.load(registry_file)
sink = mk.records.MemorySink()
llm = mk.text("hosted-mini", registry=reg, sink=sink)

r = llm.complete([{"role": "user", "content": "Name a sea shanty."}], max_tokens=50)
print("text:", r.text, "| usage:", r.usage)
assert r.text == "echo: Name a sea shanty."

r = llm.complete(
    [{"role": "user", "content": "Rate it 1-5."}],
    schema={"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}}},
)
print("structured:", r.parsed, r.structured_path)
assert r.structured_path == "constrained"

span = sink.spans[-1]
print(
    "recorded provider:",
    span["attributes"]["gen_ai.provider.name"],
    "| model:",
    span["attributes"]["gen_ai.request.model"],
)
assert span["attributes"]["gen_ai.provider.name"] == "litellm"
assert server.requests[-1]["path"] == "/v1/chat/completions"

server.stop()
