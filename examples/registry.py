"""Model registry: model ids, TOML files, capabilities, and choosing a model by what it can do.

What: add and override models in a registry file, look models up by id, use ad-hoc `provider:model` ids,
    and let hone-models pick a model that meets requirements (`require=`, `prefer=`).

How: `mk.registry.load(paths)` merges, in order, the packaged defaults, `~/.config/hone/models.toml`,
    `./hone-models.toml` and the files you pass (later files win, key by key). `reg.get(id)` returns a
    `ModelConfig`; `reg.select(require, prefer)` or `mk.text(require=..., prefer=..., registry=reg)` picks
    the best matching model. `require` keys are capability names (exact value) or `min_context`;
    `prefer` is "local", "hosted", "cheapest" or "fastest".

Why: capabilities (vision, context size, JSON schema, thinking) are checked before any request is sent,
    and model names live in one file instead of being copied across modules. Pitfalls: a
    `./hone-models.toml` in the working directory is loaded automatically; ad-hoc ids know nothing until
    their first call probes the provider (`ollama:` via `/api/show`, `litellm:` via LiteLLM's model map;
    `openai:` ids stay unknown), so register the models you use often.
"""

import tempfile
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)

# 1. A registry file: override one field of a packaged model and add a new model.
path = Path(tempfile.mkdtemp()) / "models.toml"
path.write_text(
    """
[models."gemma4-12b"]
defaults = { temperature = 0.2 }   # merged over the packaged entry; everything else is kept

[models."llama-small"]
provider = "ollama"
model = "llama3.2:1b"              # the name the provider knows; the table name is the id
[models."llama-small".capabilities]
vision = false
json_schema = true
max_input_tokens = 131072
""",
    encoding="utf-8",
)
reg = mk.registry.load(path)
print("registered:", sorted(reg.models))

gemma = reg.get("gemma4-12b")
print("gemma4-12b:", gemma.provider, gemma.name, gemma.defaults, "local" if gemma.local else "hosted")
assert gemma.defaults == {"temperature": 0.2}  # from our file
assert gemma.capabilities.thinking is True  # from the packaged defaults

# 2. Select by capability. Both llama-small and gpt-4.1-mini have 100k+ context; "local" wins here.
long_context = reg.select({"min_context": 100_000}, prefer="local")
vision = mk.text(require={"vision": True}, prefer="local", registry=reg)
print("long context:", long_context.id, "| vision:", vision.model_id)
assert long_context.id == "llama-small"
assert vision.model_id == "qwen2.5vl-7b"

# 3. Nothing meets the requirement: CapabilityError names the closest candidates.
try:
    reg.select({"vision": True, "thinking": True})
except mk.errors.CapabilityError as exc:
    print("no match:", exc)
else:
    raise AssertionError("no registered model has both vision and thinking")

# A registered id that lacks a required capability fails before any request.
try:
    mk.text("gemma4-12b", require={"vision": True}, registry=reg)
except mk.errors.CapabilityError as exc:
    print("refused:", exc)
else:
    raise AssertionError("gemma4-12b is declared without vision")

# 4. Ad-hoc ids: "<provider>:<model>" with provider ollama, openai, litellm or jev.
adhoc = mk.text("ollama:mistral:7b", registry=reg, sink=mk.records.MemorySink())
print("ad-hoc before first call:", adhoc.config.capabilities.max_input_tokens)  # unknown: None
assert adhoc.config.capabilities.max_input_tokens is None
adhoc.complete([{"role": "user", "content": "hi"}])
print("ad-hoc after probing:   ", adhoc.config.capabilities.max_input_tokens)  # from /api/show
assert adhoc.config.capabilities.max_input_tokens == 8192  # what the fake server reports

# 5. Unknown ids are a ConfigError that lists what exists.
try:
    reg.get("gemma-99")
except mk.errors.ConfigError as exc:
    print("unknown:", str(exc)[:80], "...")
else:
    raise AssertionError("gemma-99 is not registered")

server.stop()
