# The model registry

TOML files are merged in this order (later ones override key by key):

1. packaged defaults (`hone_models/data/models.toml`: `gemma4-12b`, `qwen2.5vl-7b`, `deepseek-r1-8b`,
   `nomic-embed-text`, `jev`, `gpt-4.1-mini`)
2. `~/.config/hone/models.toml` (user)
3. `./hone-models.toml` (project)
4. paths passed to `mk.registry.load(paths=[...])` or `--registry` on the CLI

```toml
[models."gemma4-12b"]
provider = "ollama"              # ollama | openai_compatible | litellm | jev | kokoro | chatterbox | comfyui
model    = "gemma4-12b:latest"   # the provider's name (defaults to the id)
defaults = { temperature = 0.8 } # params used unless the call passes its own
[models."gemma4-12b".capabilities]
vision = false
thinking = true                  # think=false is sent unless the call asks for reasoning
json_schema = true               # constrained decoding (Ollama format / OpenAI response_format)
logprobs = false
max_input_tokens = 32768
max_output_tokens = 8192
vram_gb = 7.4

[models."local-llama"]
provider = "openai_compatible"
base_url = "http://localhost:8080/v1"
api_key_env = "LLAMA_KEY"        # optional; the key is read from this variable and never recorded
```

**Local tags.** An entry's `model` is the provider's name for it; override it in the user or project
file to use the tag your machine has, keeping the registry id your code uses. For example, with Gemma 4
12B pulled from the Ollama library (`ollama pull gemma4:12b`), or under a local tag of your own:

```toml
# ~/.config/hone/models.toml: only the keys you set replace the packaged ones
[models."gemma4-12b"]
model = "gemma4:12b"          # or a local alias such as "gemma4-12b:latest"
```

A capability that is not declared is *unknown* (`None`): checks only block on a declared `false`.

Vision models can declare what an image costs in the prompt, so the context budget counts it:
`image_tokens = 256` (a flat cost per image) or `image_patch_px = 28` (one token per 28x28 pixels, as in
Qwen2.5-VL; the size is read from the image header). Without either, an image counts as 1024 tokens.
`defaults = { min_num_ctx = 8192 }` makes that the smallest context sent to Ollama.

**Ad-hoc ids** use `provider:model` and need no registry entry: `ollama:llama3.2:1b`, `openai:gpt-4.1`,
`litellm:anthropic/claude-sonnet-5`, `jev:<model>`. Ollama and LiteLLM ids are probed for capabilities
(Ollama `/api/show`, LiteLLM's model map); others stay unknown. Registered entries are not probed:
declare their capabilities in the TOML.

A LiteLLM model (extra `litellm`) is any id LiteLLM understands:

```toml
[models.sonnet]
provider = "litellm"
model = "anthropic/claude-sonnet-5"
api_key_env = "ANTHROPIC_API_KEY"   # optional: without it LiteLLM reads its own variables, which are then
                                    # not registered for exact-value secret stripping
base_url = "https://my-proxy/v1"    # optional (sent as LiteLLM's api_base)
[models.sonnet.capabilities]
vision = true
max_input_tokens = 200000
```

**Selection** picks among registered models:

```python
import hone_models as mk

reg = mk.registry.load()
print(reg.get("gemma4-12b").name)  # gemma4-12b:latest
print(reg.select({"vision": True, "min_context": 8000}, prefer="local").id)  # qwen2.5vl-7b
vision = mk.text(require={"vision": True}, prefer="local")
```

`require` keys are capability names (exact match) plus `min_context`; `prefer` is `local`, `hosted`,
`cheapest` or `fastest`. No match raises `CapabilityError` listing the closest candidates.

`hone-models models check <id>` makes a smoke call and saves the measured `speed_tok_s` to the user
registry; request timeouts grow with it: `2 * max_tokens / speed_tok_s`, at least 120 s, at most
`max_timeout_s` (default 600 s). A local model that was never measured is assumed to make 10 tokens/s,
so a long answer (`max_tokens=7000`) gets up to 600 s on a fresh machine; hosted models without a
measurement get 120 s.

**Generation models.** `kind` is `chat` (default), `embedding`, `decision`, `speech`, `image`, `music`,
`video` or `transcription`. A `comfyui` entry adds `workflow`, an `inputs` table of workflow paths and
`outputs`; generation capabilities are `max_references`, `sizes`, `max_duration_s`, `durations_s`,
`word_timestamps`, `commercial_use` (information only: it never blocks a call, and it is copied onto every
result) and `features`; `price` also takes `per_image` and `per_output_second`. `max_timeout_s` defaults
by kind: image 600 s, music 1800 s, video 3600 s, transcription 600 s. The keys `lyrics_format`,
`prompt_inputs`, `guide`, `install`, `command`, `cwd` and `env` are checked when the registry loads. See
[generation.md](generation.md#comfyui-entries).

**Runnable examples:** [registry.py](../examples/registry.py), [openai_compatible.py](../examples/openai_compatible.py), [litellm_provider.py](../examples/litellm_provider.py).
