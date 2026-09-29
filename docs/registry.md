# The model registry

TOML files are merged in this order (later ones override key by key):

1. the packaged catalog (`hone_models/data/models/<kind>.toml`, one file per kind: every model we use or
   may use, installed or not, with its licence, install table and guide; see
   [models-and-guides.md](models-and-guides.md))
2. `~/.config/hone/models.toml` (user)
3. `./hone-models.toml` (project)
4. paths passed to `mk.registry.load(paths=[...])` or `--registry` on the CLI

```toml
[models."gemma4-12b"]
provider = "ollama"              # ollama | openai_compatible | litellm | jev | kokoro | chatterbox | comfyui | command | faster_whisper | none
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
# base_url_env = "LLAMA_URL"     # optional; this variable, when set, replaces base_url
```

**Endpoints and keys from the environment.** `api_key_env` names the variable that holds an entry's key;
`base_url_env` names the variable that holds its base URL. When that variable is set it replaces
`base_url` (which then serves as the fallback); with neither, calling the model raises `ConfigError`
naming the variable before any request. A hosted judge behind an OpenAI-compatible gateway, committed
without its URL or key:

```toml
# hone-models.toml
[models.fable]
provider = "openai_compatible"
model = "fable"
base_url_env = "JUDGES_BASE_URL"
api_key_env = "JUDGES_API_KEY"
```

The values can live in a `.env` file in the directory the program runs from:

```bash
# .env: keep it out of git (add ".env" to .gitignore)
JUDGES_BASE_URL=https://gateway.example/v1
export JUDGES_API_KEY="your-key"   # "export" and quotes are optional
```

`mk.registry.load()` reads the file named by `HONE_ENV_FILE` (if set; it must exist) and then `./.env`,
each once per process, and exports only variables that are not set yet: the shell's environment wins,
then `HONE_ENV_FILE`, then `./.env`. Lines are `KEY=VALUE`; blank lines and `#` lines are skipped; a
quoted value is taken literally (no escapes); an unquoted value ends at ` #`. A malformed line raises
`ConfigError` naming the file and line number (never the line's text). Keys are never recorded, and
neither is the `user:password` of a URL (stored as `https://***@...`).

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

`require` keys are capability names (exact match) plus `min_context` and `features` (every listed feature
of the model's guide declared, case ignored); `prefer` is `local`, `hosted`, `cheapest` or `fastest`. No
match raises `CapabilityError` listing the closest candidates. `mk.select(require, kind=...)` returns every
match instead of the best one:

```python
import hone_models as mk

print([m.id for m in mk.select({"features": ["camera angle"]}, kind="image")])  # ['qwen-image-edit-2511']
```

The catalog declares only capabilities that were checked, and it lists models this machine may not have:
name the model when you call one (`mk.text("gemma4-12b")`) rather than relying on the first match.

`hone-models models check <id>` makes a smoke call to a chat model (which loads it), times a warm
~200-token reply and saves that `speed_tok_s` to the user registry; it prints the load time apart (for an image, music or video entry it runs a tiny job and reports the peak GPU memory
for `vram_gb`: [generation.md](generation.md#packaged-comfyui-models)); request timeouts grow with it: `2 * max_tokens / speed_tok_s`, at least 120 s, at most
`max_timeout_s` (default 600 s). A local model that was never measured is assumed to make 10 tokens/s,
so a long answer (`max_tokens=7000`) gets up to 600 s on a fresh machine; hosted models without a
measurement get 120 s.

**Generation models.** `kind` is `chat` (default), `embedding`, `decision`, `speech`, `image`, `music`,
`video`, `transcription` or `scoring` (catalog entries hone-models cannot call yet). A `comfyui` entry adds `workflow`, an `inputs` table of workflow paths and
`outputs`; a `command` entry adds `command`, `cwd`, `env` and its project folder variable
`install.dir_env` (registry files are trusted configuration: a `command` entry runs the program it
names); generation capabilities are `max_references`, `sizes`, `max_duration_s`, `durations_s`,
`word_timestamps`, `commercial_use` (information only: it never blocks a call, and it is copied onto every
result) and `features`; `price` also takes `per_image` and `per_output_second`. `max_timeout_s` defaults
by kind: image 600 s, music 1800 s, video 3600 s, transcription 600 s. `lyrics_format`, `prompt_inputs`, `guide` and `install` are described in
[models-and-guides.md](models-and-guides.md). An
`openai_compatible` entry of kind `image` or `video` calls a hosted images or video API; its `inputs` is a
list of extra input names sent as request fields. See [generation.md](generation.md#comfyui-entries) and
[hosted entries](generation.md#hosted-images-and-video) and
[standalone projects](generation.md#standalone-projects-command-entries).

**Runnable examples:** [registry.py](../examples/registry.py), [openai_compatible.py](../examples/openai_compatible.py), [litellm_provider.py](../examples/litellm_provider.py).
