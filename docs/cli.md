# Command line (`cli` extra)

```text
hone-models models list [--kind K] [--feature F ...] [--installed | --missing] [--tier N] [--json] [--registry FILE]
hone-models models show <id> [--json]
hone-models models guide <id> [--json]      what the model takes: guide, inputs, features, limits, licence
hone-models models guide [<id>] --stale 90  guides not checked for 90 days
hone-models models install <id> [--run]     prints the commands that fetch the model; downloads nothing
hone-models models check <id> [--json] [--out DIR]  chat: saves the warm tokens/s as speed_tok_s;
                                            image/music/video: a tiny job, its file and peak GPU memory
hone-models calls list [--since 1d] [--model ID] [--db PATH] [--json]
hone-models calls show <span_id> [--db PATH]
hone-models calls stats [--by model|provider|tag] [--since 1d] [--db PATH] [--json]
```

- `models list` shows whether each model is installed here (`yes`, `no`, `unknown`: a server that does
  not answer or an unset variable is `unknown`); `--feature` keeps the models whose guide declares every
  given feature; `--tier` is the catalog's tier (1 test first, 2 worth a try, 3 the ceiling). See
  [models-and-guides.md](models-and-guides.md).
- `models check` on a chat entry makes a smoke call ("Say OK.", up to 1024 tokens, so a model that
  thinks anyway still answers), which also loads the model, then times a ~200-token reply: `speed_tok_s`
  is that warm reply's output tokens over its time and is saved to the user registry; `load_s` is the
  smoke call less its own tokens at that speed (mostly loading), `seconds` the timed reply.
- `models check` on an image, music or video entry runs one tiny job in a session (starting ComfyUI
  from `HONE_COMFYUI_START` if nothing answers) and prints the file and `peak_vram_gb`; see
  [generation.md](generation.md#packaged-comfyui-models).
- `models install` prints `ollama pull`, `hf download ... --local-dir <ComfyUI>/models/<folder>` or the
  clone and setup steps, with the size and the free disk space; `--run` runs them (for the owner of the
  machine).
- `--since` takes `30m`, `12h`, `1d`, `2w` or an ISO time.
- `calls` read `${HONE_HOME:-.hone}/models/spans.db` unless `--db` is given; each model request is counted
  once (an emulated decision appears as its chat call).
- `--by tag` groups by the `hone.step` trace attribute (set by hone-flow).
- Errors print `error: ...` and exit with code 1.
- In code: `hone_models.calls.find_calls(mk.records.read_spans(path), since=..., model=...)` and
  `call_stats(calls, by=...)` return the same rows.

**Runnable examples:** [calls_cli.py](../examples/calls_cli.py).
