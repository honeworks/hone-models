# Command line (`cli` extra)

```text
hone-models models list [--json] [--registry FILE]
hone-models models show <id> [--json]
hone-models models check <id> [--json]      smoke call; saves measured tokens/s as speed_tok_s
hone-models calls list [--since 1d] [--model ID] [--db PATH] [--json]
hone-models calls show <span_id> [--db PATH]
hone-models calls stats [--by model|provider|tag] [--since 1d] [--db PATH] [--json]
```

- `--since` takes `30m`, `12h`, `1d`, `2w` or an ISO time.
- `calls` read `${HONE_HOME:-.hone}/models/spans.db` unless `--db` is given; each model request is counted
  once (an emulated decision appears as its chat call).
- `--by tag` groups by the `hone.step` trace attribute (set by hone-flow).
- Errors print `error: ...` and exit with code 1.
- In code: `hone_models.calls.find_calls(mk.records.read_spans(path), since=..., model=...)` and
  `call_stats(calls, by=...)` return the same rows.

**Runnable examples:** [calls_cli.py](../examples/calls_cli.py).
