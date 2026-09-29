"""The `hone-models` command: inspect recorded calls and registered models, and smoke-test a model.

What: record a few calls, then list them, show one, and sum tokens, errors and time per model with
    `hone-models calls ...`; list and check models with `hone-models models ...`; and do the same
    queries from Python with `hone_models.calls`.

How: install the `cli` extra (`pip install "hone-models[cli]"`), then
        hone-models calls list  [--since 1d] [--model ID] [--db PATH] [--json]
        hone-models calls show  <span_id> [--db PATH]
        hone-models calls stats [--by model|provider|tag] [--since 1d] [--db PATH] [--json]
        hone-models models list|show <id> [--json] [--registry FILE]
        hone-models models check <id> [--json]    smoke call; saves the warm tokens/s
    `calls` read `${HONE_HOME:-.hone}/models/spans.db` unless `--db` is given. In code,
    `hone_models.calls.find_calls(spans, since=..., model=...)` and `call_stats(calls, by=...)` are the
    same queries over `mk.records.read_spans(path)`.

Why: "which calls failed today, and what did they cost?" should not need a notebook. `models check`
    confirms a model answers before a long run and stores its speed in the user registry
    (`~/.config/hone/models.toml`), which sizes timeouts for slow models. Pitfall: an emulated decision
    and its chat call are counted once (as the chat call), so tokens are not double-counted.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import hone_models as mk
from hone_models.calls import call_stats, find_calls
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST, inherited by the CLI)
db = Path(tempfile.mkdtemp()) / "spans.db"
sink = mk.records.SqliteSpanSink(db)

# Record some calls: two chats, one failed chat and one emulated decision.
llm = mk.text("gemma4-12b", sink=sink)
llm.complete([{"role": "user", "content": "one"}])
llm.complete([{"role": "user", "content": "two"}])
server.queue("/api/chat", {"message": {"role": "assistant", "content": ""}})
failed = llm.complete([{"role": "user", "content": "three"}])
mk.decision("qwen2.5vl-7b", sink=sink).decide("It rains.", {"wet": mk.YesNo("Is it wet outside?")})

cli = Path(sys.executable).with_name("hone-models")  # the command installed next to this Python
if not cli.exists():
    raise SystemExit("this example needs the extra: pip install 'hone-models[cli]'")


def hone_models(*args: str, env: dict[str, str] | None = None) -> str:
    """Run the command and return what it printed."""
    return subprocess.run([str(cli), *args], capture_output=True, text=True, check=True, env=env).stdout


# 1. List calls (a table without --json).
print(hone_models("calls", "list", "--db", str(db)))
calls = json.loads(hone_models("calls", "list", "--db", str(db), "--json"))
assert len(calls) == 4  # the decision appears as its chat call

# 2. Show one span in full: here the failed call.
shown = json.loads(hone_models("calls", "show", failed.span_id, "--db", str(db)))
print("show:", shown["span_id"], shown["status"])
assert shown["status"]["code"] == "error"

# 3. Stats per model.
stats = json.loads(hone_models("calls", "stats", "--by", "model", "--db", str(db), "--json"))
for row in stats:
    tokens = f"{row['input_tokens']}/{row['output_tokens']}"
    print(f"stats: {row['key']:<13} calls={row['calls']} errors={row['errors']} tokens in/out={tokens}")
assert {row["key"]: row["errors"] for row in stats} == {"gemma4-12b": 1, "qwen2.5vl-7b": 0}

# The same queries from Python.
recent = find_calls(mk.records.read_spans(db), since="1h", model="gemma4-12b")
assert call_stats(recent, by="model")[0]["calls"] == 3

# 4. Models: list the registry, then smoke-test one (writes its speed to the user registry, so run it
# with a throwaway HOME here).
models = json.loads(hone_models("models", "list", "--json"))
print("models:", [m["id"] for m in models])
home = Path(tempfile.mkdtemp())
check = json.loads(
    hone_models("models", "check", "gemma4-12b", "--json", env={**os.environ, "HOME": str(home)})
)
print("check:", check)
assert check["text"] == "echo: Say OK."  # the model answered
assert check["saved_to"] == str(home / ".config" / "hone" / "models.toml")

server.stop()
