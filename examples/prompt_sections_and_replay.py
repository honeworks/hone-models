"""Prompts from named, versioned sections, recorded with character spans, then replayed with changes.

What: build a prompt from sections, see how it renders and how each section is recorded, then replay the
    recorded call without one section and with different params, to test whether that section matters.

How: `mk.Prompt(template_id, version, sections={id: text | mk.Section(text, version)}, variables=...)`.
    Sections listed in `system_sections` (default: "system") form the system message, the rest the user
    message, in the given order; `{name}` placeholders are filled from `variables` (other braces are left
    alone, so JSON examples are safe). `llm.complete(prompt)` records the template id, version, variables
    and each section's `[start, end)` span in `prompt.text`. `mk.replay.Replayer(sink).replay_call(span,
    overrides)` calls the model again; overrides are `"prompt.sections"` (`{id: new_text | None}`, `None`
    removes the section), `"params"` (merged), `"model"` and `"messages"`.

Why: when output goes wrong you want to know which part of the prompt caused it. Section versions say
    which wording produced a recorded result, and a replay changes one thing and keeps the rest.
    Pitfalls: replay rebuilds the request from the span, so it needs content capture on (records.py);
    replacement section texts are used as given (variables are not filled again).
"""

import tempfile
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
store = Path(tempfile.mkdtemp()) / "spans.db"
sink = mk.records.SqliteSpanSink(store)

prompt = mk.Prompt(
    "song_ideas",
    "3",  # the template version; sections without their own version inherit it
    sections={
        "system": "You write song pitches.",
        "rules": "Exactly 3 ideas, one per line.",
        "format_example": mk.Section("Cargo Hold Chaos - a shanty about lost luggage", version="7"),
        "brief": "Theme: {theme}",
    },
    variables={"theme": "sea shanties"},
)
print("messages:", [(m["role"], m["content"][:30]) for m in prompt.messages()])

# 1. Call with the prompt; the section spans are recorded with the call.
r = mk.text("gemma4-12b", sink=sink).complete(prompt, max_tokens=200)
span = next(s for s in mk.records.read_spans(store) if s["span_id"] == r.span_id)
for section in span["attributes"]["hone.models.prompt.sections"]:
    text = prompt.text[section["start"] : section["end"]]
    print(f"{section['id']:>15} v{section['version']} {section['role']:<6} {text!r}")
assert span["attributes"]["hone.models.prompt.template_version"] == "3"
assert prompt.text.endswith("Theme: sea shanties")

# 2. Replay without the format example, at temperature 0.
replayer = mk.replay.Replayer(sink=sink)
new = replayer.replay_call(
    span, {"prompt.sections": {"format_example": None}, "params": {"temperature": 0.0}}
)
sent = server.requests[-1]["body"]
print("replay of", new["attributes"]["hone.models.replay_of"], "-> new span", new["span_id"])
print("replayed user message:", repr(sent["messages"][-1]["content"]))
assert new["attributes"]["hone.models.replay_of"] == span["span_id"]
assert "Cargo Hold" not in str(sent["messages"])
assert sent["options"]["temperature"] == 0.0

# 3. Replay with a new wording of one section.
new = replayer.replay_call(span, {"prompt.sections": {"rules": "Exactly 5 ideas, one per line."}})
assert "Exactly 5 ideas" in str(server.requests[-1]["body"]["messages"])

server.stop()
