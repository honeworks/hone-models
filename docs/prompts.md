# Prompts with sections

`mk.Prompt` builds a prompt from named, versioned sections, so records can tell which part of a prompt
led to which output, and replays can change one part.

```python
import hone_models as mk

prompt = mk.Prompt(
    "song_ideas",
    "3",
    sections={
        "system": "You write song pitches.",
        "rules": "Exactly 12 ideas, JSONL.",
        "format_example": mk.Section('{"title": "Cargo Hold Chaos"}', version="7"),
        "brief": "Theme: {theme}",
    },
    variables={"theme": "sea shanties"},
)
print(prompt.messages())
for s in prompt.section_records():
    print(s["id"], s["version"], prompt.text[s["start"] : s["end"]])
r = mk.text("gemma4-12b").complete(prompt)
```

- Sections named in `system_sections` (default `("system",)`) form the system message; the others form
  the user message, each joined by a blank line, in the given order.
- `{name}` is replaced for names in `variables`; other braces (JSON examples) are left alone.
- A section's version defaults to the prompt's version.
- The span records `hone.models.prompt.template_id`, `.template_version`, `.variables` and `.sections`
  (id, version, role, `[start, end)` in `prompt.text`, sha256).

**Runnable examples:** [prompt_sections_and_replay.py](../examples/prompt_sections_and_replay.py).
