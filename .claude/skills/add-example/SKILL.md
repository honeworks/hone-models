---
name: add-example
description: Add a runnable, explained example to examples/ that the tests execute. Use when a change adds a concept users should see, or the user asks for an example.
---

# Add an example

1. One concept per file: `examples/<name>.py`, in the shape of `examples/decisions.py`:
   - a docstring with **What:**, **How:** and **Why:** paragraphs, in that order;
   - offline: the scriptable fake Ollama from `hone_models.testing`, no network, no GPU; only the public
     API; `assert`s the facts it shows;
2. List it in `examples/README.md`, in reading order.
3. The test finds it by itself; check that the README row links the file.
4. `uv run pytest tests/e2e/test_ac20_examples.py -q`.
