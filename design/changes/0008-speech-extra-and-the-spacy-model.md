# 0008: The speech extra needs spaCy's English model

## Status

`implemented in 0.1.0` (options 2 and 3)

## Context

Found while building course-builder (a demo app built on the honeworks packages) on the `speech` extra (change 0006). Kokoro's
English phonemizer (`misaki.en.G2P`) loads the spaCy model `en_core_web_sm`; when it is missing, misaki
calls `spacy.cli.download`, which runs `python -m pip install ...`. A uv-created virtual environment has
no `pip`, so the first synthesis in such an environment fails (or, in an environment with pip,
silently installs a package at runtime, outside the lock file). course-builder's first Kokoro call worked
only after it added the wheel as a direct dependency:

```toml
"en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
```

(plus `[tool.hatch.metadata] allow-direct-references = true`). At the time of writing,
`hone-models/.venv` has neither `en_core_web_sm` nor `pip`, so `tests/gpu/test_ac21_real_speech.py` is
expected to hit this on a clean environment.

## Options

1. Add the wheel URL to the `speech` extra. PyPI refuses packages with direct URL dependencies, so the
   published extra could not carry it.
2. Document it: `docs/speech.md` gives the one line to add (`uv pip install <wheel url>` or the
   dependency above), and the Kokoro provider turns the download failure into a `ConfigError` that says
   exactly that.
3. Check at `mk.speech("kokoro-82m")` time (`importlib.util.find_spec("en_core_web_sm")`) and fail early
   with the same message, before a GPU lease is taken.

## Decision

Proposed: 2 and 3 (no runtime installs, a clear message before any GPU work).

## Implementation

The check runs at the start of `synthesize` (before the GPU lease), not in `mk.speech(...)`: apps and
tests build the client without the extras installed. It applies to English voices when misaki is
installed. The download failure itself is not caught: the check makes it unreachable. This repository
installs the wheel from an unpublished `[dependency-groups] dev` entry, so `tests/gpu` runs on a clean
`uv sync`.
