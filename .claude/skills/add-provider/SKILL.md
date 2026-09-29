---
name: add-provider
description: Add a provider to hone-models - a new model backend - with its extra, results and errors, records, tests and docs. Use when adding support for a new model server, API or kind of model.
---

# Add a provider

A provider connects `hone_models` to a model backend (Ollama, OpenAI-compatible, LiteLLM, speech); the
registry maps model ids to providers (`design/current.md` §3, §4). A new capability or a change to the
public API is a design change: `plan-change` first.

1. **Where.** Providers are modules in `src/hone_models/providers/`; look at an existing one first.
2. **Optional dependency.** Heavy or rare libraries go behind an extra in `pyproject.toml` and are
   imported only inside the provider. The core must still import without them:
   `tests/unit/test_import_boundaries.py`.
3. **Results and errors** (`design/current.md` §2.2): problems with an answer go on the result's
   `error`; problems with the call raise a `ProviderError` or `CapabilityError` whose message says what to
   do. Retries and timeouts as the provider section says.
4. **Records** (`design/current.md` §8): every call writes its span; secrets never appear; recorded
   calls replay.
5. **Tests with recorded HTTP or the fake Ollama** in the default suite; real calls only in
   `tests/gpu/` (`real-model-tests`).
6. **Entry points** when other honeworks packages find it by name (`hone.decision_clients`,
   `hone.text_clients`, `hone.embedders` in `pyproject.toml`).
7. `sync-docs`: the install line, `docs/registry.md`, an example if it's a new kind of backend.
