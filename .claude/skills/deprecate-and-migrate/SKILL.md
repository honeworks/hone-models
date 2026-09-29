---
name: deprecate-and-migrate
description: Change or remove public API, CLI flags or stored formats without breaking users - deprecation warnings, a compatibility period, migrations. Use when a change renames, removes or changes the meaning of anything users or their saved data depend on.
---

# Deprecate and migrate

Users and their saved data outlive any version. Break nothing silently.

**Public API and CLI**
1. Keep the old name working for at least one minor release: it calls the new one and warns with
   `warnings.warn("<old> is deprecated, use <new>; it will be removed in <version>", DeprecationWarning, stacklevel=2)`.
   CLI: the same message on stderr.
2. Docs and examples show only the new form. `CHANGELOG.md`: **Deprecated** now, **Removed** when it goes.
3. Tests: the old form still works and warns; the new form works.

**The call store and recordings** (`design/current.md` §8.2, §8.6, `docs/records-and-replay.md`)
1. New span attributes are optional. Renaming or removing one, or changing the SQLite store's schema,
   is a change record with a "Migration and compatibility" section.
2. Calls recorded by older versions still replay, or replay refuses them with a `HoneModelsError` that
   says what to do. Never guess.
3. Keep a small recording from the old version in `tests/fixtures/` and test replaying it and the
   refusal.
