---
name: debug-failure
description: Find the cause of a bug, a failing test or a wrong result before changing code - reproduce, read the evidence, test one hypothesis at a time, fix the cause, keep a regression test. Use for any bug report, test failure or unexpected behaviour.
---

# Debug a failure

1. **Reproduce** with the smallest command: one test (`uv run pytest <path>::<test> -x -vv`) or a
   short script.
2. **Read the evidence before guessing:**
   - the full traceback and the error message;
   - the recorded calls: `hone-models calls list`, `hone-models calls show <id>`, `hone-models calls stats`
     (`docs/cli.md`), stored in `.hone/models/spans.db` (`HONE_HOME`);
   - replay the failing call from its recording (`docs/records-and-replay.md`) instead of calling the
     model again;
   - the result's `error` before any exception;
   - `git log -p` of the code involved.
3. **Write a failing test** that reproduces the bug before fixing it.
4. **One hypothesis at a time.** Say what you expect to see if it's true, check it, change one thing.
5. **Fix the cause**, not the symptom. Never loosen a test, catch and ignore an error, or add a retry to
   hide it.
6. If the cause is a design gap, note it in `design/decisions.md` or start a `plan-change`.
7. Commit as `fix: ...` with the cause and the fix in the body.

Adapted from the `systematic-debugging` skill of
[obra/superpowers](https://github.com/obra/superpowers) (MIT).
