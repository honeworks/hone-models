---
name: plan-change
description: Plan a design change as a change record in design/changes/ and get it approved before writing code. Use after start-task when a change touches the public API, CLI, guarantees, records, ports, extras or dependencies.
---

# Plan a design change

1. Read `design/current.md` (§1 and the sections the change touches, and the acceptance cases in
   `design/current.md` §11), `design/decisions.md` and the related records in `design/changes/`.
2. Next number: the highest `design/changes/NNNN-*.md` plus one.
3. Write `design/changes/NNNN-<short-name>.md` in the format of the existing records: Status `proposed`,
   Context, Problem, Options (at least two, one of them "leave it as it is"), Decision, Consequences,
   Migration and compatibility. Include:
   - a sketch of the public API or CLI as a user would call it;
   - the new or changed acceptance cases, numbered after the last one;
   - the tests that will prove them.
4. Answer each risk the change touches, in the record:
   - problems with an answer stay on the result's `error`, problems with a call stay typed exceptions
     (`design/current.md` §2.2); `None`, never `0`, for a missing value;
   - GPU scheduling (`design/current.md` §7): a lease is always released, nested leases still work, one
     GPU job at a time;
   - records and replay (`design/current.md` §8.4, §8.6): secrets never recorded, the content-capture
     switch respected, calls recorded by older versions still replay;
   - no silent fallback to a hosted service;
   - the core imports no extra; a new dependency is an optional extra with a reason in
     `design/decisions.md`;
   - secrets are never recorded;
   - is there a smaller option that meets the need? (CONTRIBUTING.md, "Keep it simple")
5. Commit: `docs(design): propose NNNN <short name>`.
6. **Stop and ask.** Give the user the decision and the open questions in a few lines, with the file
   path. Only the project's maintainer accepts a record. If the user is the maintainer and approves, set
   the status to `accepted (approved by the maintainer YYYY-MM-DD)`, commit, and continue with
   `implement-change`. Otherwise the status stays `proposed`: offer to open a pull request with the record
   alone, and open it (`open-pr`) only when the user says so (AGENTS.md "Git" rule); or link the record
   from the issue. The maintainer decides there. On rejection set `rejected` with the reason and stop.
