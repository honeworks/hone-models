# 0014: Nested leases of one process that together cannot fit

## Status

`implemented in 0.1.0` (options 2, 1 and 3)

## Context

Found in concept-shorts (a demo app built on the honeworks packages), and earlier the same way in explainer-channel (its commit
"fix: the TTS batch lease must not double-book the speech client's own lease"). A hone-flow step
declared `resources="gpu:tts", vram_gb=5.0`, so hone-flow took a 5120 MB lease `gpu:tts` for the batch.
Inside the step, `mk.speech("chatterbox").session()` asked the same scheduler for its own lease
(`chatterbox`, 5 GB from the registry). On the 8 GB card 5120 + 5120 > 8188 MB, so the inner lease
polled forever: the run sat in `narration: running` with the GPU at 0 % and the process sleeping, until
it was killed. The ledger showed only one entry, held by the waiting process itself.

This is a sharper case of [0004](0004-lease-that-can-never-be-granted.md): there the blocker was memory
outside the ledger; here the blocker is a ledger entry of the requesting process, in the same thread,
which by construction cannot be released while the inner lease waits.

## Problem

- Leases are reentrant only for the same `name`; a nested lease under another name from the same
  process is treated like a request from a stranger and waits for the outer one, which is a deadlock.
- The wait is silent (0004), so the symptom is a run that looks slow.
- Apps must know that the speech client leases by itself and set their batch lease to about 0; nothing
  tells them when they get it wrong.

## Options

1. **Detect it and fail fast**: in `_acquire`, if `free < need` and every entry that would have to go
   is held by `os.getpid()` (or by the same thread's `held` set), raise `CapabilityError` naming both
   leases: "lease 'chatterbox' (5120 MB) is nested inside this process's lease 'gpu:tts' (5120 MB);
   together they exceed the 8188 MB card: lower the outer vram_gb (the inner client leases by itself)".
2. **Let a nested lease borrow from the outer one**: an inner lease of the same process/thread counts
   the outer lease's reservation as available to it (the outer reservation is a claim for the whole
   block, which the inner model is part of); only the difference is added to the ledger.
3. **Document it** in `design/current.md` §GPU and in the speech docs: "inside a hone-flow GPU step, set
   the step's `vram_gb` to 0 when the client leases by itself".

## Recommendation

2 with 1 as the fallback when even borrowing cannot fit, plus 3. Borrowing matches what apps mean (the
step's lease is for the model the step uses) and makes both old and new apps correct; the error covers
the rest. Together with 0004's "fail fast / log while waiting" no GPU wait can be silent and endless.

## Implementation

Option 2 was built with [0004](0004-lease-that-can-never-be-granted.md): a lease inside other leases of
the same process and thread on the same ledger (any `GpuScheduler`) reserves only what they do not cover,
with a `nested_in` ledger entry for the difference. Option 1: a nested lease larger than the whole card
fails at once (as any lease does); one that could fit only if memory outside the ledger were freed fails
after `stall_s` with 0004's error, which then names the outer lease and says to lower its `vram_gb`. The
waiting process's own outer entries do not count as "another lease held". Option 3: docs/speech.md and
docs/gpu-and-sessions.md.
