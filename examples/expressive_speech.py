"""Expressive speech: narrate a lesson paragraph by paragraph, each with its own emotion and intensity,
in one session that loads the model once.

What: a short lesson script whose paragraphs are labelled with an intent (curious, serious,
    encouraging), synthesized one paragraph per call with `emotion=` and `intensity=` inside
    `with tts.session():`, then read back from the recorded spans.

How: `tts = mk.speech("chatterbox")` (the `expressive` extra; about 3 GB of weights on first use);
    `tts.synthesize(paragraph, voice="warm_female", emotion="encouraging", intensity=0.6, out=...)`.
    `emotion` is one of `tts.emotions`, `intensity` is 0.0 (flat) to 1.0 (theatrical) or
    `None` for the emotion's default. `tts.expressive` says whether the model applies them: Chatterbox
    does, Kokoro (`kokoro-82m`) accepts and ignores them. A paragraph (up to `tts.max_chunk_chars`) is
    synthesized in one piece so the melody flows across its sentences; blank lines separate paragraphs.
    `with tts.session():` loads the model on the first call, keeps it (and the GPU lease) for every call
    in the block and frees it once at the end; without it each call loads and frees the model.

Why: a teacher does not read every line the same way. Giving each paragraph its intent makes narration
    sound spoken, not read, and the span records which emotion and intensity produced each clip, so a
    flat or over-acted clip can be traced back and redone. Pitfalls: an unknown emotion or an intensity
    outside 0..1 raises `ConfigError` on every model, so a typo fails in tests; keep intensity around
    0.3 to 0.6 for teaching and save 0.8+ for moments. Chatterbox takes several seconds to load, so
    narrating many lines without a session is slow; but a session holds the GPU lease, so keep it to
    one batch of narration.

Runs offline: `FakeSpeech(expressive=True)` has the same API and writes silence of about `words / 2.5`
    seconds. To use the real model, replace it by `mk.speech("chatterbox", sink=sink)`.
"""

import hone_models as mk
from hone_models.testing import FakeSpeech

sink = mk.records.MemorySink()
tts = FakeSpeech(sink=sink, expressive=True)  # real: mk.speech("chatterbox", sink=sink)
print("emotions:", ", ".join(tts.emotions))

lesson = [
    ("curious", 0.5, "Think about the last time someone criticized your work. What did you feel first?"),
    ("serious", 0.3, "That jolt is normal. What matters is what you say in the next ten seconds."),
    ("encouraging", 0.6, "And here is the good news: that pause is a skill. You can practise it this week."),
]
clips = []
with tts.session():  # one model load for the whole lesson, freed at the end of the block
    for i, (emotion, intensity, paragraph) in enumerate(lesson):
        clip = tts.synthesize(
            paragraph, voice="af_heart", emotion=emotion, intensity=intensity, out=f"out/lesson_{i}.wav"
        )
        clips.append(clip)
        print(f"{emotion:>12} {intensity:.1f}: {clip.duration_s:.1f} s -> {clip.path}")
print(f"model loads: {tts.loads}, frees: {tts.frees}")
assert (tts.loads, tts.frees) == (1, 1)

for clip, span, (emotion, intensity, _) in zip(clips, sink.spans, lesson, strict=True):
    attrs = span["attributes"]
    assert span["span_id"] == clip.span_id
    assert (attrs["hone.models.speech.emotion"], attrs["hone.models.speech.intensity"]) == (
        emotion,
        intensity,
    )
    assert attrs["hone.models.speech.expressive"] is True
    assert attrs["hone.models.speech.chunks"] == 1  # each paragraph in one piece
    assert attrs["hone.models.speech.session"] is True

try:
    tts.synthesize("Hello.", emotion="angry", out="out/x.wav")
except mk.errors.ConfigError as exc:
    print("refused:", exc)
else:
    raise AssertionError("an unknown emotion must raise")
