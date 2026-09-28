"""Speech: text to a WAV file with a local voice model, long narration included.

What: narrate a short line and a minutes-long script into WAV files, pick voices and speed, get sentence
    timings, and read the recorded `hone.models.speech` span.

How: `tts = mk.speech("kokoro-82m")` (the `speech` extra; the weights download on first use);
    `r = tts.synthesize(text, voice=None, speed=1.0, out="line.wav")` writes a 16-bit mono WAV and returns
    `SpeechResult(path, duration_s, sample_rate, voice, model, span_id, segments)`; `segments` says where
    each synthesized chunk starts and ends, and `by_sentence=True` makes each sentence its own chunk.
    `voice=None` takes the first of `tts.voices`; voice names start with the accent and gender (`af_`
    American female, `am_` American male, `bf_` / `bm_` British). Long text is split into sentences,
    synthesized in chunks of at most 400 characters and joined with a short pause, so one call can
    narrate a whole script.

Why: narration for videos and courses should run on the local GPU next to the LLM, take part in GPU
    scheduling (each call holds `mk.gpu.lease`, and the model is freed afterwards) and show up in the call
    records like any other model call. The text is content: with content capture off, only its hash and
    length are stored. Pitfall: an unknown voice or a speed outside 0.5..2.0 raises `ConfigError` before
    any model loads.

Runs offline: `FakeSpeech` has the same API and writes silence of about `words / 2.5` seconds. To use
    the real model, replace `FakeSpeech(sink=sink)` by `mk.speech("kokoro-82m", sink=sink)`.
"""

import hone_models as mk
from hone_models.testing import FakeSpeech

sink = mk.records.MemorySink()
tts = FakeSpeech(sink=sink)  # real: mk.speech("kokoro-82m", sink=sink)
print("voices:", tts.voices)

line = tts.synthesize("Welcome to the course. Today we look at attention.", out="out/line.wav")
print(f"{line.path}: {line.duration_s:.1f} s at {line.sample_rate} Hz, voice {line.voice}")
assert line.path.is_file()
assert line.duration_s > 0
assert line.voice == tts.voices[0]  # voice=None takes the default

script = " ".join(f"Step {i}: the model compares every word with every other word." for i in range(1, 60))
long = tts.synthesize(script, voice="am_michael", speed=1.1, out="out/narration.wav")
span = sink.spans[-1]
chunks = span["attributes"]["hone.models.speech.chunks"]
print(f"narration: {long.duration_s / 60:.1f} min from {len(script)} characters in {chunks} chunks")
assert chunks > 1
assert long.duration_s > 60  # minutes of speech from one call
assert span["name"] == "hone.models.speech"
assert span["span_id"] == long.span_id

# Sentence timings for captions and cuts: one call, one segment per sentence.
captioned = tts.synthesize(
    "Attention compares words. Each word looks at the others.", out="out/c.wav", by_sentence=True
)
for seg in captioned.segments:
    print(f"  {seg.start_s:5.2f}-{seg.end_s:5.2f} s  {seg.text}")
assert [seg.text for seg in captioned.segments] == [
    "Attention compares words.",
    "Each word looks at the others.",
]
assert captioned.segments[-1].end_s == captioned.duration_s

try:
    tts.synthesize("Hello.", voice="nobody", out="out/x.wav")
except mk.errors.ConfigError as exc:
    print("refused:", exc)
else:
    raise AssertionError("an unknown voice must raise")
