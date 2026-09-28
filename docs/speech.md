# Speech

`mk.speech(model_id)` turns text into a WAV file with a local voice model. Two models are packaged:

| Model id | Model | Extra | Expressive | VRAM | Voices |
|---|---|---|---|---|---|
| `kokoro-82m` | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0 | `speech` | no | about 1 GB | `af_heart` (default), `am_michael`, `af_bella`, `am_fenrir`, `bf_emma`, `bm_george` |
| `chatterbox` | [Chatterbox](https://huggingface.co/ResembleAI/chatterbox), MIT | `expressive` | yes | about 5 GB | `warm_female` (default), `warm_male`, `british_female`, `british_male`, `chatterbox_default` |

Kokoro is small and fast with an even, reading delivery. Chatterbox is larger and slower but acts: pass an
`emotion` and an `intensity` and it sounds encouraging, curious or excited (see
[Emotion and intensity](#emotion-and-intensity)).

```bash
pip install "hone-models[speech]"     # Python < 3.13; pulls torch and the kokoro package
# and spaCy's English model, which Kokoro's English voices need (PyPI packages cannot depend on it):
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

In a uv project, add it to your dependencies instead, so the lock file holds it (a uv environment has no
`pip` for Kokoro's phonemizer to install it with at run time):

```toml
dependencies = [
    "hone-models[speech]",
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
]
[tool.hatch.metadata]
allow-direct-references = true   # hatchling only, for the direct URL above
```

Without it, `synthesize` with an English voice (`a...`, `b...`) raises `ConfigError` with this line before
it takes the GPU lease.

```python no-run
import hone_models as mk

tts = mk.speech("kokoro-82m")
r = tts.synthesize("Attention lets every word look at every other word.", voice="am_michael", out="line.wav")
print(r.path, r.duration_s, r.sample_rate, r.voice, r.model, r.span_id)
```

- **Output:** a 16-bit mono WAV at `out` (parent folders are created), 24 kHz for Kokoro. The result is a
  `SpeechResult(path, duration_s, sample_rate, voice, model, span_id, segments)`.
- **Timings:** `segments` lists each synthesized chunk as `SpeechSegment(text, start_s, end_s)`: where it
  sits in the WAV, from the join points. A chunk is one or more whole sentences (an over-long sentence is
  cut at spaces). `synthesize(..., by_sentence=True)` makes every sentence its own chunk, so `segments`
  gives sentence timings for captions and picture cuts from one call (at the cost of prosody across
  sentences).
- **Voices:** `tts.voices` lists them; `voice=None` takes the first. The first letter is the accent
  (`a` American, `b` British English), the second the gender (`f`, `m`): `af_heart` (default),
  `af_bella`, `am_michael`, `am_fenrir`, `bf_emma`, `bm_george`. Kokoro has more voices; add them to
  `capabilities.voices` in your registry file to allow them. An unknown voice raises `ConfigError`.
- **Speed:** `speed=1.0` is normal; 0.5 to 2.0 is allowed.
- **Paragraphs and long text:** the text is split into paragraphs (blank lines). A paragraph that fits
  the model's limit (`tts.max_chunk_chars`, 400 characters for both models) is synthesized in one piece,
  so its prosody flows across sentences; a longer one is split on sentences into chunks of about equal
  length. Chunks are joined with a 0.25 s pause, paragraphs with 0.6 s, so one call can narrate many
  minutes. Write narration in paragraphs of a few sentences, one thought each.
- **First use** downloads the weights (about 330 MB) and the voice files from Hugging Face into its
  cache (`HF_HOME`, default `~/.cache/huggingface`); later calls work offline.
- **GPU:** each call holds `mk.gpu.lease("kokoro-82m", 1.0)`, loads the model and frees it afterwards
  (the CUDA cache is emptied), so it never keeps memory on a shared GPU. For many calls in a row, use a
  [session](#sessions-many-calls-one-load). Without CUDA it runs on the CPU. Set
  `defaults = { device = "cpu" }` in the registry entry to force the CPU.
- **Records:** one `hone.models.speech` span per call with the voice, speed, character, chunk and
  paragraph counts, duration and sample rate, `expressive`, `session` and `loaded` (whether this call
  loaded the model), and the `emotion` and `intensity` when given. The text (`hone.models.speech.input`) is content: with content capture off
  only its hash and length are stored.

## Sessions: many calls, one load

A plain `synthesize` call loads the model, speaks and frees it: about a second for Kokoro, several
seconds for Chatterbox (3 GB). To narrate many lines, open a session: the model is loaded on the first
call, kept for every call in the block, and freed once at the end; the GPU lease is held for the whole
block.

```python
from hone_models.testing import FakeSpeech

tts = FakeSpeech()  # real: mk.speech("chatterbox") or mk.speech("kokoro-82m")
lines = ["Welcome back.", "Today we practise the pause.", "Ready? Let's go."]
with tts.session():
    clips = [tts.synthesize(line, emotion="warm", out=f"line_{i}.wav") for i, line in enumerate(lines)]
assert (tts.loads, tts.frees) == (1, 1)  # one load for three calls, freed at the end of the block
```

- Everything else is unchanged inside a session: each call still records its own span (with
  `hone.models.speech.session = true`; `loaded` is true only for the call that loaded the model), and
  each Chatterbox call is seeded the same way, so a line sounds the same in or out of a session.
- The lease is taken when the block starts, so other GPU work waits for the block to end: keep sessions
  to one batch of narration, not a whole program run.
- Inside your own GPU lease in the same thread (for example a hone-flow `gpu:tts` step with `vram_gb`),
  the client's lease is nested: it borrows what the outer lease reserved and adds only the difference,
  so a 5 GB step lease around Chatterbox's 5 GB costs 5 GB, not 10. Give the step the model's size (or
  0); if the two still cannot fit, the lease fails with `CapabilityError` naming the outer lease instead
  of waiting ([GPU leases](gpu-and-sessions.md)).
- The model is freed when the block ends, also when it ends with an exception. Sessions do not nest
  (`ConfigError`); a session with no calls never loads the model.

## Emotion and intensity

```python no-run
tts = mk.speech("chatterbox")  # extra `expressive`; about 3 GB downloaded on first use
tts.synthesize(
    "You did it! That was the hardest part of the course.",
    voice="warm_male",
    emotion="encouraging",
    intensity=0.7,
    out="win.wav",
)
```

- `emotion` is one of `tts.emotions`: `neutral`, `calm`, `warm`, `serious`, `curious`,
  `encouraging`, `playful`, `excited`. `intensity` runs from 0.0 (flat) to 1.0 (theatrical); `None`
  takes the emotion's own default (calm 0.1, neutral 0.25, warm 0.35, encouraging 0.5, excited 0.75).
  Anything else raises `ConfigError`, on every model.
- `tts.expressive` says whether the model applies them. **Chatterbox** does: intensity sets its
  `exaggeration` (0.25 + intensity) and the emotion its `cfg_weight` (pacing: lower for livelier
  emotions). **Kokoro** accepts and ignores them (they are still recorded, with `expressive = false`).
- Emotion is per call, so give each paragraph its own call when the intent changes, and one call for
  consecutive paragraphs with the same intent. Around 0.3 to 0.6 sounds natural for teaching; 0.8 and
  above is for moments, not whole lessons.
- **Voices:** Chatterbox copies the voice of a short reference clip. The packaged clips were rendered
  once by Kokoro's synthetic voices (`scripts/make-expressive-voices.py`): `warm_female` (from
  `af_heart`), `warm_male` (`am_michael`), `british_female` (`bf_emma`), `british_male` (`bm_george`);
  `chatterbox_default` is the voice that ships with the Chatterbox weights. No real person's voice is
  used, and `voice` takes only these names.
- **Speed:** Chatterbox has no speed control; `speed != 1.0` time-stretches the audio (fine within about
  ±15 %).
- **Cost:** each call loads the model (about 3 GB from the disk cache, several seconds) inside
  `mk.gpu.lease("chatterbox", 5.0)` and frees it afterwards; a [session](#sessions-many-calls-one-load)
  loads it once for many calls. Synthesis runs at roughly real time on an
  8 GB GPU. Its output carries Resemble AI's inaudible Perth watermark.

### Installing the `expressive` extra

`chatterbox-tts` pins `torch==2.6.0`, `torchaudio==2.6.0` and `numpy<2` exactly, but runs on newer
builds. Install it with uv and override those pins in **your** project (overrides apply only in the root
project, so hone-models' own settings do not carry over):

```toml
# pyproject.toml of the application
dependencies = ["hone-models[expressive]"]

[tool.uv]
override-dependencies = ["torch>=2.6,<2.11", "torchaudio>=2.6,<2.11", "numpy>=1.24"]
```

Python 3.11 or 3.12. No system packages are needed. (To develop against a local checkout of
hone-models instead of the published package, add `[tool.uv.sources] hone-models = { path =
"path/to/hone-models", editable = true }`.)

## Testing without the model

`hone_models.testing.FakeSpeech` has the same API, sessions included, needs no extra and no GPU, and
writes silence of about `words / 2.5` seconds (divided by `speed`). It records the same span (to a
`MemorySink` unless you pass `sink=`), every call in `tts.calls`, and counts model loads and frees in
`tts.loads` / `tts.frees`:

```python
from hone_models.testing import FakeSpeech

tts = FakeSpeech()
clip = tts.synthesize("Hello there, and welcome to the course.", voice="bf_emma", out="hello.wav")
assert round(clip.duration_s, 1) == 2.8  # 7 words / 2.5
assert tts.calls == [("Hello there, and welcome to the course.", "bf_emma", 1.0, None, None)]

expressive = FakeSpeech(expressive=True)  # an expressive fake; accepts the Chatterbox voices too
expressive.synthesize("Great question.", voice="warm_male", emotion="curious", intensity=0.5, out="q.wav")
assert expressive.calls[0][3:] == ("curious", 0.5)
assert expressive.sink.spans[0]["attributes"]["hone.models.speech.emotion"] == "curious"

actor = FakeSpeech.like("chatterbox")  # the id, voices, expressiveness and chunk size of a registry entry
assert actor.model_id == "chatterbox" and actor.voices[0] == "warm_female" and actor.expressive
```

`FakeSpeech.like(model_id)` copies any speech entry of the registry (packaged or your own), so a test
uses the real model's voice names and cannot drift from them. `FakeSpeech(expressive=True)` keeps
Kokoro's voices (and default) and adds Chatterbox's.

Start an application on `FakeSpeech` and switch to `mk.speech("kokoro-82m")` or `mk.speech("chatterbox")`
when the extra is installed.

**Runnable examples:** [speech.py](../examples/speech.py),
[expressive_speech.py](../examples/expressive_speech.py).
