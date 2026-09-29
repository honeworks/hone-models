# Transcription

`mk.transcriber(model_id)` turns an audio file into text with a time for every word, with a local
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) model (CTranslate2, MIT; no torch). Uses:
timing known lyrics against a sung take, checking that a take is intelligible, captions.

```bash
pip install "hone-models[transcribe]"
# on an NVIDIA GPU, CTranslate2 also needs the CUDA 12 cuBLAS and cuDNN 9 libraries:
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

The model is a registry entry with `provider = "faster_whisper"` and `kind = "transcription"`; `model` is
a Hugging Face repo id with a CTranslate2 conversion (downloaded into the Hugging Face cache on first use)
or a local folder:

```toml
[models."faster-whisper-large-v3-turbo"]
provider = "faster_whisper"
kind = "transcription"
model = "deepdml/faster-whisper-large-v3-turbo-ct2"
defaults = { compute_type = "float16", beam_size = 5 }   # also: device, vad_filter
[models."faster-whisper-large-v3-turbo".capabilities]
word_timestamps = true
vram_gb = 2.0
license = "MIT"
commercial_use = true
```

```python no-run
import hone_models as mk

stt = mk.transcriber("faster-whisper-large-v3-turbo")
t = stt.transcribe("takes/take_1.flac", language="en", prompt=lyrics)  # prompt biases toward the lyrics
print(t.text, t.language, t.duration_s)
for word in t.words:
    print(word.text, word.start_s, word.end_s, word.probability)
```

- **Result:** a `Transcript(text, language, duration_s, segments, model, span_id, words)`. `segments` are
  `TranscriptSegment(text, start_s, end_s, words)` in order; `words` holds every segment's words as
  `Word(text, start_s, end_s, probability)`, times in seconds from the start of the file. `language` is
  the one you passed, else the one the model detected; `duration_s` is `None` when unknown.
- **Arguments:** `language` (an ISO code; `None` detects it), `prompt` (text the model is biased toward:
  the known lyrics, names), `words=False` (no word timestamps, a little faster), `timeout_s` (default:
  the entry's `max_timeout_s`, 600 s), `trace`.
- **Checks before any work:** a missing audio file or a `timeout_s` of 0 or less raise `ConfigError`;
  `words=True` for an entry with `capabilities.word_timestamps = false` raises `CapabilityError`; a
  missing extra, or on CUDA a missing cuBLAS 12 / cuDNN 9 library, raise `ConfigError` naming what to
  install, before the GPU lease.
- **Timeout:** decoding stops between segments once `timeout_s` has passed, and `ModelTimeout` is raised;
  the model is freed as after any call.
- **Aligning known lyrics** to the heard words is the caller's job; it is not a model call.

## Defaults and the CUDA libraries

`defaults.device` is `auto` (CUDA when CTranslate2 sees a GPU, else the CPU), `cuda` or `cpu`;
`compute_type` defaults to `float16` on CUDA and `int8` on the CPU; `beam_size` to 5; `vad_filter`
(skip silence) to false.

On CUDA, `transcribe` first looks for `libcublas.so.12` and `libcudnn.so.9` on the system library path
(`LD_LIBRARY_PATH`, the loader cache), then in the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels
of the running Python, and loads them for the process. A torch built for CUDA 12 brings the same wheels,
so an environment with the `speech` extra usually has them already. A machine with only CUDA 13 needs the
wheels, or `defaults.device = "cpu"`.

## GPU and sessions

A call takes a GPU lease of the entry's `capabilities.vram_gb` (1 GB when not declared), loads the model,
transcribes, and frees the model (the CTranslate2 model is unloaded), as a speech call does. For many
files, hold one lease and one loaded model with a session:

```python no-run
with mk.transcriber("faster-whisper-large-v3-turbo").session() as stt:
    heard = [stt.transcribe(take, language="en") for take in takes]
```

The model loads on the first call and is freed once at the end of the block, also on an exception.
Sessions of one transcriber do not nest.

## Records

Each call records one `hone.models.transcribe` span (`gen_ai.operation.name = "transcription"`,
`gen_ai.provider.name = "faster_whisper"`) with `hone.models.transcribe.audio` (`{"path", "sha256",
"bytes"}`), `language`, `duration_s`, `words_count`, `session`, `loaded`, and the content attributes
`text`, `words` (`[{"text", "start_s", "end_s", "probability"}]`) and `prompt` (when given). With content
capture off these three are stored as hashes and lengths; a word list over 64 KiB goes to the `blobs`
table of the SQLite store. No audio bytes are recorded.

## Testing

`hone_models.testing.FakeTranscriber` stands in for the client: same checks, span and result, no model and
no lease. It "hears" its `text` in every file (one segment per line, words 0.4 s apart) and keeps `calls`;
`loads` and `frees` count model loads as the real client would do them. `FakeTranscriber.like(model_id)`
copies a registry entry's id and capabilities.

```python
import tempfile
from pathlib import Path

from hone_models.testing import FakeTranscriber

take = Path(tempfile.mkdtemp()) / "take.wav"
take.write_bytes(b"RIFF")  # the fake never reads the audio; the file must exist
stt = FakeTranscriber(text="Hold the line\nwe are almost home")
t = stt.transcribe(take, language="en", prompt="Hold the line")
print(t.text)  # Hold the line we are almost home
print(t.words[3])  # Word(text='we', start_s=1.2, end_s=1.6, probability=1.0)
assert [s.text for s in t.segments] == ["Hold the line", "we are almost home"]
assert stt.calls == [(take, "en", "Hold the line", True)]
```
