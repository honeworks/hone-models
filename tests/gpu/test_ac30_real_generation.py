"""AC-30 [real]: a 512x512 `z-image-turbo` image and 10 s of `ace-step-1.5-turbo` on the real ComfyUI, and
a Kokoro sentence transcribed by faster-whisper large-v3-turbo.

Run: scripts/gpu-lock.sh uv run pytest -m gpu -k ac30
ComfyUI: a running server at `HONE_COMFYUI_URL` (default `http://127.0.0.1:8188`) is used and only freed;
without one, `HONE_COMFYUI_START` (e.g. `$HOME/ai-stack/start_comfyui.sh`) starts it for the test, which
stops it at the end. The models are found under `HONE_COMFYUI_DIR` (default `~/ComfyUI`).
Whisper: `HONE_TEST_WHISPER_MODEL` (a Hugging Face repo id or a local folder; default
`deepdml/faster-whisper-large-v3-turbo-ct2`, about 1.6 GB); skipped when it is not in the Hugging Face
cache, so the test never starts a download.
"""

import importlib.util
import os
import re
from pathlib import Path

import pytest
from comfyui_real import URL, gpu_used_mb, use_real_comfyui

import hone_models as mk
from hone_models.providers import _comfyui_server
from whisper_fakes import registry

pytestmark = pytest.mark.gpu
# Read at import, before the autouse fixture points HOME at a temporary folder: the real download cache.
HF_HOME = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
MODEL = os.environ.get("HONE_TEST_WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
SENTENCE = "The lighthouse keeper counted seven ships before the morning fog rolled in."
LYRICS = "[verse]\nNeon on the water\nWe run until the morning\n"
MARGIN_MB = 512  # a server that was already running may keep a little more CUDA context


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def in_order(wanted: list[str], heard: list[str]) -> bool:
    """Whether `wanted` is a subsequence of `heard`."""
    rest = iter(heard)
    return all(word in rest for word in wanted)


def cached(model: str) -> bool:
    if Path(model).is_dir():
        return True
    return (Path(HF_HOME) / "hub" / f"models--{model.replace('/', '--')}").is_dir()


@pytest.mark.comfyui
def test_ac30_image_and_song_on_the_real_comfyui(
    gpu_lock, real_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    running_before = use_real_comfyui(monkeypatch)
    reg = mk.registry.load()
    for model in ("z-image-turbo", "ace-step-1.5-turbo"):
        if mk.catalog.installed(reg.get(model)) == "no":
            pytest.skip(f"{model} is not installed: hone-models models install {model}")
    sink = mk.records.MemorySink()
    before = gpu_used_mb()

    with mk.session("comfyui"):
        assert _comfyui_server.healthy(URL)
        shot = mk.image("z-image-turbo", registry=reg, sink=sink).generate(
            "a lighthouse at dusk, oil painting", size="512x512", seed=7, out=real_out / "shot.png"
        )
        take = mk.music("ace-step-1.5-turbo", registry=reg, sink=sink).generate(
            "calm lo-fi, soft piano, female vocals",
            lyrics=LYRICS,
            duration_s=10,
            seed=3,
            out=real_out / "take",
        )

    after = gpu_used_mb()
    assert (shot.error, take.error) == (None, None), (shot.error, take.error)
    image, song = shot.files[0], take.files[0]
    assert (image.mime, image.width, image.height) == ("image/png", 512, 512)
    assert image.bytes > 10_000
    assert song.mime == "audio/flac"
    assert song.duration_s is not None
    assert abs(song.duration_s - 10) < 0.5
    assert song.bytes > 50_000
    for span in sink.spans:
        attrs = span["attributes"]
        assert span["status"]["code"] == "ok"
        assert attrs["hone.models.media.freed"] is True  # a plain call frees ComfyUI at its end
        assert attrs["hone.models.media.workflow_sha256"]
    # ComfyUI was started only if the test started it, and then stopped; a running one is left running.
    assert _comfyui_server.healthy(URL) == running_before
    if before is not None and after is not None:
        assert after - before < MARGIN_MB, (before, after)


@pytest.mark.skipif(importlib.util.find_spec("kokoro") is None, reason="needs the `speech` extra")
@pytest.mark.skipif(importlib.util.find_spec("faster_whisper") is None, reason="needs the `transcribe` extra")
def test_ac30_kokoro_sentence_transcribed_in_order(
    gpu_lock, real_out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not cached(MODEL):
        pytest.skip(f"{MODEL} is not in the Hugging Face cache ({HF_HOME}); download it first")
    monkeypatch.setenv("HF_HOME", HF_HOME)
    spoken = mk.speech("kokoro-82m", sink=mk.records.NullSink()).synthesize(
        SENTENCE, voice="am_michael", out=real_out / "sentence.wav"
    )
    reg = registry()
    cfg = reg.get("test-whisper").model_copy(update={"model": MODEL})
    sink = mk.records.MemorySink()
    stt = mk.Transcriber(cfg, sink)
    before = gpu_used_mb()

    t = stt.transcribe(spoken.path, language="en", timeout_s=300)

    heard = [w for word in t.words for w in words_of(word.text)]
    assert in_order(words_of(SENTENCE), heard), (SENTENCE, t.text)
    assert all(w.end_s >= w.start_s for w in t.words)
    assert [w.start_s for w in t.words] == sorted(w.start_s for w in t.words)
    assert t.duration_s is not None
    assert abs(t.duration_s - spoken.duration_s) < 0.5
    assert (t.language, sink.spans[0]["status"]["code"]) == ("en", "ok")
    after = gpu_used_mb()
    if before is not None and after is not None:
        # The weights (about 1.6 GB in float16) were unloaded after the call; this process keeps only its
        # CUDA context and CTranslate2's small allocator cache.
        assert after - before < 1024, (before, after)
