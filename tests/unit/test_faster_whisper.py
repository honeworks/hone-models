"""The `faster_whisper` provider against fake `faster_whisper` / `ctranslate2` modules: options, words,
timeouts, freeing, the missing extra and the CUDA library check."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any

import pytest

import hone_models as mk
from hone_models.errors import ConfigError, ModelTimeout, ProviderError
from hone_models.providers import faster_whisper
from hone_models.speech import write_wav
from media_fixtures import LeaseRecorder
from whisper_fakes import install, registry


@pytest.fixture
def take(tmp_path: Path) -> Path:
    path = tmp_path / "take.wav"
    write_wav(path, b"\x00\x00" * 1600, 16000)
    return path


def client(model_id: str = "test-whisper") -> mk.Transcriber:
    stt = mk.transcriber(model_id, registry=registry(), sink=mk.records.MemorySink())
    stt.gpu = LeaseRecorder()
    return stt


def test_words_segments_and_options_reach_the_model(take: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch)
    t = client().transcribe(take, language="en", prompt="Hello there")
    assert fake.loads == [
        {"name": "deepdml/faster-whisper-large-v3-turbo-ct2", "device": "cpu", "compute_type": "float16"}
    ]
    (call,) = fake.calls
    assert call == {
        "audio": str(take),
        "language": "en",
        "initial_prompt": "Hello there",
        "word_timestamps": True,
        "beam_size": 5,
        "vad_filter": False,
    }
    assert t.text == "Hello there, and welcome. This is the second line."
    assert [w.text for w in t.words] == t.text.split()
    assert len(t.words) == 9
    assert (t.words[0].start_s, t.words[0].end_s, t.words[0].probability) == (0.0, 0.4, 0.9)
    assert (t.segments[1].start_s, t.segments[1].end_s) == (1.9, 3.7)
    assert (t.language, t.duration_s) == ("en", 4.0)


def test_entry_defaults_and_words_off(take: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch, cuda_devices=1)  # the entry says cpu: no CUDA check
    t = client("test-whisper-cpu").transcribe(take, words=False)
    assert fake.loads[0] == {"name": "tiny.en", "device": "cpu", "compute_type": "int8"}
    assert (fake.calls[0]["beam_size"], fake.calls[0]["vad_filter"], fake.calls[0]["word_timestamps"]) == (
        1,
        True,
        False,
    )
    assert t.words == []
    assert all(s.words == [] for s in t.segments)


def test_timeout_stops_decoding_and_frees(take: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch, segment_delay_s=0.05)
    stt = client()
    with pytest.raises(ModelTimeout, match=r"stopped at 1\.6 s"):
        stt.transcribe(take, timeout_s=0.01)
    assert fake.unloads == 1
    assert stt.sink.spans[0]["status"]["code"] == "error"  # type: ignore[attr-defined]


def test_load_failure_is_a_provider_error(take: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, fail_on_load=OSError("no such repo"))
    with pytest.raises(ProviderError, match="faster-whisper model load failed: OSError: no such repo"):
        client().transcribe(take)


def test_missing_extra_names_it_before_the_lease(take: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    stt = client()
    with pytest.raises(ConfigError, match=r"hone-models\[transcribe\]"):
        stt.transcribe(take)
    assert stt.gpu.calls == []  # type: ignore[attr-defined]
    cpu = registry().get("test-whisper-cpu")
    with pytest.raises(ConfigError, match=r"hone-models\[transcribe\]"):
        faster_whisper.Engine(cpu)


def test_closing_a_model_without_unload_still_frees(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    engine = faster_whisper.Engine(registry().get("test-whisper-cpu"))
    engine.model = object()  # an older CTranslate2 model without unload_model
    engine.close()
    assert engine.model is None


class Libraries:
    """A stand-in for `ctypes.CDLL`: loads only the names or paths in `present`."""

    def __init__(self, *present: str) -> None:
        self.present = set(present)
        self.tried: list[str] = []

    def __call__(self, name: str, mode: int = 0) -> object:
        self.tried.append(name)
        assert mode == ctypes.RTLD_GLOBAL
        if name not in self.present:
            raise OSError(f"{name}: cannot open shared object file")
        return object()


@pytest.fixture
def cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A machine with one GPU, no CUDA libraries loaded yet, and `tmp_path/site` on `sys.path`."""
    install(monkeypatch, cuda_devices=1)
    monkeypatch.setattr(faster_whisper, "_loaded", set())
    site = tmp_path / "site"
    monkeypatch.setattr(sys, "path", [str(site), ""])
    return site


def test_cuda_libraries_on_the_system_path_are_used(cuda: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    libs = Libraries("libcublas.so.12", "libcudnn.so.9")
    monkeypatch.setattr(faster_whisper.ctypes, "CDLL", libs)
    cfg = registry().get("test-whisper")
    faster_whisper.check_ready(cfg)
    faster_whisper.check_ready(cfg)  # found once, remembered
    assert libs.tried == ["libcublas.so.12", "libcudnn.so.9"]
    assert faster_whisper.settings(cfg) == ("cuda", "float16")


def test_cuda_libraries_are_found_in_the_nvidia_wheels(cuda: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    found: list[str] = []
    for folder, soname in (("cublas", "libcublas.so.12"), ("cudnn", "libcudnn.so.9")):
        lib = cuda / "nvidia" / folder / "lib" / soname
        lib.parent.mkdir(parents=True)
        lib.write_bytes(b"")
        found.append(str(lib))
    libs = Libraries(*found)
    monkeypatch.setattr(faster_whisper.ctypes, "CDLL", libs)
    faster_whisper.check_ready(registry().get("test-whisper"))
    assert libs.tried == ["libcublas.so.12", found[0], "libcudnn.so.9", found[1]]


def test_missing_cuda_libraries_raise_before_the_lease(
    cuda: Path, take: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(faster_whisper.ctypes, "CDLL", Libraries("libcublas.so.12"))
    stt = client()
    with pytest.raises(ConfigError, match=r"libcudnn\.so\.9 .*pip install nvidia-cudnn-cu12`") as caught:
        stt.transcribe(take)
    assert "nvidia-cublas-cu12" not in str(caught.value)
    assert "defaults.device = 'cpu'" in str(caught.value)
    assert stt.gpu.calls == []  # type: ignore[attr-defined]
    with pytest.raises(ConfigError, match="nvidia-cudnn-cu12"), stt.session():
        pass
    assert stt.gpu.calls == []  # type: ignore[attr-defined]


def test_both_missing_name_both_wheels(cuda: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(faster_whisper.ctypes, "CDLL", Libraries())
    with pytest.raises(ConfigError, match="pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"):
        faster_whisper.check_ready(registry().get("test-whisper"))


def test_session_keeps_one_model(take: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch)
    stt = client()
    with stt.session():
        stt.transcribe(take)
        stt.transcribe(take)
        assert (len(fake.loads), fake.unloads) == (1, 0)
    assert fake.unloads == 1
    spans: list[dict[str, Any]] = stt.sink.spans  # type: ignore[attr-defined]
    assert [s["attributes"]["hone.models.transcribe.loaded"] for s in spans] == [True, False]
