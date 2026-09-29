"""The HeartMuLa adapter: request -> heartlib's input and checkpoint folders, and a run against fake
torch / heartlib modules (never the real project)."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hone_models.providers import command


def load_adapter() -> Any:
    spec = importlib.util.spec_from_file_location(
        "heartmula_adapter", command.adapters_folder() / "heartmula.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


heartmula = load_adapter()
LYRICS = "[Verse]\nA quiet light across the water"


def request(tmp_path: Path, **inputs: Any) -> dict[str, Any]:
    return {
        "model": "heartmula-rl-3b",
        "prompt": "calm acoustic guitar, warm , slow",
        "seed": 7,
        "out_dir": str(tmp_path / "out"),
        "inputs": {"lyrics": LYRICS, **inputs},
        "defaults": {"duration_s": 60, "checkpoint": "HeartMuLa-RL-oss-3B-20260123", "low_mem": True},
    }


def checkpoints(folder: Path, model: str = "HeartMuLa-RL-oss-3B-20260123") -> Path:
    for name in (model, "HeartCodec-oss-20260123"):
        (folder / name).mkdir(parents=True)
    (folder / "tokenizer.json").write_text("{}")
    (folder / "gen_config.json").write_text("{}")
    return folder


def test_adapter_imports_only_the_standard_library() -> None:
    source = (command.adapters_folder() / "heartmula.py").read_text()
    top = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    modules = {line.split()[1].split(".")[0] for line in top}
    assert modules <= set(sys.stdlib_module_names) | {"__future__"}, modules


def test_adapter_parses_as_python_3_10() -> None:
    source = (command.adapters_folder() / "heartmula.py").read_text()
    ast.parse(source, feature_version=(3, 10))  # heartlib's environment runs Python 3.10


def test_request_becomes_heartlib_input(tmp_path: Path) -> None:
    song = heartmula.heartmula_input(request(tmp_path, duration_s=10))
    assert song == {
        "lyrics": LYRICS,
        "tags": "calm acoustic guitar,warm,slow",  # comma-separated without spaces (heartlib's README)
        "max_audio_length_ms": 10_000,
        "topk": 50,
        "temperature": 1.0,
        "cfg_scale": 1.5,
    }


def test_duration_and_sampling_from_the_entry_defaults(tmp_path: Path) -> None:
    req = request(tmp_path)
    req["defaults"] |= {"cfg_scale": 2.0, "topk": 40}
    song = heartmula.heartmula_input(req)
    assert (song["max_audio_length_ms"], song["cfg_scale"], song["topk"]) == (60_000, 2.0, 40)
    req["defaults"] = {}
    assert heartmula.heartmula_input(req)["max_audio_length_ms"] == heartmula.DURATION_S * 1000


@pytest.mark.parametrize("lyrics", [None, "", "   "])
def test_lyrics_are_required(tmp_path: Path, lyrics: str | None) -> None:
    with pytest.raises(heartmula.RequestError, match="'lyrics' must be given"):
        heartmula.heartmula_input(request(tmp_path, lyrics=lyrics))


@pytest.mark.parametrize("duration", [0, -5, "ten"])
def test_duration_must_be_positive(tmp_path: Path, duration: Any) -> None:
    with pytest.raises(heartmula.RequestError, match="duration_s must be a positive number"):
        heartmula.heartmula_input(request(tmp_path, duration_s=duration))


def test_checkpoints_relative_to_comfyui_models(tmp_path: Path) -> None:
    folder = checkpoints(tmp_path / "comfy" / "models" / "HeartMuLa")
    paths = heartmula.checkpoint_paths(request(tmp_path), {"HONE_COMFYUI_DIR": str(tmp_path / "comfy")})
    assert paths == {
        "mula": folder / "HeartMuLa-RL-oss-3B-20260123",
        "codec": folder / "HeartCodec-oss-20260123",
        "tokenizer": folder / "tokenizer.json",
        "gen_config": folder / "gen_config.json",
    }


def test_checkpoints_default_to_home_comfyui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = checkpoints(tmp_path / "ComfyUI" / "models" / "HeartMuLa", model="HeartMuLa-oss-3B")
    paths = heartmula.checkpoint_paths({"defaults": {}}, {})
    assert paths["mula"] == folder / "HeartMuLa-oss-3B"


def test_checkpoints_as_an_absolute_folder(tmp_path: Path) -> None:
    folder = checkpoints(tmp_path / "ckpt")
    req = request(tmp_path)
    req["defaults"]["checkpoints"] = str(folder)
    assert heartmula.checkpoint_paths(req, {})["codec"] == folder / "HeartCodec-oss-20260123"


def test_missing_checkpoints_are_named(tmp_path: Path) -> None:
    folder = checkpoints(tmp_path / "ckpt")
    (folder / "gen_config.json").unlink()
    req = request(tmp_path)
    req["defaults"]["checkpoints"] = str(folder)
    with pytest.raises(heartmula.RequestError, match=r"gen_config\.json"):
        heartmula.checkpoint_paths(req, {})


def test_cache_fits_the_song() -> None:
    backbone = SimpleNamespace(max_seq_len=8192)
    heartmula.fit_cache(backbone, 120, 10_000)
    assert backbone.max_seq_len == 120 + 125 + 1  # 80 ms frames
    heartmula.fit_cache(backbone, 120, 10_000_000)
    assert backbone.max_seq_len == 246  # never above what it was


class FakeHeartlib:
    """Stand-ins for torch, numpy, tokenizers, transformers, soundfile and heartlib's pipeline module."""

    def __init__(self, *, cuda: bool = True, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.fail = fail
        record = self.calls.append
        self.torch = SimpleNamespace(
            manual_seed=lambda s: record(("torch", s)),
            device=lambda name: f"device:{name}",
            bfloat16="bf16",
            float32="fp32",
            no_grad=_Nothing,
            cuda=SimpleNamespace(
                is_available=lambda: cuda,
                manual_seed_all=lambda s: record(("cuda", s)),
                empty_cache=lambda: record(("empty_cache", None)),
            ),
        )
        self.numpy = SimpleNamespace(random=SimpleNamespace(seed=lambda s: record(("numpy", s))))
        self.tokenizers = SimpleNamespace(
            Tokenizer=SimpleNamespace(from_file=lambda p: f"tokenizer:{Path(p).name}")
        )
        self.transformers = SimpleNamespace(BitsAndBytesConfig=lambda **kw: ("nf4", kw))
        self.soundfile = SimpleNamespace(write=self._write)
        self.pipelines = SimpleNamespace(
            HeartMuLaGenPipeline=self._pipeline,
            HeartMuLaGenConfig=SimpleNamespace(from_file=lambda p: f"config:{Path(p).name}"),
            HeartMuLa=SimpleNamespace(from_pretrained=self._mula),
        )

    def _pipeline(self, **kwargs: Any) -> "FakePipeline":
        self.calls.append(("pipeline", kwargs))
        return FakePipeline(self)

    def _mula(self, path: str, **kwargs: Any) -> Any:
        self.calls.append(("mula", (Path(path).name, kwargs)))
        return SimpleNamespace(backbone=SimpleNamespace(max_seq_len=8192))

    def _write(self, path: str, data: Any, rate: int, format: str) -> None:
        self.calls.append(("write", (data, rate, format)))
        Path(path).write_bytes(b"fLaC")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        modules = {"torch": self.torch, "numpy": self.numpy, "tokenizers": self.tokenizers,
                   "transformers": self.transformers, "soundfile": self.soundfile,
                   "heartlib.pipelines.music_generation": self.pipelines}  # fmt: skip
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)


class _Nothing:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


class FakeWave:
    """A decoded song: 2 channels, 10 s at 48 kHz, as numpy would hold it."""

    shape = (2, 480_000)
    T = "transposed"

    def to(self, dtype: str) -> "FakeWave":
        return self

    def cpu(self) -> "FakeWave":
        return self

    def numpy(self) -> "FakeWave":
        return self


class FakePipeline:
    def __init__(self, fake: FakeHeartlib) -> None:
        self.fake = fake
        self._mula: Any = None
        self.mula = SimpleNamespace(backbone=SimpleNamespace(max_seq_len=8192))
        self.codec_device = "device:cuda"
        self.codec = SimpleNamespace(
            detokenize=self._detokenize,
            flow_matching=SimpleNamespace(to=lambda device: fake.calls.append(("flow to", device))),
            scalar_model=SimpleNamespace(decode=lambda latent: fake.calls.append(("decode", latent))),
        )

    def preprocess(self, inputs: dict[str, Any], cfg_scale: float) -> dict[str, Any]:
        self.fake.calls.append(("preprocess", (inputs, cfg_scale)))
        return {"tokens": SimpleNamespace(shape=(2, 30, 9))}

    def _forward(self, prompt: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.fake.calls.append(("forward", (self.mula.backbone.max_seq_len, kwargs)))
        if self.fake.fail:
            raise self.fake.fail
        return {"frames": SimpleNamespace(to=lambda device: ("frames", device))}

    def _detokenize(self, frames: Any) -> FakeWave:
        self.fake.calls.append(("detokenize", frames))
        self.codec.scalar_model.decode("latent")  # after the flow-matching part made every latent
        return FakeWave()

    def _unload(self) -> None:
        self.fake.calls.append(("unload", None))


def run_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeHeartlib, **inputs: Any) -> dict:
    fake.install(monkeypatch)
    monkeypatch.setenv("HONE_COMFYUI_DIR", str(tmp_path / "comfy"))
    checkpoints(tmp_path / "comfy" / "models" / "HeartMuLa")
    job = tmp_path / "job"
    (job / "out").mkdir(parents=True)
    req = {**request(tmp_path, **inputs), "out_dir": str(job / "out")}
    (job / "request.json").write_text(json.dumps(req))
    assert heartmula.main([str(job / "request.json")]) == 0
    return json.loads((job / "out" / "result.json").read_text())


def test_run_seeds_loads_4bit_and_writes_flac(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHeartlib()
    result = run_adapter(tmp_path, monkeypatch, fake, duration_s=10)
    assert result == {
        "files": ["take.flac"],
        "error": None,
        "meta": {"checkpoint": "HeartMuLa-RL-oss-3B-20260123", "codec": "HeartCodec-oss-20260123",
                 "low_mem": True, "seconds": 10.0},
    }  # fmt: skip
    assert (tmp_path / "job" / "out" / "take.flac").read_bytes() == b"fLaC"
    names = [c[0] for c in fake.calls]
    assert names == ["pipeline", "mula", "numpy", "torch", "cuda", "preprocess", "forward", "detokenize",
                     "flow to", "empty_cache", "decode", "unload", "write"]  # fmt: skip
    assert fake.calls[8][1] == "cpu"
    built = fake.calls[0][1]
    assert built["heartmula_path"].endswith("HeartMuLa-RL-oss-3B-20260123")
    assert (built["lazy_load"], built["heartmula_dtype"], built["heartcodec_dtype"]) == (True, "bf16", "fp32")
    assert (built["text_tokenizer"], built["config"]) == (
        "tokenizer:tokenizer.json",
        "config:gen_config.json",
    )
    assert fake.calls[1][1][1]["quantization_config"][1]["bnb_4bit_quant_type"] == "nf4"
    assert [c[1] for c in fake.calls[2:5]] == [7, 7, 7]
    assert fake.calls[5][1] == ({"lyrics": LYRICS, "tags": "calm acoustic guitar,warm,slow"}, 1.5)
    cache, settings = fake.calls[6][1]
    assert cache == 30 + 125 + 1
    assert settings == {"max_audio_length_ms": 10_000, "temperature": 1.0, "topk": 50, "cfg_scale": 1.5}
    assert fake.calls[-1][1] == ("transposed", 48_000, "FLAC")


def test_full_precision_and_codec_on_the_gpu_without_low_mem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeHeartlib()
    monkeypatch.setattr(heartmula, "heartmula_input", _defaults_off(heartmula.heartmula_input))
    result = run_adapter(tmp_path, monkeypatch, fake)
    assert result["meta"]["low_mem"] is False
    names = [c[0] for c in fake.calls]
    assert "mula" not in names
    assert "flow to" not in names
    assert "decode" in names


def _defaults_off(convert: Any) -> Any:
    def wrapped(req: dict[str, Any]) -> Any:
        req["defaults"]["low_mem"] = False
        return convert(req)

    return wrapped


def test_generation_failure_is_an_error_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHeartlib(fail=RuntimeError("CUDA out of memory. Tried to allocate 96 MiB"))
    result = run_adapter(tmp_path, monkeypatch, fake)
    assert result["files"] == []
    assert result["error"] == "RuntimeError: CUDA out of memory. Tried to allocate 96 MiB"
    assert result["meta"]["checkpoint"] == "HeartMuLa-RL-oss-3B-20260123"


def test_no_cuda_is_an_error_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = run_adapter(tmp_path, monkeypatch, FakeHeartlib(cuda=False))
    assert result["error"] == "RuntimeError: CUDA is not available: HeartMuLa needs a GPU"


def test_a_broken_environment_crashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHeartlib()
    fake.install(monkeypatch)
    monkeypatch.setitem(sys.modules, "heartlib.pipelines.music_generation", None)  # import fails
    monkeypatch.setenv("HONE_COMFYUI_DIR", str(tmp_path / "comfy"))
    checkpoints(tmp_path / "comfy" / "models" / "HeartMuLa")
    (tmp_path / "out").mkdir()
    (tmp_path / "request.json").write_text(json.dumps(request(tmp_path)))
    with pytest.raises(ImportError):
        heartmula.main([str(tmp_path / "request.json")])


def test_bad_request_is_an_error_result_before_importing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeHeartlib()
    result = run_adapter(tmp_path, monkeypatch, fake, lyrics="")
    assert "'lyrics' must be given" in result["error"]
    assert fake.calls == []
