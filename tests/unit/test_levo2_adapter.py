"""The SongGeneration (LeVo 2) adapter: request -> LeVo's JSONL and arguments, and a run against fake
project modules (never the real project)."""

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hone_models.providers import command


def load_adapter() -> Any:
    spec = importlib.util.spec_from_file_location("levo2_adapter", command.adapters_folder() / "levo2.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


levo2 = load_adapter()
LYRICS = "[intro-short] ; [verse] Trails wind through the forest. Trees stand tall. ; [outro-short]"


def request(tmp_path: Path, **inputs: Any) -> dict[str, Any]:
    return {
        "model": "songgeneration-v2-medium",
        "prompt": "female, pop, piano",
        "seed": 7,
        "out_dir": str(tmp_path / "out"),
        "inputs": {"lyrics": LYRICS, "generate_type": "mixed", **inputs},
        "defaults": {"generate_type": "mixed", "low_mem": True},
    }


def test_adapter_imports_only_the_standard_library() -> None:
    source = (command.adapters_folder() / "levo2.py").read_text()
    top = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    modules = {line.split()[1].split(".")[0] for line in top}
    assert modules <= set(sys.stdlib_module_names) | {"__future__"}, modules


def test_adapter_parses_as_python_3_10() -> None:
    source = (command.adapters_folder() / "levo2.py").read_text()
    ast.parse(source, feature_version=(3, 10))  # the project's environment runs Python 3.10


def test_request_becomes_one_jsonl_item(tmp_path: Path) -> None:
    item = levo2.levo_item(request(tmp_path))
    assert item == {"idx": "take", "gt_lyric": LYRICS, "descriptions": "female, pop, piano"}


def test_lyrics_pass_through_as_given(tmp_path: Path) -> None:
    lyrics = "[verse] Line one. Line two. ; [chorus] Hook line."
    assert levo2.levo_item(request(tmp_path, lyrics=f"\n{lyrics}\n"))["gt_lyric"] == lyrics


def test_no_prompt_means_no_descriptions(tmp_path: Path) -> None:
    item = levo2.levo_item({**request(tmp_path), "prompt": "  "})
    assert "descriptions" not in item


@pytest.mark.parametrize("lyrics", [None, "", "   "])
def test_lyrics_are_required(tmp_path: Path, lyrics: str | None) -> None:
    with pytest.raises(levo2.RequestError, match="'lyrics' must be given"):
        levo2.levo_item(request(tmp_path, lyrics=lyrics))


def test_arguments_from_inputs_and_entry_defaults(tmp_path: Path) -> None:
    args = levo2.levo_args(request(tmp_path), tmp_path / "levo2", tmp_path / "in.jsonl")
    assert args.ckpt_path == str(tmp_path / "levo2" / "songgeneration_v2_medium")
    assert (args.input_jsonl, args.save_dir) == (str(tmp_path / "in.jsonl"), str(tmp_path / "out"))
    assert (args.generate_type, args.low_mem, args.use_flash_attn) == ("mixed", True, False)
    assert levo2.script_flags(args) == ["--low_mem", "--not_use_flash_attn"]
    assert levo2.output_files(args) == ["audios/take.flac"]


def test_separate_with_flash_attention_on_another_checkpoint(tmp_path: Path) -> None:
    req = request(tmp_path, generate_type="separate")
    req["defaults"] = {"checkpoint": "songgeneration_v2_large", "flash_attn": True}
    args = levo2.levo_args(req, tmp_path, tmp_path / "in.jsonl")
    assert args.ckpt_path.endswith("songgeneration_v2_large")
    assert levo2.script_flags(args) == ["--separate"]
    assert levo2.output_files(args) == ["audios/take.flac", "audios/take_vocal.flac", "audios/take_bgm.flac"]


def test_unknown_generate_type(tmp_path: Path) -> None:
    with pytest.raises(levo2.RequestError, match="generate_type must be one of"):
        levo2.levo_args(request(tmp_path, generate_type="karaoke"), tmp_path, tmp_path / "in.jsonl")


class FakeProject:
    """Stand-ins for torch, numpy, omegaconf and the project's generate.py, installed in sys.modules."""

    def __init__(self, *, cuda: bool = True, fail: Exception | None = None, free_gb: float = 8) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.resolvers: dict[str, Any] = {}
        self.fail = fail
        record = self.calls.append
        props = SimpleNamespace(total_memory=int(free_gb * 1024**3))
        self.torch = SimpleNamespace(
            manual_seed=lambda s: record(("torch", s)),
            backends=SimpleNamespace(cudnn=SimpleNamespace(enabled=True)),
            cuda=SimpleNamespace(
                is_available=lambda: cuda,
                manual_seed_all=lambda s: record(("cuda", s)),
                current_device=lambda: 0,
                get_device_properties=lambda d: props,
                memory_reserved=lambda d: 0,
            ),
        )
        self.numpy = SimpleNamespace(random=SimpleNamespace(seed=lambda s: record(("numpy", s))))
        omegaconf = SimpleNamespace(register_new_resolver=self.resolvers.__setitem__, load=lambda x: {"a": 1})
        self.omegaconf = SimpleNamespace(OmegaConf=omegaconf)
        self.generate = SimpleNamespace(generate=self._run("generate"), generate_lowmem=self._run("lowmem"))
        self.offload = SimpleNamespace(OffloadProfiler="profiler", OffloadParamParse="parse")

    def _run(self, name: str) -> Any:
        def run(args: Any) -> None:
            self.calls.append((name, args))
            if self.fail:
                raise self.fail
            audios = Path(args.save_dir) / "audios"
            audios.mkdir(parents=True, exist_ok=True)
            for file in levo2.output_files(args):
                (Path(args.save_dir) / file).write_bytes(b"fLaC")

        return run

    def install(self, monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
        modules = {"torch": self.torch, "numpy": self.numpy, "omegaconf": self.omegaconf,
                   "generate": self.generate, "codeclm.utils.offload_profiler": self.offload}  # fmt: skip
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(sys, "argv", list(sys.argv))
        for key in levo2.project_environment(project):
            monkeypatch.setenv(key, os.environ.get(key, ""))
        monkeypatch.chdir(project)


def run_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeProject, **inputs: Any) -> dict:
    project = tmp_path / "levo2"
    project.mkdir()
    fake.install(monkeypatch, project)
    job = tmp_path / "job"
    (job / "out").mkdir(parents=True)
    req = {**request(tmp_path, **inputs), "out_dir": str(job / "out")}
    (job / "request.json").write_text(json.dumps(req))
    assert levo2.main([str(job / "request.json")]) == 0
    return json.loads((job / "out" / "result.json").read_text())


def test_run_seeds_and_uses_the_low_memory_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeProject()
    result = run_adapter(tmp_path, monkeypatch, fake)
    assert result["files"] == ["audios/take.flac"]
    assert result["error"] is None
    assert result["meta"]["flags"] == ["--low_mem", "--not_use_flash_attn"]
    assert fake.calls[:3] == [("numpy", 7), ("torch", 7), ("cuda", 7)]
    name, args = fake.calls[3]
    assert name == "lowmem"
    assert fake.generate.OffloadProfiler == "profiler"
    assert fake.torch.backends.cudnn.enabled is False
    assert sorted(fake.resolvers) == ["concat", "eval", "get_fname", "load_yaml"]
    assert fake.resolvers["concat"]([1], [2, 3]) == [1, 2, 3]
    assert fake.resolvers["eval"]("1 + 1") == 2
    assert fake.resolvers["get_fname"]() == "--ckpt_path"  # as when generate.py runs as a script
    assert fake.resolvers["load_yaml"]("x.yaml") == ["a"]
    assert Path.cwd() == tmp_path / "levo2"
    assert os.environ["TRANSFORMERS_CACHE"] == str(tmp_path / "levo2" / "third_party" / "hub")
    jsonl = Path(args.input_jsonl).read_text().splitlines()
    assert json.loads(jsonl[0]) == {"idx": "take", "gt_lyric": LYRICS, "descriptions": "female, pop, piano"}


def test_run_with_enough_memory_and_no_low_mem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeProject(free_gb=40)
    monkeypatch.setattr(levo2, "levo_args", _without_low_mem(levo2.levo_args))
    result = run_adapter(tmp_path, monkeypatch, fake, generate_type="separate")
    assert [c[0] for c in fake.calls][-1] == "generate"
    assert len(result["files"]) == 3


def _without_low_mem(levo_args: Any) -> Any:
    def wrapped(*a: Any) -> Any:
        args = levo_args(*a)
        args.low_mem = False
        return args

    return wrapped


def test_generation_failure_is_an_error_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeProject(fail=RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    result = run_adapter(tmp_path, monkeypatch, fake)
    assert result["files"] == []
    assert result["error"] == "RuntimeError: CUDA out of memory. Tried to allocate 2 GiB"


def test_no_cuda_is_an_error_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = run_adapter(tmp_path, monkeypatch, FakeProject(cuda=False))
    assert result["error"] == "RuntimeError: CUDA is not available: SongGeneration needs a GPU"


def test_bad_request_is_an_error_result_before_importing_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeProject()
    result = run_adapter(tmp_path, monkeypatch, fake, lyrics="")
    assert "'lyrics' must be given" in result["error"]
    assert fake.calls == []
    assert fake.resolvers == {}
