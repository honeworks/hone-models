"""hone-models adapter for SongGeneration / LeVo 2 (https://github.com/tencent-ailab/SongGeneration).

The `command` provider runs it with the project's own Python, in the project's folder:

    <levo2>/.venv/bin/python levo2.py <request.json>

It imports only the standard library until the request is converted to LeVo's input (one JSONL line with
`idx`, `gt_lyric`, `descriptions`; the lyrics arrive already in LeVo's section format). Then it hands over
to the project: it sets the environment and import paths of the project's `generate.sh`, imports its
`generate.py`, sets the seeds the project does not take as options (Python, NumPy, torch) and runs the
same generation as `generate.sh <checkpoint> <jsonl> <out_dir> [--low_mem] [--not_use_flash_attn]
[--bgm|--vocal|--separate]`. It writes `result.json` into `out_dir`: the files, or the error of a job
that ran and failed. Failing to import the project is a crash (a non-zero exit with the traceback).

It calls `generate()` / `generate_lowmem()` itself instead of running `generate.py` as a script, because
the script reseeds NumPy from the clock and accepts only the v1 checkpoint names (design/decisions.md).
It runs on the project's Python (3.10).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

IDX = "take"  # LeVo names the output `audios/<idx>.flac`
CHECKPOINT = "songgeneration_v2_medium"  # a folder in the project; the entry's `defaults.checkpoint`
GENERATE_TYPES = ("mixed", "vocal", "bgm", "separate")
FULL_MEMORY_GB = 24  # generate.py uses its low-memory path below this much free GPU memory (large: 36)


class RequestError(ValueError):
    """The request cannot be converted to LeVo's input."""


def levo_item(request: dict[str, Any]) -> dict[str, Any]:
    """LeVo's JSONL line for one request: `idx`, `gt_lyric` and, when the prompt is given, `descriptions`
    (LeVo wants comma-separated tags: gender, genre, emotion, instruments)."""
    lyrics = request.get("inputs", {}).get("lyrics")
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise RequestError("SongGeneration sings lyrics: the input 'lyrics' must be given")
    item: dict[str, Any] = {"idx": IDX, "gt_lyric": lyrics.strip()}
    descriptions = str(request.get("prompt") or "").strip()
    if descriptions:
        item["descriptions"] = descriptions
    return item


def levo_args(request: dict[str, Any], project: Path, jsonl: Path) -> argparse.Namespace:
    """The arguments of `generate.py` for one request: the checkpoint, input and output, the generate
    type (an input), low memory and flash attention (the entry's `defaults.low_mem` / `flash_attn`)."""
    inputs, defaults = request.get("inputs", {}), request.get("defaults", {})
    generate_type = inputs.get("generate_type", defaults.get("generate_type", "mixed"))
    if generate_type not in GENERATE_TYPES:
        raise RequestError(f"generate_type must be one of {list(GENERATE_TYPES)}, not {generate_type!r}")
    return argparse.Namespace(
        ckpt_path=str(project / str(defaults.get("checkpoint", CHECKPOINT))),
        input_jsonl=str(jsonl),
        save_dir=str(request["out_dir"]),
        generate_type=generate_type,
        use_flash_attn=bool(defaults.get("flash_attn", False)),
        low_mem=bool(defaults.get("low_mem", False)),
    )


def script_flags(args: argparse.Namespace) -> list[str]:
    """The `generate.sh` flags that ask for the same run (recorded in `result.json`'s `meta`)."""
    flags = ["--low_mem"] if args.low_mem else []
    flags += [] if args.use_flash_attn else ["--not_use_flash_attn"]
    return flags + ([] if args.generate_type == "mixed" else [f"--{args.generate_type}"])


def output_files(args: argparse.Namespace) -> list[str]:
    """The files LeVo writes for the request, relative to `out_dir`."""
    stems = [IDX, f"{IDX}_vocal", f"{IDX}_bgm"] if args.generate_type == "separate" else [IDX]
    return [f"audios/{stem}.flac" for stem in stems]


def project_environment(project: Path) -> dict[str, str]:
    """The variables `generate.sh` exports (set before torch is imported)."""
    return {
        "USER": "root",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TRANSFORMERS_CACHE": str(project / "third_party" / "hub"),
        "NCCL_HOME": "/usr/local/tccl",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_LAUNCH_BLOCKING": "0",
    }


def write_result(out_dir: Path, files: list[str], error: str | None, meta: dict[str, Any]) -> None:
    body = {"files": files, "error": error, "meta": meta}
    (out_dir / "result.json").write_text(json.dumps(body, indent=2), encoding="utf-8")


def hand_over(project: Path, args: argparse.Namespace) -> tuple[Any, Any]:
    """Import the project as `generate.sh` runs it; returns its `generate` module and torch."""
    os.environ.update(project_environment(project))
    sys.dont_write_bytecode = True  # PYTHONDONTWRITEBYTECODE only counts when Python starts
    tokenizer = project / "codeclm" / "tokenizer"
    sys.path[:0] = [str(tokenizer), str(project), str(tokenizer / "Flow1dVAE")]
    os.chdir(project)  # the project opens files relative to its folder
    sys.argv = ["generate.py", "--ckpt_path", args.ckpt_path, "--input_jsonl", args.input_jsonl]
    torch = importlib.import_module("torch")
    levo = importlib.import_module("generate")
    torch.backends.cudnn.enabled = False
    register_resolvers(importlib.import_module("omegaconf").OmegaConf)
    return levo, torch


def register_resolvers(omegaconf: Any) -> None:
    """The OmegaConf resolvers `generate.py` registers when it runs as a script; its configs use them."""

    def evaluate(text: str) -> Any:
        return eval(text)  # noqa: S307 - the project's own config, as in generate.py

    def concat(*lists: list[Any]) -> list[Any]:
        return [item for part in lists for item in part]

    def file_name() -> str:
        return Path(sys.argv[1]).stem

    def load_yaml(path: str) -> list[Any]:
        return list(omegaconf.load(path))

    for name, resolver in (("eval", evaluate), ("concat", concat), ("get_fname", file_name),
                           ("load_yaml", load_yaml)):  # fmt: skip
        omegaconf.register_new_resolver(name, resolver)


def generate(levo: Any, torch: Any, args: argparse.Namespace, seed: int) -> None:
    """Seed everything, then run the generation `generate.py`'s main would choose."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: SongGeneration needs a GPU")
    random.seed(seed)
    importlib.import_module("numpy").random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    free_gb = (total - torch.cuda.memory_reserved(device)) / 1024**3
    enough = 36 if "large" in Path(args.ckpt_path).name else FULL_MEMORY_GB
    if not args.low_mem and free_gb > enough:
        levo.generate(args)
        return
    offload = importlib.import_module("codeclm.utils.offload_profiler")
    levo.OffloadProfiler, levo.OffloadParamParse = offload.OffloadProfiler, offload.OffloadParamParse
    levo.generate_lowmem(args)


def main(argv: list[str] | None = None) -> int:
    """Run one request; the exit code is 0 whenever `result.json` was written."""
    request_path = Path((argv if argv is not None else sys.argv[1:])[0]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    out_dir, project = Path(request["out_dir"]), Path.cwd()
    try:
        item = levo_item(request)
        args = levo_args(request, project, request_path.parent / "levo_input.jsonl")
    except RequestError as exc:
        write_result(out_dir, [], str(exc), {})
        return 0
    Path(args.input_jsonl).write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {"checkpoint": Path(args.ckpt_path).name, "generate_type": args.generate_type,
            "flags": script_flags(args)}  # fmt: skip
    levo, torch = hand_over(project, args)
    try:
        generate(levo, torch, args, int(request["seed"]))
    except Exception as exc:  # the job ran and failed: a result, not a crash
        traceback.print_exc()
        write_result(out_dir, [], f"{type(exc).__name__}: {exc}", meta)
        return 0
    write_result(out_dir, [f for f in output_files(args) if (out_dir / f).is_file()], None, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
