# pyright: reportAttributeAccessIssue=false, reportUndefinedVariable=false
"""Fake generate.py: writes what it was given as the song."""

import json
import os
import sys
from pathlib import Path

import omegaconf
import torch


def generate(args):
    raise AssertionError("an 8 GiB card takes the low-memory path")


def generate_lowmem(args):
    item = json.loads(Path(args.input_jsonl).read_text(encoding="utf-8"))
    seen = {
        "item": item,
        "args": vars(args),
        "seeds": torch.seeds,
        "cudnn": torch.backends.cudnn.enabled,
        "resolvers": sorted(omegaconf.resolvers),
        "offload": OffloadProfiler is not None,  # noqa: F821 - set on this module by the adapter
        "cwd": os.getcwd(),
        "path": sys.path[:3],
        "omp": os.environ.get("OMP_NUM_THREADS"),
    }
    audios = Path(args.save_dir) / "audios"
    audios.mkdir(parents=True, exist_ok=True)
    (audios / f"{item['idx']}.flac").write_text(json.dumps(seen), encoding="utf-8")
