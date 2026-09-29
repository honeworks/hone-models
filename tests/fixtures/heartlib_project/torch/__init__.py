"""Fake torch: records the seeds and what happened on the GPU."""

import contextlib
from types import SimpleNamespace

seeds: dict[str, int] = {}
events: list[str] = []
bfloat16, float32 = "bf16", "fp32"


def manual_seed(seed: int) -> None:
    seeds["torch"] = seed


def device(name: str) -> str:
    return name


def no_grad() -> contextlib.AbstractContextManager[None]:
    return contextlib.nullcontext()


def _cuda_seed(seed: int) -> None:
    seeds["cuda"] = seed


cuda = SimpleNamespace(
    is_available=lambda: True, manual_seed_all=_cuda_seed, empty_cache=lambda: events.append("empty_cache")
)
