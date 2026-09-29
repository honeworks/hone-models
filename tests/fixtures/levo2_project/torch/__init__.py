"""Fake torch: records the seeds, has a CUDA device with 8 GiB free."""

from types import SimpleNamespace

seeds: dict[str, int] = {}


def manual_seed(seed: int) -> None:
    seeds["torch"] = seed


def _cuda_seed(seed: int) -> None:
    seeds["cuda"] = seed


backends = SimpleNamespace(cudnn=SimpleNamespace(enabled=True))
cuda = SimpleNamespace(
    is_available=lambda: True,
    manual_seed_all=_cuda_seed,
    current_device=lambda: 0,
    get_device_properties=lambda device: SimpleNamespace(total_memory=8 * 1024**3),
    memory_reserved=lambda device: 0,
)
