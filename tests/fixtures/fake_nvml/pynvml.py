"""A fake `pynvml`: one GPU ("Fake GPU"), 8 GB total, nothing used, 0 % utilization (AC-12)."""

from types import SimpleNamespace

GB = 1024**3


class NVMLError(Exception):
    pass


def nvmlInit() -> None:
    pass


def nvmlShutdown() -> None:
    pass


def nvmlDeviceGetHandleByIndex(index: int) -> int:
    return index


def nvmlDeviceGetMemoryInfo(handle: int) -> SimpleNamespace:
    return SimpleNamespace(total=8 * GB, used=0, free=8 * GB)


def nvmlDeviceGetComputeRunningProcesses(handle: int) -> list[SimpleNamespace]:
    return []


def nvmlDeviceGetCount() -> int:
    return 1


def nvmlDeviceGetName(handle: int) -> str:
    return "Fake GPU"


def nvmlDeviceGetUtilizationRates(handle: int) -> SimpleNamespace:
    return SimpleNamespace(gpu=0, memory=0)
