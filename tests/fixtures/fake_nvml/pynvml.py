"""A fake `pynvml` reporting one GPU with 8 GB total and nothing used (AC-12)."""

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
