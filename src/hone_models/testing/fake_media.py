"""`FakeMedia`: an offline stand-in for `mk.image(...)`, `mk.music(...)` and `mk.video(...)`.

    img = FakeMedia.like("z-image-turbo")             # the entry's id, kind, inputs and capabilities
    r = img.generate("a lighthouse", size="512x512", out="shot.png")    # a black 512 x 512 PNG
    img.calls                                         # [(prompt, inputs, seed), ...]
    FakeMedia(kind="music").generate("lo-fi", duration_s=3, out="take.wav")   # 3 s of silence
    img.fail_next("CUDA out of memory")               # the next call: result.error, error_kind

Same validation (unknown inputs, missing files, sizes and durations), span and result as the real client;
no server, no GPU lease. Images are black PNGs of the requested `size` (default 64 x 64, `n` of them),
music is silence of `duration_s` (default 1 s), video a packaged one-second MP4.
"""

from __future__ import annotations

from typing import Any

from .._media_files import output_paths
from ..errors import ConfigError
from ..gpu import NullGpuLease
from ..media import MediaClient
from ..ports import RecordSink
from ..providers import MEDIA
from ..providers.common import MediaJob, MediaOutcome, MediaProvider, error_kind, listed_inputs
from ..records import MemorySink
from ..registry import MEDIA_KINDS, ModelConfig, Registry, load
from . import _fake_files


class FakeMedia(MediaClient):
    """Writes small valid files instead of generating; records every call in `calls`. Without `like`,
    it takes the shared inputs (`size`, `duration_s`, `references`, ...). `sink` defaults to a
    `MemorySink` (read the spans from `fake.sink.spans`)."""

    def __init__(
        self, model_id: str | None = None, *, kind: str = "image", sink: RecordSink | None = None
    ) -> None:
        if kind not in MEDIA_KINDS:
            raise ConfigError(f"kind must be one of {list(MEDIA_KINDS)}, not {kind!r}")
        config = ModelConfig(id=model_id or f"fake-{kind}", provider="comfyui", kind=kind)  # pyright: ignore[reportArgumentType]
        super().__init__(config, sink if sink is not None else MemorySink(), lease=NullGpuLease())
        self.calls: list[tuple[str, dict[str, Any], int]] = []
        self._failures: list[str] = []

    @classmethod
    def like(
        cls, model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None
    ) -> FakeMedia:
        """A fake with the id, kind, inputs, defaults and capabilities of the registry entry `model_id`."""
        cfg = (registry or load()).get(model_id)
        if cfg.kind not in MEDIA_KINDS:
            raise ConfigError(f"model {cfg.id!r} is a {cfg.kind} model, not an image, music or video model")
        fake = cls(cfg.id, kind=cfg.kind, sink=sink)
        fake.config = cfg
        return fake

    def fail_next(self, message: str) -> None:
        """Make the next call a job that ran without output: `result.error` is `message`."""
        self._failures.append(message)

    def _provider(self) -> MediaProvider:
        real = MEDIA.get(self.config.provider)
        accepted = real.accepted if real is not None and self.config.inputs is not None else listed_inputs
        return MediaProvider(accepted=accepted, run=self._run)

    def _run(self, job: MediaJob) -> MediaOutcome:
        self.calls.append((job.prompt, dict(job.inputs), job.seed))
        job_id = f"fake-{len(self.calls)}"
        if self._failures:
            message = self._failures.pop(0)
            return MediaOutcome([], job_id, message, error_kind(message))
        data, suffix = _file(self.config.kind, job.inputs)
        paths = output_paths(job.out, [suffix] * int(job.inputs.get("n") or 1))
        for path in paths:
            path.write_bytes(data)
        return MediaOutcome(paths, job_id)


def _file(kind: str, inputs: dict[str, Any]) -> tuple[bytes, str]:
    if kind == "image":
        width, height = str(inputs.get("size") or "64x64").split("x")
        return _fake_files.png(int(width), int(height)), ".png"
    if kind == "music":
        return _fake_files.wav(float(inputs.get("duration_s") or 1.0)), ".wav"
    return _fake_files.mp4(), ".mp4"
