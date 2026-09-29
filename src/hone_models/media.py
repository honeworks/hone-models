"""Images, music and video: `mk.image(...)`, `mk.music(...)`, `mk.video(...)` (change 0015 §1).

    img = mk.image("z-image-turbo")
    r = img.generate("a lighthouse at dusk", size="1024x1024", seed=7, out="shots/01.png")
    r.path, r.files, r.seed, r.error, r.error_kind, r.span_id, r.cost_usd
    with mk.music("ace-step-1.5-xl-turbo").session() as song:   # lease, server and model held
        song.generate("dark trap, 95 bpm", lyrics=text, duration_s=150, out="takes/take.flac")

One client class for the three kinds. A call checks its named inputs before anything else (an input the
model does not take, a missing file, a size or duration the entry does not declare: `ConfigError` /
`CapabilityError`), picks the seed (`defaults.seed`, else random; always recorded), records one
`hone.models.<kind>` span, measures the files written and prices them from the registry.

Providers plug in through `providers.MEDIA` (registry `provider` -> `MediaProvider`, see
`providers/common.py`): `accepted(cfg)` names the inputs an entry takes, `run(job)` runs one job and
returns the files it wrote (entering `job.lease()` around the GPU work), and `session(cfg)` holds
whatever a `client.session()` block keeps (a server, a loaded model). `comfyui` is the first; a hosted
or `command` provider is one more module and one more row in that table.
"""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from . import gpu
from ._media_files import MediaFile as MediaFile  # noqa: PLC0414 - part of this module's API
from ._media_files import file_record, measure
from ._tracing import start_span
from .errors import CapabilityError, ConfigError
from .ports import RecordSink
from .providers import MEDIA, lookup, model_attributes
from .providers.common import FILE_INPUTS, MediaJob, MediaProvider
from .records import default_sink
from .registry import ModelConfig, Registry, load

_SIZE = re.compile(r"^(\d+)x(\d+)$")


@dataclass(frozen=True, slots=True)
class MediaResult:
    """One generation call: the files written (`path` is the first, or `None`), the seed used, the span,
    the provider's job id, and for a job that ran without usable output `error` and `error_kind`
    (`out_of_memory`, `refused`, `invalid_input`, `no_output` or `failed`). `cost_usd` is a naive flat
    estimate from the registry price (`cost_estimated`), `None` without one; `license` and
    `commercial_use` come from the registry (`None` when not declared)."""

    files: list[MediaFile]
    model: str
    seed: int
    span_id: str
    elapsed_s: float
    error: str | None = None
    error_kind: str | None = None
    job_id: str | None = None
    cost_usd: float | None = None
    cost_estimated: bool = False
    license: str | None = None
    commercial_use: bool | None = None

    @property
    def path(self) -> Path | None:
        """The first file, or `None` when there is none."""
        return self.files[0].path if self.files else None


class MediaClient:
    """Generates images, music or video with one registry model; records a `hone.models.<kind>` span per
    call and takes a GPU lease for local models."""

    def __init__(self, config: ModelConfig, sink: RecordSink, *, lease: Any = None) -> None:
        self.config = config
        self.sink = sink
        self.gpu = lease or gpu.GPU
        self._state: dict[str, Any] | None = None  # the provider's session state inside `session()`

    @property
    def model_id(self) -> str:
        return self.config.id

    @property
    def kind(self) -> str:
        """`image`, `music` or `video`."""
        return self.config.kind

    @property
    def inputs(self) -> list[str]:
        """The named inputs `generate` takes besides `prompt` and `seed`."""
        return sorted(self._provider().accepted(self.config))

    def generate(
        self,
        prompt: str,
        *,
        out: str | Path,
        seed: int | None = None,
        timeout_s: float | None = None,
        trace: Mapping[str, str] | None = None,
        **inputs: Any,
    ) -> MediaResult:
        """Run one job and write its files to `out` (several: `<stem>_1<suffix>`, ...).

        `inputs` are the named inputs of `self.inputs` (`size="1024x1024"`, `references=[Path(...)]`,
        `duration_s=30`, ...); a `pathlib.Path` is a file. `timeout_s` defaults to the entry's
        `max_timeout_s`; on timeout the job is cancelled and `ModelTimeout` raised."""
        cfg, provider = self.config, self._provider()
        named = self._checked(provider, inputs)
        seed = seed if seed is not None else int(cfg.defaults.get("seed", secrets.randbelow(2**32)))
        caps = cfg.capabilities
        attrs = {
            **model_attributes(cfg, cfg.kind),
            "gen_ai.request.seed": seed,
            "hone.models.media.prompt": prompt,
            "hone.models.media.inputs": _recorded(named),
            "hone.models.media.session": self._state is not None,
            "hone.models.media.license": caps.license,
            "hone.models.media.commercial_use": caps.commercial_use,
        }
        started = time.monotonic()
        with start_span(f"hone.models.{cfg.kind}", self.sink, attrs, trace=trace) as span:
            job = MediaJob(
                cfg, prompt, named, seed, Path(out), timeout_s or cfg.max_timeout_s, span["attributes"],
                lease=lambda: self._call_lease(trace), session=self._state,
            )  # fmt: skip
            outcome = provider.run(job)
            files = [measure(p) for p in outcome.files]
            error, kind = outcome.error, outcome.error_kind
            if error is None and not files:
                error, kind = "the job finished without output files", "no_output"
            cost = _cost(cfg, files)
            _record(span, files, outcome.job_id, cost, (error, kind or "failed"))
        return MediaResult(
            files, cfg.id, seed, span["span_id"], round(time.monotonic() - started, 3),
            error, kind if error else None, outcome.job_id, cost, cost is not None,
            caps.license, caps.commercial_use,
        )  # fmt: skip

    @contextmanager
    def session(self, *, trace: Mapping[str, str] | None = None) -> Generator[MediaClient]:
        """Hold the GPU lease, the server and the loaded model for the `generate` calls in the block and
        free them once at the end (also on an exception). Sessions of one client do not nest.

            with mk.music("ace-step-1.5-xl-turbo").session() as song:
                for i, seed in enumerate(seeds):
                    song.generate(style, lyrics=text, seed=seed, out=f"takes/take_{i}.flac")
        """
        if self._state is not None:
            raise ConfigError(f"a session of {self.model_id!r} is already open")
        provider = self._provider()
        with (
            self._lease(trace) if self.config.local else nullcontext(),
            provider.session(self.config) as state,
        ):
            self._state = state
            try:
                yield self
            finally:
                self._state = None

    def _provider(self) -> MediaProvider:
        return lookup(MEDIA, self.config, f"{self.config.kind} generation")

    def _lease(self, trace: Mapping[str, str] | None) -> AbstractContextManager[Any]:
        return self.gpu.lease(self.config.id, self.config.capabilities.vram_gb or 1.0, trace=trace)

    def _call_lease(self, trace: Mapping[str, str] | None) -> AbstractContextManager[Any]:
        """A plain call's own lease; none inside a session (it holds one) or for a hosted model."""
        if self._state is not None or not self.config.local:
            return nullcontext()
        return self._lease(trace)

    def _checked(self, provider: MediaProvider, given: dict[str, Any]) -> dict[str, Any]:
        accepted = provider.accepted(self.config)
        unknown = sorted(set(given) - accepted)
        if unknown:
            raise ConfigError(
                f"model {self.model_id!r} does not take the input(s) {unknown}; it takes {sorted(accepted)}"
            )
        named = {k: v for k, v in self.config.defaults.items() if k in accepted} | given
        named = {k: _as_files(k, v) for k, v in named.items()}
        _check_capabilities(self.config, named)
        return named


def _as_files(name: str, value: Any) -> Any:
    """File inputs as `Path`s (a string names a file for `references`, `image` and `source`); every file
    must exist."""
    if name in FILE_INPUTS:
        value = [_path(v) for v in _items(value)] if isinstance(value, list) else _path(value)
    for item in _items(value):
        if isinstance(item, Path) and not item.is_file():
            raise ConfigError(f"input {name!r}: file {item} does not exist")
    return value


def _items(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else [value]


def _path(value: Any) -> Any:
    return Path(value) if isinstance(value, str) else value


def _check_capabilities(cfg: ModelConfig, named: dict[str, Any]) -> None:
    """Refuse what the entry declares impossible (unknown stays allowed)."""
    caps, model = cfg.capabilities, cfg.id
    size = named.get("size")
    if size is not None and not _SIZE.match(str(size)):
        raise ConfigError(f"size must read 'WxH' (e.g. '1024x1024'), got {size!r}")
    if size is not None and caps.sizes is not None and size not in caps.sizes:
        raise CapabilityError(f"model {model!r} makes the sizes {caps.sizes}, not {size!r}")
    duration = named.get("duration_s")
    if duration is not None and caps.durations_s is not None and duration not in caps.durations_s:
        raise CapabilityError(f"model {model!r} makes durations of {caps.durations_s} s, not {duration}")
    if duration is not None and caps.max_duration_s is not None and duration > caps.max_duration_s:
        raise CapabilityError(f"model {model!r} makes at most {caps.max_duration_s} s, not {duration}")
    references: list[Any] = named.get("references") or []
    if caps.max_references is not None and len(references) > caps.max_references:
        raise CapabilityError(f"model {model!r} takes at most {caps.max_references} references")


def _recorded(named: dict[str, Any]) -> dict[str, Any]:
    """The inputs as recorded: files as `{"path", "sha256", "bytes"}`."""

    def one(value: Any) -> Any:
        return file_record(value) if isinstance(value, Path) else value

    return {k: [one(i) for i in _items(v)] if isinstance(v, list) else one(v) for k, v in named.items()}


def _cost(cfg: ModelConfig, files: list[MediaFile]) -> float | None:
    """A naive estimate: one flat price per image or per second of output; `None` without a price."""
    price = cfg.capabilities.price
    if price is None or not files:
        return None
    if price.per_image:
        return round(price.per_image * sum(1 for f in files if (f.mime or "").startswith("image/")), 6)
    seconds = [f.duration_s for f in files if f.duration_s is not None]
    if price.per_output_second and seconds:
        return round(price.per_output_second * sum(seconds), 6)
    return None


def _record(
    span: dict[str, Any],
    files: list[MediaFile],
    job_id: str | None,
    cost: float | None,
    error: tuple[str | None, str],
) -> None:
    attrs = span["attributes"]
    attrs["hone.models.media.outputs"] = [f.record() for f in files]
    attrs["hone.models.media.job_id"] = job_id
    attrs["hone.models.media.cost_estimated"] = cost is not None
    if cost is not None:
        attrs["hone.models.cost_usd"] = cost
    message, kind = error
    if message is not None:
        attrs["hone.models.media.error"] = message
        attrs["hone.models.media.error_kind"] = kind
        span["status"] = {"code": "error", "message": message}


def _client(kind: str, model_id: str, registry: Registry | None, sink: RecordSink | None) -> MediaClient:
    cfg = (registry or load()).get(model_id)
    if cfg.kind != kind:
        raise ConfigError(f"model {cfg.id!r} is a {cfg.kind} model; mk.{kind}() takes {kind} models")
    return MediaClient(cfg, sink or default_sink())


def image(model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None) -> MediaClient:
    """A `MediaClient` for a registered image model (`kind = "image"`)."""
    return _client("image", model_id, registry, sink)


def music(model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None) -> MediaClient:
    """A `MediaClient` for a registered music model (`kind = "music"`)."""
    return _client("music", model_id, registry, sink)


def video(model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None) -> MediaClient:
    """A `MediaClient` for a registered video model (`kind = "video"`)."""
    return _client("video", model_id, registry, sink)
