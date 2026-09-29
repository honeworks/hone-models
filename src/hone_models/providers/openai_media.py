"""OpenAI-compatible images and video (change 0015 §2): the `openai_compatible` provider's generation row.

Images: `POST {base_url}/images/generations` (JSON), or with `references` `POST /images/edits` as
multipart with one `image[]` part per reference. A `b64_json` answer is decoded straight to the output
file, a `url` answer downloaded (without the API key: the URL is another host's). `revised_prompt` is
recorded. Video: `POST /videos` (multipart with `input_reference` when `image` is given), then
`GET /videos/{id}` every `POLL_S` seconds (a `progress` event per 10 %), then `GET /videos/{id}/content`.

Submitting uses the retries of `_http` (connection errors, 429, 5xx); a submitted video job is never
submitted again: polling tolerates `MAX_POLL_ERRORS - 1` transient errors in a row, then raises
`ProviderError` with the job id. A timeout, or any exception while waiting, deletes the job
(`DELETE /videos/{id}`). A moderation refusal (at submission, or a failed job) and a failed job are
results (`error`, `error_kind`), not exceptions. Sizes, durations and the cost estimate are the media
client's (`media.py`); nothing here takes a GPU lease unless the server is on this machine.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

import httpx

from .._http import bearer_headers, request_json
from .._media_files import output_paths
from .._tracing import add_event
from ..errors import ConfigError, ModelTimeout, ProviderError
from ..records import log
from ..registry import ModelConfig
from .common import MediaJob, MediaOutcome, MediaProvider, error_kind

POLL_S = 5.0
MAX_POLL_ERRORS = 5  # transient polling failures in a row before giving up (the job keeps running)
SUBMIT_TIMEOUT_S = 120.0  # submitting a video job; an image request waits for the image (`timeout_s`)
FETCH_TIMEOUT_S = 300.0
TERMINAL = ("completed", "failed", "cancelled")
# Named inputs per kind besides `prompt` and `seed`; an entry lists more (`quality`, `background`, ...).
KIND_INPUTS = {"image": {"size", "n", "references"}, "video": {"size", "duration_s", "image"}}


def accepted(cfg: ModelConfig) -> set[str]:
    """The named inputs of a hosted image or video entry: its kind's plus the extra names it lists."""
    if cfg.kind not in KIND_INPUTS:
        raise ConfigError(f"model {cfg.id!r}: openai_compatible makes images and video, not {cfg.kind}")
    extra = cfg.inputs if isinstance(cfg.inputs, list) else []
    return {*KIND_INPUTS[cfg.kind], *extra}


def run(job: MediaJob) -> MediaOutcome:
    """One image request or one video job; its files are written next to `job.out`."""
    with job.lease():  # a no-op for a hosted model
        return _video(job) if job.config.kind == "video" else _images(job)


def _images(job: MediaJob) -> MediaOutcome:
    cfg, fields = job.config, dict(job.inputs)
    references: list[Path] = fields.pop("references", None) or []
    fields = {"model": cfg.name, "prompt": job.prompt, **fields}
    if references:
        parts = [("image[]", _part(p)) for p in references]
        answer = _submit(cfg, "/images/edits", job.timeout_s, form=_form(fields), files=parts)
    else:
        answer = _submit(cfg, "/images/generations", job.timeout_s, payload=fields)
    if isinstance(answer, MediaOutcome):
        return answer
    items: list[dict[str, Any]] = answer.get("data") or []
    revised = list(dict.fromkeys(str(i["revised_prompt"]) for i in items if i.get("revised_prompt")))
    if revised:
        job.attrs["hone.models.media.revised_prompt"] = "\n".join(revised)
    suffix = "." + str(fields.get("output_format", "png")).replace("jpeg", "jpg")
    paths = output_paths(job.out, [suffix] * len(items))
    for item, path in zip(items, paths, strict=True):
        _write_image(item, path)
    return MediaOutcome(paths)


def _write_image(item: dict[str, Any], path: Path) -> None:
    if item.get("b64_json"):
        path.write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        _fetch(str(item["url"]), path, None, "an image the provider returned")
    else:
        raise ProviderError(f"the image answer has neither b64_json nor url: {sorted(item)}")


def _video(job: MediaJob) -> MediaOutcome:
    cfg, fields = job.config, dict(job.inputs)
    start: Path | None = fields.pop("image", None)
    if "duration_s" in fields:
        fields["seconds"] = _seconds(fields.pop("duration_s"))
    fields = {"model": cfg.name, "prompt": job.prompt, **fields}
    if start is not None:
        parts = [("input_reference", _part(start))]
        answer = _submit(cfg, "/videos", SUBMIT_TIMEOUT_S, form=_form(fields), files=parts)
    else:
        answer = _submit(cfg, "/videos", SUBMIT_TIMEOUT_S, payload=fields)
    if isinstance(answer, MediaOutcome):
        return answer
    if not answer.get("id"):
        raise ProviderError(f"{cfg.base_url}/videos returned no job id: {str(answer)[:200]}")
    job_id = str(answer["id"])
    job.attrs["hone.models.media.job_id"] = job_id
    final = _wait(cfg, job_id, time.monotonic() + job.timeout_s)
    if final.get("status") != "completed":
        message = _job_error(final)
        return MediaOutcome([], job_id, message, error_kind(message))
    (path,) = output_paths(job.out, [".mp4"])
    _fetch(_url(cfg, f"/videos/{job_id}/content"), path, bearer_headers(cfg), f"video job {job_id}")
    return MediaOutcome([path], job_id)


def _seconds(value: Any) -> str:
    """`duration_s` as the API's `seconds` field: a string, `"8"` rather than `"8.0"`."""
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def _job_error(job: dict[str, Any]) -> str:
    error: dict[str, Any] = job.get("error") or {}
    code, message = error.get("code"), error.get("message")
    text = ": ".join(str(p) for p in (code, message) if p)
    return text or f"the video job ended with status {job.get('status')!r}"


def _wait(cfg: ModelConfig, job_id: str, deadline: float) -> dict[str, Any]:
    """Poll the job until it ends; delete it on timeout or any exception."""
    url, errors, shown = _url(cfg, f"/videos/{job_id}"), 0, -1
    try:
        while True:
            try:
                state = _poll(url, cfg)
                errors = 0
            except (httpx.HTTPError, ValueError, _BusyError) as exc:
                errors += 1
                if errors >= MAX_POLL_ERRORS:
                    raise ProviderError(
                        f"lost track of video job {job_id} after {errors} failed polls ({exc}); it may "
                        f"still finish: fetch it later with GET /videos/{job_id}"
                    ) from exc
                state = {}
            shown = _progress(state, shown)
            if state.get("status") in TERMINAL:
                return state
            now = time.monotonic()
            if now >= deadline:
                raise ModelTimeout(f"video job {job_id} did not finish in time; it was cancelled")
            time.sleep(min(POLL_S, deadline - now))
    except BaseException:
        _cancel(cfg, job_id)
        raise


class _BusyError(Exception):
    """A 429 or 5xx while polling: try again."""


def _poll(url: str, cfg: ModelConfig) -> dict[str, Any]:
    resp = httpx.get(url, headers=bearer_headers(cfg), timeout=30)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise _BusyError(f"HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise ProviderError(
            f"{url} returned HTTP {resp.status_code}: {resp.text[:500]}", status=resp.status_code
        )
    return resp.json()


def _progress(state: dict[str, Any], shown: int) -> int:
    """Add a `progress` event when the job passed another 10 %; the percent last shown."""
    percent = state.get("progress")
    if isinstance(percent, int | float) and percent // 10 > shown // 10:
        add_event("progress", {"percent": percent, "status": state.get("status")})
        return int(percent)
    return shown


def _cancel(cfg: ModelConfig, job_id: str) -> None:
    """Delete the job at the provider; never raises (the caller is already failing)."""
    add_event("cancelled", {"job_id": job_id})
    try:
        httpx.delete(_url(cfg, f"/videos/{job_id}"), headers=bearer_headers(cfg), timeout=30)
    except (httpx.HTTPError, ConfigError) as exc:
        log.warning("hone-models: could not delete video job %s: %s", job_id, exc)


def _submit(cfg: ModelConfig, path: str, timeout: float, **body: Any) -> Any:
    """Send the request (retrying transient failures); a moderation refusal is a `MediaOutcome`."""
    try:
        return request_json("POST", _url(cfg, path), headers=bearer_headers(cfg), timeout=timeout, **body)
    except ProviderError as exc:
        message = str(exc)
        if exc.status is not None and 400 <= exc.status < 500 and error_kind(message) == "refused":
            return MediaOutcome([], None, message, "refused")
        raise


def _fetch(url: str, path: Path, headers: dict[str, str] | None, what: str) -> None:
    """Stream a file to `path`. The URL is not in the message: a signed URL is a credential."""
    try:
        with httpx.stream(
            "GET", url, headers=headers, timeout=FETCH_TIMEOUT_S, follow_redirects=True
        ) as resp:
            if not resp.is_success:
                raise ProviderError(
                    f"could not fetch {what}: HTTP {resp.status_code}", status=resp.status_code
                )
            with path.open("wb") as fh:
                for block in resp.iter_bytes():
                    fh.write(block)
    except httpx.HTTPError as exc:
        raise ProviderError(f"could not fetch {what}: {type(exc).__name__}") from exc


def _url(cfg: ModelConfig, path: str) -> str:
    if not cfg.base_url:
        raise ConfigError(
            f"model {cfg.id!r} (openai_compatible) needs base_url, e.g. https://api.openai.com/v1"
        )
    return cfg.base_url.rstrip("/") + path


def _form(fields: dict[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in fields.items()}


def _part(path: Path) -> tuple[str, bytes, str]:
    """A file as a multipart part: bytes in memory, so a retry can send it again."""
    return path.name, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"


PROVIDER = MediaProvider(accepted=accepted, run=run)
