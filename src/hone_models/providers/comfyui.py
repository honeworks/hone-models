"""ComfyUI: images, music and video from an API-format workflow per registry entry (change 0015 §2).

One job: the workflow is filled with the call's inputs (`_comfyui_workflow`), file inputs are uploaded
once under their SHA-256 name into `input/hone/`, the job is queued with `POST /prompt` (`node_errors`
become a `ConfigError`), `GET /history/{id}` is polled (1 s, then 2 s after the first minute) and the
output files are fetched with `GET /view`. A timeout or any exception while waiting cancels the job
(`POST /api/jobs/{id}/cancel`, else `/queue` delete or `/interrupt`). A plain call ends with `POST /free`;
a session frees once at its end. Every job is noted in the loaded-models file (`_comfyui_loaded`).
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import httpx

from .. import _comfyui_loaded
from .._media_files import file_digest, output_paths
from .._tracing import add_event
from ..errors import ConfigError, ModelTimeout, ProviderError
from ..records import log
from ..registry import ModelConfig
from . import _comfyui_server as server
from . import _comfyui_workflow as workflows
from .common import MediaJob, MediaOutcome, MediaProvider, error_kind

POLL_S = 1.0
POLL_LATE_S = 2.0  # after the first minute
MAX_POLL_ERRORS = 5  # transient polling failures in a row before giving up (the job keeps running)
UPLOAD_FOLDER = "hone"
_uploaded: set[tuple[str, str]] = set()  # (server URL, uploaded name): each file is sent once per process


def run(job: MediaJob) -> MediaOutcome:
    """Run one job at the entry's server and write its files next to `job.out`."""
    cfg, attrs = job.config, job.attrs
    url = cfg.comfyui_url
    workflow, attrs["hone.models.media.workflow_sha256"] = workflows.load(cfg)
    server.require(url)
    server.register_release(url)
    session = job.session
    attrs["hone.models.media.server_started"] = bool(session and session.pop("server_started", False))
    attrs["hone.models.media.loaded"] = cfg.id not in {
        e["model_id"] for e in _comfyui_loaded.read().get(url, [])
    }
    with job.lease():  # freed inside the lease, so the next lease sees the memory back
        try:
            values = {k: _uploaded_names(url, v) for k, v in job.inputs.items()}
            filled = workflows.fill(workflow, cfg, {"prompt": job.prompt, "seed": job.seed, **values})
            job_id, submitted_ms = _submit(url, filled, cfg)
            attrs["hone.models.media.job_id"] = job_id
            _comfyui_loaded.add(url, cfg.id, job_id)
            entry = _wait(url, job_id, time.monotonic() + job.timeout_s)
            _queue_wait(entry, submitted_ms, attrs)
            return _outcome(url, job_id, entry, cfg, job.out)
        finally:
            attrs["hone.models.media.freed"] = session is None and server.free(url)


@contextmanager
def session(cfg: ModelConfig) -> Generator[dict[str, Any]]:
    """Keep the server (started here if need be) and the loaded model for the block; free once at the
    end, then stop the server only if this session started it."""
    url = cfg.comfyui_url
    with server.running(url) as started:
        try:
            yield {"server_started": started}
        finally:
            server.free(url)


def _uploaded_names(url: str, value: Any) -> Any:
    """File inputs replaced by their uploaded names (`hone/<sha256><suffix>`)."""
    if isinstance(value, list):
        return [_uploaded_names(url, v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    if not isinstance(value, Path):
        return value
    name = file_digest(value)[0] + value.suffix.lower()
    if (url, name) not in _uploaded:
        with value.open("rb") as fh:
            files = {"image": (name, fh, "application/octet-stream")}
            form = {"subfolder": UPLOAD_FOLDER, "type": "input", "overwrite": "true"}
            _checked(httpx.post(f"{url}/upload/image", files=files, data=form, timeout=300), "upload")
        _uploaded.add((url, name))
    return f"{UPLOAD_FOLDER}/{name}"


def _submit(url: str, workflow: dict[str, Any], cfg: ModelConfig) -> tuple[str, float]:
    """Queue the job; its id and the wall-clock time (ms) it was sent. `node_errors` -> `ConfigError`."""
    body = {"prompt": workflow, "client_id": f"hone-models-{secrets.token_hex(8)}"}
    sent_ms = time.time() * 1000
    try:
        resp = httpx.post(f"{url}/prompt", json=body, timeout=60)
    except httpx.HTTPError as exc:
        raise ProviderError(f"ComfyUI at {url} did not take the job: {exc}") from exc
    if resp.status_code == 400:
        raise ConfigError(f"model {cfg.id!r}: ComfyUI refused workflow {cfg.workflow}: {_node_errors(resp)}")
    return str(_checked(resp, "queue the job").json()["prompt_id"]), sent_ms


def _node_errors(resp: httpx.Response) -> str:
    """ComfyUI's validation errors, one per node and input."""
    try:
        data: Any = resp.json()
    except ValueError:
        return resp.text[:500]
    found: list[str] = []
    for node, info in data.get("node_errors", {}).items():
        for e in info.get("errors", []):
            details = f": {e['details']}" if e.get("details") else ""
            name = e.get("extra_info", {}).get("input_name")
            found.append(
                f"node {node} ({info.get('class_type')}), input {name!r}: {e.get('message')}{details}"
            )
    error: Any = data.get("error", {})
    return "; ".join(found) or f"{error.get('message')}: {error.get('details', '')}".rstrip(": ")


def _wait(url: str, job_id: str, deadline: float) -> Any:
    """Poll the history until the job is there; cancel it on timeout or any exception."""
    started, errors = time.monotonic(), 0
    try:
        while True:
            try:
                entry = _checked(httpx.get(f"{url}/history/{job_id}", timeout=30), "poll").json().get(job_id)
                errors = 0
            except (httpx.HTTPError, ProviderError, ValueError) as exc:
                errors += 1
                if errors >= MAX_POLL_ERRORS:
                    raise ProviderError(f"lost ComfyUI job {job_id} at {url}: {exc}") from exc
                entry = None
            if entry:
                return entry
            now = time.monotonic()
            if now >= deadline:
                raise ModelTimeout(f"ComfyUI job {job_id} did not finish in time; it was cancelled")
            time.sleep(min(POLL_S if now - started < 60 else POLL_LATE_S, deadline - now))
    except BaseException:
        _cancel(url, job_id)
        raise


def _cancel(url: str, job_id: str) -> None:
    """Stop the job wherever it is; never raises (the caller is already failing)."""
    add_event("cancelled", {"job_id": job_id})
    try:
        if httpx.post(f"{url}/api/jobs/{job_id}/cancel", timeout=10).is_success:
            return
        queue = httpx.get(f"{url}/queue", timeout=10).json()  # an older server
        if any(item[1] == job_id for item in queue.get("queue_running", [])):
            httpx.post(f"{url}/interrupt", json={"prompt_id": job_id}, timeout=10)
        else:
            httpx.post(f"{url}/queue", json={"delete": [job_id]}, timeout=10)
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("hone-models: could not cancel ComfyUI job %s at %s: %s", job_id, url, exc)


def _queue_wait(entry: Any, submitted_ms: float, attrs: dict[str, Any]) -> None:
    for name, data in _messages(entry):
        if name == "execution_start" and "timestamp" in data:
            attrs["hone.models.media.queue_wait_ms"] = max(0, round(data["timestamp"] - submitted_ms))


def _messages(entry: Any) -> list[tuple[str, Any]]:
    """The job's status messages (`execution_start`, `execution_error`, ...) as (name, data)."""
    messages: Any = entry.get("status", {}).get("messages", [])
    return [(str(m[0]), m[1]) for m in messages if len(m) >= 2 and hasattr(m[1], "get")]


def _outcome(url: str, job_id: str, entry: Any, cfg: ModelConfig, out: Path) -> MediaOutcome:
    """The job's files fetched to `out`, or its execution error."""
    for name, data in _messages(entry):
        if name == "execution_error":
            message = f"{data.get('exception_type')}: {data.get('exception_message')}".strip()
            where = f" (node {data.get('node_id')}, {data.get('node_type')})"
            return MediaOutcome([], job_id, message + where, error_kind(message))
        if name == "execution_interrupted":
            return MediaOutcome([], job_id, "the job was interrupted", "failed")
    found = _output_files(entry.get("outputs", {}), cfg.outputs)
    paths = output_paths(out, [Path(f["filename"]).suffix for f in found])
    for item, path in zip(found, paths, strict=True):
        params = {k: item.get(k, "") for k in ("filename", "subfolder", "type")}
        with httpx.stream("GET", f"{url}/view", params=params, timeout=300) as resp:
            _checked(resp, f"fetch {item['filename']}")
            with path.open("wb") as fh:
                for block in resp.iter_bytes():
                    fh.write(block)
    return MediaOutcome(paths, job_id)


def _output_files(outputs: Any, nodes: list[str] | None) -> list[Any]:
    """The files of the `nodes` listed (default: every node's files except temporary previews)."""
    order: list[str] = nodes or sorted(outputs, key=lambda n: (len(n), n))
    found: list[Any] = []
    for node in order:
        for items in outputs.get(node, {}).values():
            listed = cast(list[Any], items) if isinstance(items, list) else []  # files, or flags ("animated")
            files: list[Any] = [i for i in listed if isinstance(i, dict) and "filename" in i]
            found += [i for i in files if nodes or i.get("type") != "temp"]
    return found


def _checked(resp: httpx.Response, what: str) -> httpx.Response:
    if not resp.is_success:
        raise ProviderError(f"ComfyUI could not {what}: HTTP {resp.status_code}", status=resp.status_code)
    return resp


def unload(cfg: ModelConfig) -> None:
    """`mk.unload` for a ComfyUI entry: ComfyUI can only free everything at once (`POST /free`)."""
    if not server.free(cfg.comfyui_url):
        raise ProviderError(f"ComfyUI at {cfg.comfyui_url} did not free its models")


PROVIDER = MediaProvider(accepted=workflows.accepted, run=run, session=session)
