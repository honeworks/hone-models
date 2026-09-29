"""A fake ComfyUI server on localhost, for running generation code without ComfyUI, models or a GPU.

    with FakeComfyUI() as server:                     # sets HONE_COMFYUI_URL while running
        r = mk.image("my-image-model").generate("a lighthouse", out="shot.png")
        server.submitted[-1]["6"]["inputs"]["text"]   # "a lighthouse": the filled workflow
        server.frees                                  # 1: a plain call frees ComfyUI afterwards

It answers `/system_stats`, `/prompt` (links to missing nodes come back as `node_errors`),
`/history[/{id}]`, `/api/jobs/{id}` and `/cancel`, `/interrupt`, `/queue`, `/upload/image`, `/view` and
`/free`. A job "runs" for `run_s` seconds and then writes, for each save node of its workflow, a black
PNG (`SaveImage`: the size of the workflow's `width` / `height` inputs), a WAV of silence (`SaveAudio*`:
as long as its `seconds` or `duration` input) or a tiny MP4 (`SaveVideo`, `VHS_VideoCombine`).

`server.queue(outcome, **details)` scripts the next jobs: `"node_errors"` (`node=`, `input=`,
`message=`), `"execution_error"` (`message=`, `type=`: e.g. CUDA out of memory), `"hang"` (never ends),
`"no_output"`, or `"ok"` with its own `run_s=`. `requests` records every request, `submitted` every
filled workflow, `uploads` the uploaded files by name. `FakeComfyUI(jobs_api=False)` acts like an older
server without `/api/jobs`. `python -m hone_models.testing.fake_comfyui --port 8188` serves one in the
foreground (a stand-in for `HONE_COMFYUI_START`).
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import _fake_files

SAVE_NODES = {  # save node class -> the key of its files in the history
    "SaveImage": "images",
    "SaveAudio": "audio", "SaveAudioMP3": "audio", "SaveAudioOpus": "audio",
    "SaveVideo": "gifs", "VHS_VideoCombine": "gifs",
}  # fmt: skip
Reply = tuple[int, Any]  # (status, JSON value or bytes)
SYSTEM_STATS = {"system": {"comfyui_version": "fake"}, "devices": [{"name": "fake", "torch_vram_total": 0}]}


@dataclass
class _Job:
    id: str
    workflow: dict[str, Any]
    outcome: str
    details: dict[str, Any]
    done_at: float | None  # monotonic time; None: never ends
    started_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    cancelled: bool = False
    history: dict[str, Any] | None = None


class FakeComfyUI:
    """Serves the fake API on a local port (`port=0`: a free one); `url` is its address."""

    def __init__(self, *, run_s: float = 0.0, port: int = 0, jobs_api: bool = True) -> None:
        self.run_s, self.port, self.jobs_api = run_s, port, jobs_api
        self.requests: list[dict[str, Any]] = []
        self.submitted: list[dict[str, Any]] = []
        self.uploads: dict[str, bytes] = {}
        self.url = ""
        self._jobs: dict[str, _Job] = {}
        self._scripted: list[tuple[str, dict[str, Any]]] = []
        self._files: dict[tuple[str, str, str], bytes] = {}
        self._lock = threading.Lock()
        self._previous: str | None = None

    @property
    def frees(self) -> int:
        """How many `POST /free` requests arrived."""
        return sum(1 for r in self.requests if r["path"] == "/free")

    def queue(self, outcome: str, **details: Any) -> None:
        """Script the next job: `ok`, `node_errors`, `execution_error`, `hang` or `no_output`."""
        if outcome not in ("ok", "node_errors", "execution_error", "hang", "no_output"):
            raise ValueError(f"unknown outcome {outcome!r}")
        with self._lock:
            self._scripted.append((outcome, details))

    def respond(self, method: str, target: str, body: bytes, content_type: str) -> Reply:
        """One request: recorded, then answered."""
        parts = urlsplit(target)
        path, query = parts.path, {k: v[0] for k, v in parse_qs(parts.query).items()}
        data: Any = json.loads(body) if body and "json" in content_type else None
        with self._lock:
            self.requests.append({"method": method, "path": path, "query": query, "body": data})
            return self._route(path, query, data, (body, content_type))

    def _route(self, path: str, query: dict[str, str], data: Any, raw: tuple[bytes, str]) -> Reply:
        head, _, rest = path.strip("/").partition("/")
        routes: dict[str, Any] = {
            "system_stats": lambda: (200, SYSTEM_STATS),
            "prompt": lambda: self._submit(data["prompt"]),
            "history": lambda: self._histories(rest),
            "api": lambda: self._job_api(rest),
            "queue": lambda: self._queue(data or {}),
            "interrupt": self._interrupt,
            "upload": lambda: self._upload(*raw),
            "view": lambda: self._view(query),
            "free": lambda: (200, b""),
        }
        route = routes.get(head)
        return route() if route else (404, {"error": f"no route {path}"})

    def _histories(self, job_id: str) -> Reply:
        ids = [job_id] if job_id else list(self._jobs)
        return 200, {i: h for i in ids if (h := self._history(i)) is not None}

    def _queue(self, data: Any) -> Reply:
        running: list[Any] = [
            [n, j.id, {}, {}, []] for n, j in enumerate(self._jobs.values()) if self._running(j)
        ]
        for job_id in data.get("delete", []):
            self._cancel(str(job_id))
        return 200, {"queue_running": running, "queue_pending": []}

    def _interrupt(self) -> Reply:
        for job in [j for j in self._jobs.values() if self._running(j)]:
            self._cancel(job.id)
        return 200, {}

    def _view(self, query: dict[str, str]) -> Reply:
        found = self._files.get(
            (query.get("subfolder", ""), query.get("filename", ""), query.get("type", ""))
        )
        return (200, found) if found is not None else (404, {"error": "not found"})

    def _submit(self, workflow: Any) -> Reply:
        outcome, details = self._scripted.pop(0) if self._scripted else ("ok", dict[str, Any]())
        errors = self._validation_errors(workflow)
        if outcome == "node_errors":
            node = str(details.get("node", "1"))
            errors[node] = [(details.get("input"), details.get("message", "bad value"))]
        if errors:
            return 400, _refusal(workflow, errors)
        self.submitted.append(workflow)
        run_s = float(details.get("run_s", self.run_s))
        done_at = None if outcome == "hang" else time.monotonic() + run_s
        job = _Job(str(uuid.uuid4()), workflow, outcome, details, done_at)
        self._jobs[job.id] = job
        return 200, {"prompt_id": job.id, "number": len(self._jobs), "node_errors": {}}

    @staticmethod
    def _validation_errors(workflow: dict[str, Any]) -> dict[str, list[tuple[str | None, str]]]:
        """Inputs linked to nodes that are not in the workflow."""
        errors: dict[str, list[tuple[str | None, str]]] = {}
        for node, spec in workflow.items():
            for name, value in spec.get("inputs", {}).items():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) not in workflow:  # pyright: ignore[reportUnknownArgumentType]
                    errors.setdefault(node, []).append((name, f"linked node {value[0]} does not exist"))
        return errors

    def _running(self, job: _Job) -> bool:
        return not job.cancelled and (job.done_at is None or time.monotonic() < job.done_at)

    def _cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or not self._running(job):
            return False
        job.cancelled = True
        return True

    def _job_api(self, rest: str) -> Reply:
        """`jobs/<id>` and `jobs/<id>/cancel`; 404 on a server without them (`jobs_api=False`)."""
        parts = rest.split("/")
        job = self._jobs.get(parts[1]) if len(parts) > 1 and parts[0] == "jobs" else None
        if not self.jobs_api or job is None:
            return 404, {"error": "Job not found"}
        if parts[-1] == "cancel":
            return 200, {"cancelled": self._cancel(job.id)}
        status = "cancelled" if job.cancelled else "in_progress" if self._running(job) else "completed"
        return 200, {"id": job.id, "status": status}

    def _history(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        if job is None or job.cancelled or self._running(job):
            return None
        if job.history is None:
            job.history = self._finish(job)
        return job.history

    def _finish(self, job: _Job) -> dict[str, Any]:
        messages: list[Any] = [["execution_start", {"prompt_id": job.id, "timestamp": job.started_ms}]]
        if job.outcome == "execution_error":
            node = next(iter(job.workflow))
            error = {
                "prompt_id": job.id, "node_id": node, "node_type": job.workflow[node]["class_type"],
                "exception_type": job.details.get("type", "RuntimeError"),
                "exception_message": job.details.get("message", "the fake job failed"),
            }  # fmt: skip
            messages.append(["execution_error", error])
            return {
                "outputs": {},
                "status": {"status_str": "error", "completed": False, "messages": messages},
            }
        outputs = {} if job.outcome == "no_output" else self._outputs(job.workflow)
        messages.append(["execution_success", {"prompt_id": job.id}])
        return {
            "outputs": outputs,
            "status": {"status_str": "success", "completed": True, "messages": messages},
        }

    def _outputs(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """One file per save node, stored for `/view`."""
        values = {k: v for spec in workflow.values() for k, v in spec.get("inputs", {}).items()}
        outputs: dict[str, Any] = {}
        for node, spec in workflow.items():
            key = SAVE_NODES.get(spec.get("class_type"))
            if key is None:
                continue
            data, ext = _fake_files.for_save_node(key, values)
            prefix = str(spec.get("inputs", {}).get("filename_prefix", "ComfyUI"))
            subfolder, stem = posixpath.split(prefix)
            name = f"{stem}_{len(self._files) + 1:05d}_.{ext}"
            self._files[(subfolder, name, "output")] = data
            outputs[node] = {key: [{"filename": name, "subfolder": subfolder, "type": "output"}]}
        return outputs

    def _upload(self, body: bytes, content_type: str) -> Reply:
        fields = _fake_files.form_fields(body, content_type)
        filename, data = str(fields["image"][0]), fields["image"][1]
        subfolder = fields.get("subfolder", (None, b""))[1].decode()
        self.uploads[posixpath.join(subfolder, filename)] = data
        self._files[(subfolder, filename, "input")] = data
        return 200, {"name": filename, "subfolder": subfolder, "type": "input"}

    def start(self) -> FakeComfyUI:
        """Start serving and point `HONE_COMFYUI_URL` at this server."""
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), _handler(self))
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        self._previous = os.environ.get("HONE_COMFYUI_URL")
        os.environ["HONE_COMFYUI_URL"] = self.url
        return self

    def stop(self) -> None:
        """Stop serving and restore the previous `HONE_COMFYUI_URL`."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._previous is None:
            os.environ.pop("HONE_COMFYUI_URL", None)
        else:
            os.environ["HONE_COMFYUI_URL"] = self._previous

    def __enter__(self) -> FakeComfyUI:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _refusal(workflow: Any, errors: dict[str, list[tuple[str | None, str]]]) -> dict[str, Any]:
    """ComfyUI's answer to a workflow that fails validation."""
    node_errors: dict[str, Any] = {}
    for node, found in errors.items():
        listed = [
            {"type": "value_not_valid", "message": m, "details": "", "extra_info": {"input_name": i}}
            for i, m in found
        ]
        class_type = workflow[node]["class_type"] if node in workflow else None
        node_errors[node] = {"class_type": class_type, "dependent_outputs": [], "errors": listed}
    error = {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"}
    return {"error": error, "node_errors": node_errors}


def _handler(server: FakeComfyUI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            status, data = server.respond(self.command, self.path, body, self.headers.get("Content-Type", ""))
            raw = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header(
                "Content-Type", "application/octet-stream" if isinstance(data, bytes) else "application/json"
            )
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = _answer  # noqa: N815 - http.server's method names

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def main(argv: list[str] | None = None) -> None:
    """Serve a `FakeComfyUI` in the foreground until the process is stopped."""
    parser = argparse.ArgumentParser(description="A fake ComfyUI server for tests and examples.")
    parser.add_argument("--port", type=int, default=8188)
    port = parser.parse_args(argv).port
    ThreadingHTTPServer(("127.0.0.1", port), _handler(FakeComfyUI(port=port))).serve_forever()


if __name__ == "__main__":
    main()
