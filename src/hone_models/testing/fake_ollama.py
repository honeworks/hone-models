"""A fake Ollama server on localhost, for running code that uses hone-models without models or a GPU.

    with FakeOllama() as server:                      # sets OLLAMA_HOST while running
        r = mk.text("gemma4-12b").complete([{"role": "user", "content": "hi"}])
        server.requests[-1]["path"]                   # "/api/chat"

Chat replies echo the last user message; with a JSON Schema (`format`) the reply is a sample object that
matches it. Embeddings are stable hash vectors. `/api/show`, `/api/ps`, `/api/version` and unloads answer
too, and so do Ollama's OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings`.

`server.queue(path, *answers)` scripts the next answers for one path (truncation, thinking-only replies,
bad JSON, server errors):

    server.queue("/api/chat", {"done_reason": "length"}, 503)

`FakeOllama(responder=fn)` computes answers from the request instead (by schema title, by prompt, ...):
`fn(path, body)` returns a dict merged over the default reply, or `None` for the default reply.
Queued answers still come first.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from ..providers.common import text_of

DIMENSIONS = 8
Responder = Callable[[str, dict[str, Any]], "dict[str, Any] | None"]  # (path, JSON body) -> reply override


def sample(schema: dict[str, Any], root: dict[str, Any] | None = None) -> Any:  # noqa: PLR0911 - one return per schema keyword
    """A value matching a simple JSON Schema: first enum value / `anyOf` branch, a number in range, "ok",
    true; `$ref`s to `#/$defs/...` are followed."""
    root = root or schema
    if "$ref" in schema:
        return sample(root["$defs"][schema["$ref"].rsplit("/", 1)[-1]], root)
    if "anyOf" in schema:
        return sample(schema["anyOf"][0], root)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {k: sample(v, root) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return [sample(schema.get("items", {}), root)]
    if kind in ("number", "integer"):
        low = schema.get("minimum", schema.get("maximum", 1))
        high = schema.get("maximum", low)
        middle = (low + high) / 2
        return round(middle) if kind == "integer" else middle
    if kind == "boolean":
        return True
    return "ok"


def vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255 + 0.01 for b in digest[:DIMENSIONS]]


def _reply_text(messages: list[dict[str, Any]], schema: Any) -> str:
    """The echo of the last user message, or a sample object for a schema."""
    if schema is not None:
        return json.dumps(sample(schema))
    users = [text_of(m["content"]) for m in messages if m["role"] == "user"]
    return f"echo: {users[-1] if users else ''}"


def _prompt_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(len(text_of(m["content"])) // 4 for m in messages)


def _ollama_chat(body: dict[str, Any]) -> dict[str, Any]:
    content = _reply_text(body.get("messages", []), body.get("format"))
    return {
        "model": body.get("model"),
        "message": {"role": "assistant", "content": content},
        "done_reason": "stop",
        "prompt_eval_count": _prompt_tokens(body.get("messages", [])),
        "eval_count": len(content) // 4 + 1,
    }


def _openai_chat(body: dict[str, Any]) -> dict[str, Any]:
    response_format: dict[str, Any] = body.get("response_format") or {}
    schema = response_format.get("json_schema", {}).get("schema")
    content = _reply_text(body.get("messages", []), schema)
    return {
        "model": body.get("model"),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": _prompt_tokens(body.get("messages", [])),
            "completion_tokens": len(content) // 4 + 1,
        },
    }


def _ollama_embed(body: dict[str, Any]) -> dict[str, Any]:
    return {"model": body.get("model"), "embeddings": [vector(t) for t in body.get("input", [])]}


def _openai_embed(body: dict[str, Any]) -> dict[str, Any]:
    rows = [{"index": i, "embedding": vector(t)} for i, t in enumerate(body.get("input", []))]
    return {"model": body.get("model"), "data": rows}


_REPLIES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "/api/chat": _ollama_chat,
    "/api/embed": _ollama_embed,
    "/api/show": lambda b: {"capabilities": ["completion"], "model_info": {"fake.context_length": 8192}},
    "/api/ps": lambda b: {"models": []},
    "/v1/chat/completions": _openai_chat,
    "/v1/embeddings": _openai_embed,
}


def reply(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """The default JSON answer for one request (`/api/version` and unloads get a plain acknowledgement)."""
    handler = _REPLIES.get(path)
    return handler(body) if handler else {"version": "0.0.0-fake", "done": True}


class FakeOllama:
    """Serves the fake API on a free local port; `url` is its address, `requests` what it received
    (`path` without the query string, JSON `body`, `headers`).

    `responder(path, body)` answers requests itself: a dict is merged over the default reply (top-level
    keys replaced), `None` keeps the default. Use it as a context manager, or call `start()` / `stop()`.
    """

    def __init__(self, responder: Responder | None = None) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self._queued: dict[str, list[dict[str, Any] | int]] = {}
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                status, data = server._respond(urlsplit(self.path).path, body, dict(self.headers))
                raw = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _answer  # noqa: N815 - http.server's method names

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._handler = Handler
        self.url = ""
        self._previous: str | None = None

    def queue(self, path: str, *answers: dict[str, Any] | int) -> None:
        """Answer the next requests to `path` with `answers`, in order; then the default replies resume.

        A dict is merged over the default reply (top-level keys replaced), an int is an HTTP error status.
        """
        with self._lock:
            self._queued.setdefault(path, []).extend(answers)

    def _respond(
        self, path: str, body: dict[str, Any], headers: dict[str, str]
    ) -> tuple[int, dict[str, Any]]:
        with self._lock:
            self.requests.append({"path": path, "body": body, "headers": headers})
            queued = self._queued.get(path)
            scripted = queued.pop(0) if queued else None
        if isinstance(scripted, int):
            return scripted, {"error": f"fake error {scripted}"}
        answered = self.responder(path, body) if self.responder else None
        return 200, reply(path, body) | (answered or {}) | (scripted or {})

    def start(self) -> FakeOllama:
        """Start serving and point `OLLAMA_HOST` at this server."""
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        self._previous = os.environ.get("OLLAMA_HOST")
        os.environ["OLLAMA_HOST"] = self.url
        return self

    def stop(self) -> None:
        """Stop serving and restore the previous `OLLAMA_HOST`."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._previous is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = self._previous

    def __enter__(self) -> FakeOllama:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
