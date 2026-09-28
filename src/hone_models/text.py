"""Text / chat client: `mk.text(...)` implements the text client shape (design/current.md §10).

llm = mk.text("gemma4-12b")
r = llm.complete([{"role": "user", "content": "Write a haiku about rain."}])
r.text, r.usage, r.finish_reason
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ._http import register_key
from ._tracing import start_span
from .budget import min_context, output_budget, plan_context, timeout_for
from .errors import CapabilityError, ConfigError
from .ports import RecordSink
from .prompt import Prompt
from .providers import CHAT, PROBE, lookup, model_attributes
from .providers.common import ChatReply, ChatRequest, images_of
from .records import default_sink
from .registry import ModelConfig, Registry, load, unmet
from .structured import Outcome, Schema, run_structured, schema_dict, with_schema_instruction

REQUEST_ATTRS = ("temperature", "top_p", "seed")


@dataclass(slots=True)
class TextResult:
    """Outcome of one `complete` call. `error` is set instead of raising for model-quality problems."""

    text: str = ""
    parsed: Any = None
    error: str | None = None
    model: str = ""
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict[str, int])
    structured_path: str | None = None
    attempts: int = 0
    span_id: str | None = None
    logprobs: list[dict[str, Any]] | None = None


class TextClient:
    """Calls one chat model; checks capabilities first and records every call as a span."""

    def __init__(self, config: ModelConfig, sink: RecordSink, *, probe: bool = False) -> None:
        self.config = config
        self.sink = sink
        self._probe = probe and config.provider in PROBE

    @property
    def model_id(self) -> str:
        return self.config.id

    def _resolved(self) -> ModelConfig:
        """The config, with capabilities probed once for ad-hoc ids (Ollama `/api/show`, LiteLLM's map)."""
        if self._probe:
            self.config = self.config.model_copy(
                update={"capabilities": PROBE[self.config.provider](self.config)}
            )
            self._probe = False
        return self.config

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]] | Prompt,
        *,
        schema: Schema | None = None,
        trace: Mapping[str, str] | None = None,
        **params: Any,
    ) -> TextResult:
        """Send `messages` (or a `Prompt`); with `schema` (Pydantic class or JSON Schema dict)
        fill `result.parsed`.

        Params: `temperature`, `top_p`, `seed`, `max_tokens`, `stop`, `think` (reasoning on/off),
        `logprobs`; unknown params are ignored. Raises `CapabilityError` / `ContextOverflow` before
        calling, `ProviderError` on transport failures.
        """
        msgs = messages.messages() if isinstance(messages, Prompt) else [dict(m) for m in messages]
        params = {**self.config.defaults, **params}
        json_schema = schema_dict(schema)
        attrs = request_attributes(self.config, msgs, params, json_schema)
        if isinstance(messages, Prompt):
            attrs.update(messages.attributes())
        with start_span("hone.models.chat", self.sink, attrs, trace=trace) as span:
            result = self._run(msgs, schema, json_schema, params, span["attributes"])
            result.span_id = span["span_id"]
            if result.error:
                span["status"] = {"code": "error", "message": result.error}
        return result

    def _run(
        self,
        msgs: list[dict[str, Any]],
        schema: Schema | None,
        json_schema: dict[str, Any] | None,
        params: dict[str, Any],
        attrs: dict[str, Any],
    ) -> TextResult:
        cfg = self._resolved()
        check_images(cfg, msgs)
        request = build_request(cfg, msgs, json_schema, params, attrs)
        chat = lookup(CHAT, cfg, "chat")
        min_ctx = min_context(params)

        def send(messages: list[dict[str, Any]]) -> ChatReply:
            # structured retries grow the conversation: re-check it still fits
            _, ctx, _ = plan_context(cfg, messages, request.params["max_tokens"], min_ctx)
            return chat(cfg, replace(request, messages=messages, num_ctx=ctx))

        if schema is None:
            reply = send(request.messages)
            out = Outcome(reply, None, empty_error(reply), None, 1, reply.usage)
        else:
            out = run_structured(send, request.messages, schema, constrained=request.schema is not None)
            if out.parsed is None:
                out.error = empty_error(out.reply) or out.error
        reply = out.reply
        result = TextResult(
            text=reply.text,
            parsed=out.parsed,
            error=out.error,
            model=reply.model,
            finish_reason=reply.finish_reason,
            usage=out.usage,
            structured_path=out.path,
            attempts=out.attempts,
            logprobs=reply.logprobs,
        )
        attrs.update(result_attributes(cfg, result, reply.digest))
        return result


def build_request(
    cfg: ModelConfig,
    msgs: list[dict[str, Any]],
    json_schema: dict[str, Any] | None,
    params: dict[str, Any],
    attrs: dict[str, Any],
) -> ChatRequest:
    """The provider request: schema instruction, output budget, context size, images inlined.
    Records the budget numbers in `attrs`; raises `ContextOverflow` when the prompt cannot fit."""
    sent = with_schema_instruction(msgs, json_schema) if json_schema else msgs
    max_tokens = output_budget(cfg, params)
    attrs["gen_ai.request.max_tokens"] = max_tokens
    attrs["hone.models.context.limit"] = cfg.capabilities.max_input_tokens  # recorded even on overflow
    estimated, num_ctx, images = plan_context(cfg, sent, max_tokens, min_context(params))
    attrs["hone.models.context.estimated_prompt_tokens"] = estimated
    if images:
        attrs["hone.models.context.estimated_image_tokens"] = images
    constrained = json_schema is not None and cfg.capabilities.json_schema is True
    return ChatRequest(
        messages=load_images(sent),
        params={k: v for k, v in params.items() if k not in ("think", "logprobs", "min_num_ctx")}
        | {"max_tokens": max_tokens},
        timeout_s=timeout_for(cfg, max_tokens),
        schema=json_schema if constrained else None,
        num_ctx=num_ctx,
        think=params.get("think", False if cfg.capabilities.thinking else None),
        logprobs=bool(params.get("logprobs")),
    )


def empty_error(reply: ChatReply) -> str | None:
    """An explicit error for an empty reply (never a silent empty string)."""
    if reply.text.strip():
        return None
    if reply.thinking:
        return "model returned only 'thinking' text and no content; raise max_tokens or pass think=False"
    return "model returned an empty response"


def check_images(cfg: ModelConfig, messages: Sequence[Mapping[str, Any]]) -> None:
    """Raise `CapabilityError` if images go to a model declared without vision."""
    has_images = any(images_of(m.get("content")) for m in messages)
    if has_images and cfg.capabilities.vision is False:
        raise CapabilityError(
            f"model {cfg.id!r} cannot read images (capabilities.vision = false); "
            "use a vision model, e.g. mk.text(require={'vision': True})"
        )


def load_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace `{"type": "image", "path": ...}` parts with inline base64 data."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        parts: list[Any] = content if isinstance(content, list) else []  # pyright: ignore[reportUnknownVariableType]
        out.append({**msg, "content": [inline_image(p) for p in parts]} if parts else msg)
    return out


def inline_image(part: Any) -> Any:
    """An image part with a `path` as one with `data_b64` + `mime`; other parts unchanged."""
    if not isinstance(part, dict):
        return part
    fields: dict[str, Any] = part  # pyright: ignore[reportUnknownVariableType]
    if fields.get("type") != "image" or "data_b64" in fields:
        return fields
    if "path" not in fields:
        raise ConfigError(
            'an image part needs "path" or "data_b64" + "mime": {"type": "image", "path": "a.png"}'
        )
    path = Path(str(fields["path"]))
    if not path.is_file():
        raise ConfigError(f"image {path} does not exist")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {"type": "image", "data_b64": base64.b64encode(path.read_bytes()).decode("ascii"), "mime": mime}


def request_attributes(
    cfg: ModelConfig, messages: list[dict[str, Any]], params: Mapping[str, Any], json_schema: Any
) -> dict[str, Any]:
    """Span attributes known before the call (design/current.md §8.3)."""
    attrs: dict[str, Any] = {
        **model_attributes(cfg, "chat"),
        "gen_ai.input.messages": messages,
        "hone.models.request.params": dict(params),
    }
    attrs.update({f"gen_ai.request.{k}": params[k] for k in REQUEST_ATTRS if k in params})
    if json_schema is not None:
        attrs["hone.models.structured.schema"] = json_schema
    return attrs


def result_attributes(cfg: ModelConfig, r: TextResult, digest: str | None = None) -> dict[str, Any]:
    """Span attributes known after the call."""
    attrs: dict[str, Any] = {
        "gen_ai.response.model": r.model,
        "gen_ai.response.finish_reasons": [r.finish_reason] if r.finish_reason else [],
        "gen_ai.output.messages": [{"role": "assistant", "content": r.text}],
    }
    attrs.update(usage_attributes(cfg, r.usage))
    if r.structured_path is not None:
        attrs["hone.models.structured.path"] = r.structured_path
        attrs["hone.models.structured.attempts"] = r.attempts
    if digest:
        attrs["hone.models.model_digest"] = digest
    return attrs


def usage_attributes(cfg: ModelConfig, usage: Mapping[str, int]) -> dict[str, Any]:
    """`gen_ai.usage.*` token counts and `hone.models.cost_usd` when the price is known."""
    attrs: dict[str, Any] = {
        f"gen_ai.usage.{k}": usage[k] for k in ("input_tokens", "output_tokens") if k in usage
    }
    cost = cost_usd(cfg, usage)
    if cost is not None:
        attrs["hone.models.cost_usd"] = cost
    return attrs


def cost_usd(cfg: ModelConfig, usage: Mapping[str, int]) -> float | None:
    """Cost from the registry price; `None` without a price or usage, or when a priced token count is
    missing (a side priced 0 needs no count, e.g. Jev's output)."""
    price = cfg.capabilities.price
    if price is None or not usage:
        return None
    rates = {"input_tokens": price.input_per_mtok, "output_tokens": price.output_per_mtok}
    if any(rate and key not in usage for key, rate in rates.items()):
        return None
    return sum(usage.get(key, 0) * rate for key, rate in rates.items()) / 1_000_000


def text(
    model_id: str | None = None,
    *,
    require: Mapping[str, Any] | None = None,
    prefer: str | None = None,
    registry: Registry | None = None,
    sink: RecordSink | None = None,
) -> TextClient:
    """A `TextClient` for a registry id, an ad-hoc `provider:model` id, or the best model meeting `require`.

    llm = mk.text("gemma4-12b")
    vision = mk.text(require={"vision": True, "min_context": 8000}, prefer="local")
    """
    reg = registry or load()
    cfg = reg.get(model_id) if model_id else reg.select(require, prefer)
    missing = unmet(cfg, require or {}) if model_id else []
    if missing:
        raise CapabilityError(f"model {cfg.id!r} does not meet {', '.join(missing)}")
    if cfg.kind != "chat":
        raise ConfigError(
            f"model {cfg.id!r} is a {cfg.kind} model; use mk.embedder(), mk.decision() or mk.speech()"
        )
    register_key(cfg)
    return TextClient(cfg, sink or default_sink(), probe=cfg.id not in reg.models)
