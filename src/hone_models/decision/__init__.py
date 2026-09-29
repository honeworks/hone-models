"""Decision questions about a state: `mk.decision(...)` implements the decision client shape (design §6).

    dm = mk.decision("gemma4-12b")          # an LLM emulating a decision model
    a = dm.decide("The deploy failed twice and customers see 500s.", {
        "urgent": mk.YesNo("Does this need attention now?"),
        "team": mk.Choice(["infra", "backend", "frontend"], "Which team owns it?"),
        "severity": mk.ScoreQ("How severe?", scale=(1, 5)),
    })
    a["urgent"]["value"], a["team"]["choice"], a["severity"]["raw"]

Native decision models (`kind = "decision"`, e.g. Jev) are called directly; chat models answer through
one structured call (`emulate.py`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .._http import register_key
from .._tracing import start_span
from ..budget import timeout_for
from ..errors import ConfigError
from ..ports import RecordSink
from ..providers import jev, model_attributes
from ..records import default_sink
from ..registry import ModelConfig, Registry, load, require_client
from ..text import TextClient, inline_image, text, usage_attributes
from .emulate import emulate
from .questions import Choice, ScoreQ, YesNo, check_questions

__all__ = ["Choice", "DecisionClient", "ScoreQ", "YesNo", "decision"]


class DecisionClient:
    """Answers decision questions with one model; records a `hone.models.decide` span per call."""

    def __init__(self, config: ModelConfig, sink: RecordSink, llm: TextClient | None = None) -> None:
        self.config = config
        self.sink = sink
        self._llm = llm

    @property
    def model_id(self) -> str:
        return self.config.id

    def decide(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        *,
        images: Sequence[str] = (),
        trace: Mapping[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Answer every question about `state`; returns `{name: Answer}` with the same keys.

        A question that could not be answered has `value=None` and `error` set; transport and
        configuration problems raise.
        """
        qs = check_questions(questions)
        attrs = {
            **model_attributes(self.config, "decide"),
            "hone.models.decision.state": state,
            "hone.models.decision.questions": qs,
        }
        with start_span("hone.models.decide", self.sink, attrs, trace=trace) as span:
            if self._llm is not None:
                answers = emulate(self._llm, state, qs, images)
            else:
                answers = self._native(state, qs, images, span["attributes"])
            span["attributes"]["hone.models.decision.answers"] = answers
            errors = [a["error"] for a in answers.values() if a["error"]]
            if errors and len(errors) == len(answers):
                span["status"] = {"code": "error", "message": errors[0]}
        return answers

    def _native(
        self,
        state: str | Mapping[str, Any],
        qs: dict[str, dict[str, Any]],
        images: Sequence[str],
        attrs: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        if self.config.provider != "jev":
            raise ConfigError(f"no native decision provider {self.config.provider!r}; use provider 'jev'")
        parts = [inline_image({"type": "image", "path": p}) for p in images]
        answers, usage, model = jev.decide(self.config, state, qs, parts, timeout_for(self.config, 0))
        attrs["gen_ai.response.model"] = model
        attrs.update(usage_attributes(self.config, usage))
        return answers


def decision(
    model_id: str, *, registry: Registry | None = None, sink: RecordSink | None = None
) -> DecisionClient:
    """A `DecisionClient`: native for decision models (`jev`), emulated with one structured call for LLMs.

    dm = mk.decision("gemma4-12b")
    """
    reg = registry or load()
    cfg = reg.get(model_id)
    sink = sink or default_sink()
    if cfg.kind in ("embedding", "speech"):
        raise ConfigError(f"model {cfg.id!r} is a {cfg.kind} model; use mk.embedder() or mk.speech()")
    if cfg.kind == "decision":
        require_client(cfg)
        register_key(cfg)
        return DecisionClient(cfg, sink)
    llm = text(model_id, registry=reg, sink=sink)
    return DecisionClient(cfg, sink, llm)
