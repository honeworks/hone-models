"""Decision questions answered by an LLM: yes/no, choice and score, each with its own answer or error.

What: ask several questions about one state (a text or a dict) in one call to an ordinary chat model,
    and read the normalized answers, including a question the model answered badly.

How: `dm = mk.decision(model_id)`; questions are `mk.YesNo(instructions)`,
    `mk.Choice(options, instructions)` and `mk.ScoreQ(instructions, scale=(low, high), anchors=...)`.
    `dm.decide(state, {name: question})` returns `{name: answer}` with the same names. Every answer has
    `value` (0-1: P(yes), the chosen option's probability, or the normalized score), `choice`, `raw`
    (the score on its scale), `probabilities`, `confidence`, `calibrated`, `rationale` and `error`.
    For a chat model this is one schema-constrained call (a decision model such as Jev is called
    natively: jev_decisions.py).

Why: judges and panels (hone-select, hone-taste) need numbers they can compare, not prose. One call
    answers all questions, and a bad answer to one question sets only that answer's `error` with
    `value=None` ("could not score" is never a 0). `calibrated` is True only when P(yes) comes from token
    logprobs (a registry entry with `logprobs = true`, on a server that returns them); otherwise it is the
    model's stated confidence, so do not treat it as a probability.
"""

import json

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for Ollama (sets OLLAMA_HOST)
sink = mk.records.MemorySink()
dm = mk.decision("gemma4-12b", sink=sink)
questions = {
    "urgent": mk.YesNo("Does this need attention now?"),
    "team": mk.Choice(["infra", "backend", "frontend"], "Which team owns it?"),
    "severity": mk.ScoreQ("How severe?", scale=(1, 5), anchors={1: "cosmetic", 5: "outage"}),
}

# 1. All questions answered in one call.
answers = dm.decide("The deploy failed twice and customers see 500s.", questions)
for name, a in answers.items():
    print(
        f"{name:>8}: value={a['value']:.2f} choice={a['choice']} raw={a['raw']} calibrated={a['calibrated']}"
    )
assert answers["team"]["choice"] == "infra"
assert answers["severity"]["raw"] == 3
assert answers["severity"]["value"] == 0.5  # (3 - 1) / (5 - 1)
assert answers["urgent"]["calibrated"] is False  # Ollama returns no logprobs: stated confidence only

# 2. The model gets one answer wrong (severity 9 on a 1-5 scale) on every attempt: only that answer fails.
bad = json.dumps(
    {
        "urgent": {"answer": "yes", "confidence": 0.9, "rationale": "customers affected"},
        "team": {"answer": "infra", "confidence": 0.7, "rationale": "deploys"},
        "severity": {"answer": 9, "confidence": 0.8, "rationale": "very bad"},
    }
)
reply = {"message": {"role": "assistant", "content": bad}}
server.queue("/api/chat", reply, reply, reply)  # the first attempt and both retries
state = {"service": "checkout", "errors_per_min": 340, "deploys_failed": 2}  # a dict state works too
answers = dm.decide(state, questions)
for name, a in answers.items():
    print(f"{name:>8}: value={a['value']} error={a['error']}")
assert answers["urgent"]["value"] == 0.9
assert answers["severity"]["value"] is None
assert "outside the scale" in answers["severity"]["error"]

# Each decide() is one hone.models.decide span with the chat call nested under it.
decide_span, chat_span = sink.spans[-1], sink.spans[-2]
assert chat_span["parent_span_id"] == decide_span["span_id"]

server.stop()
