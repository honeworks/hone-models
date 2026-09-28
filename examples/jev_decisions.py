"""A native decision model (Jev): the same questions and answers as with an LLM, calibrated by the model.

What: send decision questions to Jev, a model built to answer them (probabilities and scores instead of
    text), and read calibrated answers, the cost of the call and a per-question error.

How: `mk.decision("jev")` uses the packaged registry entry (`provider = "jev"`, `kind = "decision"`,
    key from `TYPESAFE_API_KEY`); `dm.decide(state, questions)` makes one `POST {base_url}/decide` call.
    The questions and the answer shape are exactly those of decisions.py, so code written against
    `mk.decision(...)` does not care which kind of model answers. `base_url` can be overridden in the
    registry (here: to point at the fake server).

Why: an LLM emulating a decision model reports stated confidence; a decision model reports calibrated
    probabilities (`calibrated=True`) and is cheaper per question. Pitfalls: the Jev API shape in
    hone-models is unverified (built from public articles), and a question Jev did not answer gets its
    own `error` with `value=None`; the other answers are kept.

Runs offline: `FakeOllama` answers the `/decide` request with a scripted Jev reply.
"""

import os
import tempfile
from pathlib import Path

import hone_models as mk
from hone_models.testing import FakeOllama

server = FakeOllama().start()  # offline stand-in for the Jev API
os.environ["TYPESAFE_API_KEY"] = "ts-example-key-0123456789"  # normally set in your shell

registry_file = Path(tempfile.mkdtemp()) / "models.toml"
registry_file.write_text(f'[models.jev]\nbase_url = "{server.url}"\n', encoding="utf-8")  # the fake API
reg = mk.registry.load(registry_file)

server.queue(
    "/decide",
    {
        "model": "jev-1",
        "answers": [
            {"id": "urgent", "p_yes": 0.92},
            {"id": "team", "probabilities": {"infra": 0.7, "backend": 0.25, "frontend": 0.05}},
            {"id": "severity", "score": 4},
        ],  # no answer for "rollback"
        "usage": {"input_tokens": 57},
    },
)
sink = mk.records.MemorySink()
dm = mk.decision("jev", registry=reg, sink=sink)
answers = dm.decide(
    "The deploy failed twice and customers see 500s.",
    {
        "urgent": mk.YesNo("Does this need attention now?"),
        "team": mk.Choice(["infra", "backend", "frontend"], "Which team owns it?"),
        "severity": mk.ScoreQ("How severe?", scale=(1, 5)),
        "rollback": mk.YesNo("Should the deploy be rolled back?"),
    },
)
for name, a in answers.items():
    print(
        f"{name:>8}: value={a['value']} choice={a['choice']} calibrated={a['calibrated']} error={a['error']}"
    )

sent = server.requests[-1]["body"]
span = sink.spans[-1]
print(
    "sent questions:",
    [q["id"] for q in sent["questions"]],
    "| cost USD:",
    span["attributes"]["hone.models.cost_usd"],
)
assert abs(span["attributes"]["hone.models.cost_usd"] - 57 * 0.042 / 1_000_000) < 1e-15  # input tokens only
assert answers["urgent"]["value"] == 0.92
assert answers["urgent"]["calibrated"] is True
assert answers["team"]["choice"] == "infra"
assert answers["severity"]["value"] == 0.75  # (4 - 1) / (5 - 1)
assert answers["rollback"]["value"] is None
assert "missing" in answers["rollback"]["error"]
assert server.requests[-1]["headers"]["Authorization"] == "Bearer ts-example-key-0123456789"
assert "ts-example-key-0123456789" not in str(sink.spans)  # sent to the API, never recorded

server.stop()
