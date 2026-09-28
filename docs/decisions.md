# Decisions

A decision client answers structured questions about a state (text or a JSON-able object):

```python
import hone_models as mk

dm = mk.decision("gemma4-12b")
a = dm.decide(
    "The deploy failed twice and customers see 500s.",
    {
        "urgent": mk.YesNo("Does this need attention now?"),
        "team": mk.Choice(["infra", "backend", "frontend"], "Which team owns it?"),
        "severity": mk.ScoreQ("How severe?", scale=(1, 5), anchors={1: "cosmetic", 5: "outage"}),
    },
)
print(a["urgent"]["value"], a["team"]["choice"], a["severity"]["raw"])
```

Questions are plain dicts (`{"type": "yes_no" | "choice" | "score", "instructions": ..., "options": ...,
"scale": ..., "anchors": ...}`); `mk.YesNo`, `mk.Choice` and `mk.ScoreQ` build them. Every answer has the
same keys:

| Key | Meaning |
|---|---|
| `value` | yes/no: P(yes); choice: probability of the chosen option; score: `(raw - low) / (high - low)` |
| `choice`, `probabilities`, `raw` | the chosen option, per-option probabilities, the score on its scale |
| `confidence`, `rationale` | as stated by the model |
| `calibrated` | `True` only when probabilities come from token logprobs or a decision model |
| `error` | set (and `value` is `None`) when this question could not be answered |

**On LLMs** the client makes one schema-constrained call answering every question, at temperature 0.
Invalid answers are sent back for correction (up to 2 retries); what is still invalid becomes that
question's `error`. When the model declares `logprobs = true` (an OpenAI-compatible server that returns
them), P(yes) comes from the token logprobs and is marked `calibrated`.

**Jev** (TypeSafe AI) is a native decision model: `mk.decision("jev")` with `TYPESAFE_API_KEY` set. Its
API shape is reconstructed from public articles and **unverified**; see `providers/jev.py`.

```python no-run
dm = mk.decision("jev")
```

**Runnable examples:** [decisions.py](../examples/decisions.py), [jev_decisions.py](../examples/jev_decisions.py).
