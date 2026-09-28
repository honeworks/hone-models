# Text and structured output

```python
import hone_models as mk

llm = mk.text("gemma4-12b")
r = llm.complete(
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name a sea shanty."}],
    temperature=0.2,
    seed=7,
    max_tokens=100,
)
print(r.text, r.finish_reason, r.usage, r.span_id)
```

Params: `temperature`, `top_p`, `seed`, `max_tokens`, `stop`, `think` (reasoning on/off for thinking
models; Ollama only), `logprobs` (token logprobs on `r.logprobs`, OpenAI-compatible servers). Unknown params are
ignored. `max_tokens` defaults to the model's `max_output_tokens`, else 2048, and is always sent, so a
cut-off reply is reported as `finish_reason == "length"`.

**Images** are content parts: `{"type": "image", "path": "frame.png"}` or
`{"type": "image", "data_b64": ..., "mime": "image/png"}`.

## Structured output

Pass a Pydantic model class or a JSON Schema dict as `schema`:

```python
import hone_models as mk
from pydantic import BaseModel


class Verdict(BaseModel):
    score: int
    rationale: str


r = mk.text("gemma4-12b").complete([{"role": "user", "content": "Rate this pitch 1-5 ..."}], schema=Verdict)
print(r.parsed, r.structured_path, r.attempts, r.error)
```

The pipeline:
1. **Budget:** estimated prompt tokens (characters / 3.5, plus the images) plus requested output must
   fit `max_input_tokens`, else `ContextOverflow` before any request. Ollama gets `num_ctx` sized to the
   request (rounded up to 2048, so the model is not reloaded for every size). An image costs the model's
   `image_tokens` (flat), else one token per `image_patch_px` square of its size (read from the file
   header; `qwen2.5vl-7b`: 28 px), else 1024; the image share is recorded as
   `hone.models.context.estimated_image_tokens`. `min_num_ctx=` (a call param, or `defaults` in the
   registry) sets a floor for `num_ctx`: for a model whose image cost is unknown, or to keep one context
   size so Ollama never reloads the model.
2. **Constrain** when the model declares `json_schema` (Ollama `format`, OpenAI `response_format`). The
   schema is also described in the system message.
3. **Parse:** markdown fences and surrounding prose are stripped.
4. **Validate** with Pydantic (a class) or a small JSON Schema validator (a dict).
5. **Retry** up to 2 times, sending the validation error back.
6. **Repair** truncated JSON as a last resort.

`r.structured_path` is `constrained` (clean output under a constraint), `parsed` (cleanup needed),
`retried`, `repaired` or `failed`; with `failed`, `r.parsed` is `None` and `r.error` says why.

**Runnable examples:** [chat_and_structured.py](../examples/chat_and_structured.py), [context_budget.py](../examples/context_budget.py), [thinking_models.py](../examples/thinking_models.py), [vision.py](../examples/vision.py).
