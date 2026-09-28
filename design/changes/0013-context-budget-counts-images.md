# 0013: The context budget counts images

## Status

`implemented in 0.1.0` (option 3)

## Context

Found in concept-shorts (a demo app built on the honeworks packages). Its vision judge (`qwen2.5vl-7b` on Ollama) reviews two
keyframes per scene (540x960 and 960x540) with a short prompt and `max_tokens=400`. The real call failed:

    ProviderError: .../api/chat returned HTTP 400: request (2389 tokens) exceeds the available context
    size (2048 tokens)

`budget.estimate_tokens` counts only the text of the messages (`text_of`, characters / 3.5), so
`plan_context` sized `num_ctx` for about 700 prompt tokens + 400 output and rounded it to 2048. The two
images alone cost about 1400 tokens in Qwen2.5-VL (one token per 28x28 pixels). The overflow was not
caught before the HTTP call, which `budget.py` promises ("Nothing is truncated silently ... raised before
any HTTP call").

## Problem

Every vision call with images of more than a few hundred thousand pixels gets a `num_ctx` that is too
small. Callers can only work around it by inflating `max_tokens` (which also raises `num_predict`), since
there is no per-call or per-model context override. `ContextOverflow` is also never raised for images
that don't fit the model's declared `max_input_tokens`.

## Options

1. Count images in `estimate_tokens`: read each image part's size (path or `data_b64`, header only) and
   add a per-model cost. A capability such as `image_patch_px` (28 for Qwen2.5-VL, 14 for Gemma 3 at a
   fixed 256 tokens, ...) or a flat `image_tokens` estimate, with a conservative default (e.g. 1024 per
   image) when neither is declared.
2. Let callers or the registry set a minimum context (`min_num_ctx` in `defaults` or a `num_ctx` param).
3. Both: 1 as the default behaviour, 2 as an escape hatch for models whose image cost is unknown.

## Recommendation

Option 3. Option 1 fixes the silent wrong sizing and restores the promise that overflow is found before
the call; option 2 is small and useful for other cases too (a fixed context so Ollama doesn't reload the
model). Record the image share in the span (`hone.models.context.estimated_image_tokens`). After this,
concept-shorts can drop its workaround (`storyboard.review_budget`, which reserves the images' share
through `max_tokens`).

## Implementation

Capabilities `image_tokens` (flat per image) and `image_patch_px` (one token per square; `qwen2.5vl-7b`
declares 28), else 1024 per image. Sizes come from the image header (PNG, JPEG, GIF, WebP) without an
imaging library, for `path` and `data_b64` parts. The escape hatch is one name, `min_num_ctx`, usable as a
call param or a registry default (defaults are params). `hone.models.context.estimated_prompt_tokens` now
includes the images; `estimated_image_tokens` is their share, recorded when images were sent.
concept-shorts can drop `storyboard.review_budget`.
