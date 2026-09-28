# Third-party notices

hone-models is licensed under the Apache License 2.0 (see `LICENSE`). This file lists the third-party
material shipped in the package and the licences of notable optional dependencies.

## Shipped in the package

### Voice reference clips (`src/hone_models/data/voices/*.flac`)

The four reference clips used by the `chatterbox` speech model were rendered with
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (hexgrad), licensed under the Apache License 2.0,
using Kokoro's synthetic voices. The spoken text was written for this project; no real person's
recording is used. They were made with `scripts/make-expressive-voices.py`:

| Clip | Kokoro-82M voice |
|---|---|
| `warm_female.flac` | `af_heart` |
| `warm_male.flac` | `am_michael` |
| `british_female.flac` | `bf_emma` |
| `british_male.flac` | `bm_george` |

## Model weights

No model weights are shipped. Models are downloaded by their own libraries or servers on first use, under
their own licences, for example: Gemma 4 (Gemma Terms of Use), Kokoro-82M (Apache-2.0), Chatterbox
(MIT; its output carries Resemble AI's Perth watermark), Qwen2.5-VL (Apache-2.0).

## Optional extras with copyleft dependencies

The core depends only on permissively licensed packages (pydantic: MIT, httpx: BSD-3-Clause). Some
optional extras pull in copyleft packages. They are not redistributed with hone-models: you install them
when you install the extra, and hone-models imports them at run time only when the extra is used.

| Extra | Package | Licence | Pulled in by |
|---|---|---|---|
| `speech` | phonemizer-fork | GPL-3.0-or-later | kokoro -> misaki[en] |
| `speech` | espeakng-loader (bundles espeak-ng) | GPL-3.0 (espeak-ng) | kokoro -> misaki[en] |
| `speech` | num2words | LGPL | kokoro -> misaki[en] |
| `speech` | en_core_web_sm (installed by you, see docs/speech.md) | MIT | spaCy's English model for misaki |
| `expressive` | pykakasi | GPL-3.0-or-later | chatterbox-tts |
| `expressive` | soxr | LGPL-2.1-or-later | librosa |

torch, installed by both speech extras, pulls NVIDIA's CUDA wheels under NVIDIA's own licence terms.
If you redistribute an application that bundles these extras, check the licences of what you ship.
