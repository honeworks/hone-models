"""hone-models adapter for HeartMuLa (https://github.com/HeartMuLa/heartlib).

The `command` provider runs it with heartlib's own Python, in heartlib's folder:

    <heartlib>/.venv/bin/python heartmula.py <request.json>

It imports only the standard library until the request is converted to HeartMuLa's input (the tags, the
lyrics already in HeartMuLa's section form, the longest length in ms, the sampling settings) and the
checkpoint folders are found. Then it hands over: it imports torch and heartlib, builds heartlib's
`HeartMuLaGenPipeline` from the entry's checkpoint and codec folders with lazy loading (HeartMuLa is
freed before HeartCodec loads), sizes HeartMuLa's key-value cache for the song, seeds Python, NumPy and
torch, and runs the pipeline's `preprocess` and `_forward` as its `__call__` does. It decodes with
HeartCodec and writes a 48 kHz FLAC itself, because the pipeline's own save calls `torchaudio.save`,
which needs `torchcodec` in torchaudio 2.9 and later. It writes `result.json` into `out_dir`: the file,
or the error of a job that ran and failed. Failing to import heartlib is a crash (a non-zero exit with
the traceback).

`defaults.low_mem` fits an 8 GB card: HeartMuLa in 4-bit (bitsandbytes NF4, as the ComfyUI node's
`use_4bit`), and HeartCodec's flow-matching part moved to the CPU before its waveform decoding.

The checkpoints folder is heartlib's `ckpt` layout (`tokenizer.json`, `gen_config.json` and the model
folders): `defaults.checkpoints`, an absolute path, or a path relative to ComfyUI's model folder
(`$HONE_COMFYUI_DIR/models`, else `~/ComfyUI/models`), where the ComfyUI node kept the same files.
It runs on heartlib's Python (3.10).
"""

from __future__ import annotations

import importlib
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

OUTPUT = "take.flac"
SAMPLE_RATE = 48000  # HeartCodec's output
FRAME_MS = 80  # one HeartMuLa frame (12.5 Hz)
CHECKPOINTS = "HeartMuLa"  # the entry's `defaults.checkpoints`, relative to ComfyUI's model folder
MODEL = "HeartMuLa-oss-3B"  # `defaults.checkpoint`: HeartMuLa's folder in the checkpoints folder
CODEC = "HeartCodec-oss-20260123"  # `defaults.codec`: HeartCodec's folder
DURATION_S = 60  # heartlib's `max_audio_length_ms` is 240 s by default; a song ends earlier at its end token
SAMPLING = {"topk": 50, "temperature": 1.0, "cfg_scale": 1.5}  # heartlib's defaults


class RequestError(ValueError):
    """The request cannot be converted to HeartMuLa's input."""


def heartmula_input(request: dict[str, Any]) -> dict[str, Any]:
    """The pipeline's input and settings: `lyrics`, `tags` (heartlib wants them comma-separated without
    spaces), `max_audio_length_ms` from `duration_s`, and `topk`, `temperature`, `cfg_scale`."""
    inputs, defaults = request.get("inputs", {}), request.get("defaults", {})
    lyrics = inputs.get("lyrics")
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise RequestError("HeartMuLa sings lyrics: the input 'lyrics' must be given")
    duration_s = inputs.get("duration_s", defaults.get("duration_s", DURATION_S))
    if not isinstance(duration_s, (int, float)) or duration_s <= 0:
        raise RequestError(f"duration_s must be a positive number of seconds, not {duration_s!r}")
    tags = ",".join(t.strip() for t in str(request.get("prompt") or "").split(",") if t.strip())
    settings = {name: defaults.get(name, value) for name, value in SAMPLING.items()}
    return {"lyrics": lyrics.strip(), "tags": tags, "max_audio_length_ms": int(duration_s * 1000), **settings}


def checkpoint_paths(request: dict[str, Any], environ: dict[str, str]) -> dict[str, Path]:
    """HeartMuLa's and HeartCodec's folders, `tokenizer.json` and `gen_config.json`; a `RequestError`
    naming what is missing."""
    defaults = request.get("defaults", {})
    folder = Path(str(defaults.get("checkpoints", CHECKPOINTS))).expanduser()
    if not folder.is_absolute():
        comfyui = Path(environ.get("HONE_COMFYUI_DIR") or Path.home() / "ComfyUI").expanduser()
        folder = comfyui / "models" / folder
    paths = {
        "mula": folder / str(defaults.get("checkpoint", MODEL)),
        "codec": folder / str(defaults.get("codec", CODEC)),
        "tokenizer": folder / "tokenizer.json",
        "gen_config": folder / "gen_config.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise RequestError(f"HeartMuLa's checkpoints are not there: {missing} (defaults.checkpoints)")
    return paths


def write_result(out_dir: Path, files: list[str], error: str | None, meta: dict[str, Any]) -> None:
    body = {"files": files, "error": error, "meta": meta}
    (out_dir / "result.json").write_text(json.dumps(body, indent=2), encoding="utf-8")


def pipeline(paths: dict[str, Path], low_mem: bool) -> tuple[Any, Any]:
    """heartlib's pipeline on the GPU with lazy loading, and torch; HeartMuLa in 4-bit when asked."""
    torch = importlib.import_module("torch")
    heartlib = importlib.import_module("heartlib.pipelines.music_generation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: HeartMuLa needs a GPU")
    device = torch.device("cuda")
    pipe = heartlib.HeartMuLaGenPipeline(
        heartmula_path=str(paths["mula"]), heartcodec_path=str(paths["codec"]),
        heartmula_device=device, heartcodec_device=device,
        heartmula_dtype=torch.bfloat16, heartcodec_dtype=torch.float32,  # heartlib's defaults
        lazy_load=True, muq_mulan=None,
        text_tokenizer=importlib.import_module("tokenizers").Tokenizer.from_file(str(paths["tokenizer"])),
        config=heartlib.HeartMuLaGenConfig.from_file(str(paths["gen_config"])),
    )  # fmt: skip
    if low_mem:
        transformers = importlib.import_module("transformers")
        nf4 = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )  # fmt: skip
        pipe._mula = heartlib.HeartMuLa.from_pretrained(
            str(paths["mula"]), device_map=device, dtype=torch.bfloat16, quantization_config=nf4
        )
    return pipe, torch


def fit_cache(backbone: Any, prompt_len: int, max_audio_length_ms: int) -> None:
    """Size HeartMuLa's key-value cache for this song instead of its 8192 positions (655 s): torchtune
    caches every attention head, 5.6 GB in bf16 at full length with guidance, more than an 8 GB card has
    beside the model. The pipeline writes one position per 80 ms frame after the prompt."""
    backbone.max_seq_len = min(backbone.max_seq_len, prompt_len + max_audio_length_ms // FRAME_MS + 1)


def free_before_waveform(codec: Any, torch: Any) -> None:
    """Move HeartCodec's flow-matching part (5.6 GB in fp32) to the CPU once it has made every latent,
    before its scalar model decodes them to audio: both do not fit on an 8 GB card with the decoding."""
    flow, decode = codec.flow_matching, codec.scalar_model.decode

    def decode_alone(latent: Any) -> Any:
        flow.to("cpu")
        torch.cuda.empty_cache()
        return decode(latent)

    codec.scalar_model.decode = decode_alone


def generate(pipe: Any, torch: Any, song: dict[str, Any], seed: int, target: Path, *, low_mem: bool) -> float:
    """Seed everything, generate the frames, decode them and write `target`; returns its seconds."""
    random.seed(seed)
    importlib.import_module("numpy").random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        prompt = pipe.preprocess(
            {"lyrics": song["lyrics"], "tags": song["tags"]}, cfg_scale=song["cfg_scale"]
        )
        fit_cache(pipe.mula.backbone, prompt["tokens"].shape[1], song["max_audio_length_ms"])
        frames = pipe._forward(  # what the pipeline's __call__ runs; it then saves with torchaudio
            prompt, max_audio_length_ms=song["max_audio_length_ms"], temperature=song["temperature"],
            topk=song["topk"], cfg_scale=song["cfg_scale"],
        )["frames"]  # fmt: skip
        if low_mem:
            free_before_waveform(pipe.codec, torch)
        wav = pipe.codec.detokenize(frames.to(pipe.codec_device)).to(torch.float32).cpu().numpy()
    pipe._unload()
    importlib.import_module("soundfile").write(str(target), wav.T, SAMPLE_RATE, format="FLAC")
    return wav.shape[-1] / SAMPLE_RATE


def main(argv: list[str] | None = None) -> int:
    """Run one request; the exit code is 0 whenever `result.json` was written."""
    request_path = Path((argv if argv is not None else sys.argv[1:])[0]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    out_dir, defaults = Path(request["out_dir"]), request.get("defaults", {})
    try:
        song = heartmula_input(request)
        paths = checkpoint_paths(request, dict(os.environ))
    except RequestError as exc:
        write_result(out_dir, [], str(exc), {})
        return 0
    low_mem = bool(defaults.get("low_mem", False))
    meta: dict[str, Any] = {"checkpoint": paths["mula"].name, "codec": paths["codec"].name,
                            "low_mem": low_mem}  # fmt: skip
    try:
        pipe, torch = pipeline(paths, low_mem)
        meta["seconds"] = generate(pipe, torch, song, int(request["seed"]), out_dir / OUTPUT, low_mem=low_mem)
    except ImportError:
        raise  # heartlib or its environment is broken: a crash, not a result
    except Exception as exc:  # the job ran and failed: a result, not a crash
        traceback.print_exc()
        write_result(out_dir, [], f"{type(exc).__name__}: {exc}", meta)
        return 0
    write_result(out_dir, [OUTPUT], None, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
