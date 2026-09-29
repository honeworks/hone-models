# pyright: reportAttributeAccessIssue=false
"""Fake heartlib pipeline module: records what the adapter asks for."""

import os
from pathlib import Path
from types import SimpleNamespace

import torch

seen: dict = {}


class HeartMuLaGenConfig:
    @staticmethod
    def from_file(path: str) -> str:
        return f"config:{Path(path).name}"


class HeartMuLa:
    @staticmethod
    def from_pretrained(path: str, **kwargs: object) -> SimpleNamespace:
        seen["mula_4bit"] = {"path": path, "quantization": kwargs.get("quantization_config")}
        return SimpleNamespace(backbone=SimpleNamespace(max_seq_len=8192))


class _Wave:
    shape = (2, 480_000)

    def to(self, dtype: str) -> "_Wave":
        return self

    def cpu(self) -> "_Wave":
        return self

    def numpy(self) -> "_Wave":
        return self

    @property
    def T(self) -> dict:
        return {**seen, "seeds": torch.seeds, "events": torch.events, "cwd": os.getcwd()}


class HeartMuLaGenPipeline:
    def __init__(self, **kwargs: object) -> None:
        seen["pipeline"] = {k: str(v) for k, v in kwargs.items()}
        self._mula = None
        self.codec_device = "cuda"
        self.codec = SimpleNamespace(
            detokenize=self._detokenize,
            flow_matching=SimpleNamespace(to=lambda device: torch.events.append(f"flow to {device}")),
            scalar_model=SimpleNamespace(decode=lambda latent: torch.events.append("decode")),
        )

    @property
    def mula(self) -> SimpleNamespace:
        if self._mula is None:
            self._mula = SimpleNamespace(backbone=SimpleNamespace(max_seq_len=8192))
        return self._mula

    def preprocess(self, inputs: dict, cfg_scale: float) -> dict:
        seen["input"] = {**inputs, "cfg_scale": cfg_scale}
        return {"tokens": SimpleNamespace(shape=(2, 30, 9))}

    def _forward(self, prompt: dict, **settings: object) -> dict:
        seen["forward"] = {**settings, "cache": self.mula.backbone.max_seq_len}
        return {"frames": SimpleNamespace(to=lambda device: "frames")}

    def _detokenize(self, frames: object) -> _Wave:
        self.codec.scalar_model.decode("latent")
        return _Wave()

    def _unload(self) -> None:
        torch.events.append("unload")
