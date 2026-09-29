"""The packaged ComfyUI workflows (change 0015 §3b): each parses as API format, every `inputs` path names a
node and an input of its workflow, the outputs are save nodes writing under `hone/`, and every ComfyUI
model installed on the reference machine has a workflow."""

from importlib import resources
from pathlib import Path

import pytest

import hone_models as mk
from hone_models._registry_shapes import node_targets
from hone_models.providers import _comfyui_workflow

WORKFLOWS = Path(str(resources.files("hone_models") / "data" / "workflows"))
# The ComfyUI models installed on the reference machine (2026-09-29), each proven with `models check`.
PROVEN = {
    "z-image-turbo",
    "ace-step-1.5-turbo",
    "ace-step-1.5-xl-turbo",
    "ace-step-1.5-xl-sft",
    "minimax-music3",
    "yue2-3b",
    "heartmula-3b",
    "heartmula-rl-3b",
    "stable-audio-open-1.0",
    "wan2.2-i2v-14b",
    "wan2.2-ti2v-5b",
    "ltx-video-2b-0.9.5",
}
SAVE_NODES = {"SaveImage", "SaveAudio", "SaveAudioAdvanced", "SaveAudioMP3", "SaveVideo"}


def comfyui_entries() -> list[mk.registry.ModelConfig]:
    return [m for m in mk.registry.load().models.values() if m.provider == "comfyui" and m.workflow]


def test_every_installed_comfyui_model_has_a_workflow() -> None:
    assert {m.id for m in comfyui_entries()} >= PROVEN
    used = {Path(str(m.workflow)).name for m in comfyui_entries()}
    assert used == {p.name for p in WORKFLOWS.glob("*.json")}  # no workflow file without an entry


@pytest.mark.parametrize("cfg", comfyui_entries(), ids=lambda m: m.id)
def test_workflow_matches_its_entry(cfg: mk.registry.ModelConfig) -> None:
    workflow, _ = _comfyui_workflow.load(cfg)  # API format: node id -> {class_type, inputs}
    assert isinstance(cfg.inputs, dict)
    assert {"prompt", "seed"} <= set(cfg.inputs)
    for name, mapping in cfg.inputs.items():
        for target in node_targets(mapping):
            node, field = target.path.split(".", 1)
            assert node in workflow, (name, target.path)
            assert field in workflow[node]["inputs"], (name, target.path)
    for node, spec in workflow.items():
        for value in spec["inputs"].values():
            if isinstance(value, list) and len(value) == 2:  # a link: [node id, output index]
                assert str(value[0]) in workflow, (node, value)
    assert cfg.outputs
    for node in cfg.outputs:
        assert workflow[node]["class_type"] in SAVE_NODES
        assert workflow[node]["inputs"]["filename_prefix"].startswith("hone/")
    assert cfg.capabilities.vram_gb is not None  # measured by `models check` on the reference machine
    filled = _comfyui_workflow.fill(workflow, cfg, {"prompt": "a test", "seed": 3, **cfg.defaults})
    assert filled != workflow
