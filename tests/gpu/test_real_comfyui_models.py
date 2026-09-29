"""Every installed ComfyUI and `command` entry of the packaged catalog runs its tiny job (`hone-models
models check`): an output of the right kind, and the GPU memory given back afterwards. Slow (the video
models take minutes); run by hand when ComfyUI, a workflow, an adapter or the catalog changes.

Run: scripts/gpu-lock.sh uv run pytest -m "gpu and slow" tests/gpu/test_real_comfyui_models.py
ComfyUI as for AC-30 (tests/gpu/test_ac30_real_generation.py): a running server is used and freed, else
`HONE_COMFYUI_START` starts one per entry and stops it afterwards. A `command` entry runs when its project
variable is set (HeartMuLa: `HONE_HEARTLIB_DIR`) and needs no ComfyUI.
"""

from pathlib import Path

import pytest
from comfyui_real import COMFYUI_DIR, gpu_used_mb, use_real_comfyui

import hone_models as mk
from hone_models import _media_check

pytestmark = [pytest.mark.gpu, pytest.mark.comfyui, pytest.mark.slow]
MIMES = {"image": "image/", "music": "audio/", "video": "video/"}
MARGIN_MB = 512


def installed_entries() -> list[str]:
    """The packaged ComfyUI entries with a workflow, and the `command` entries, installed on this machine."""
    reg = mk.registry.load()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HONE_COMFYUI_DIR", COMFYUI_DIR)
        checker = mk.catalog.Checker()
        found = [m.id for m in reg.models.values() if m.workflow or (m.provider == "command" and m.command)]
        return sorted(m for m in found if checker.installed(reg.get(m)) == "yes")


@pytest.mark.parametrize("model_id", installed_entries())
def test_tiny_job(model_id: str, gpu_lock, real_out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = mk.registry.load().get(model_id)
    if cfg.provider == "comfyui":
        use_real_comfyui(monkeypatch)
    else:
        monkeypatch.setenv("HONE_COMFYUI_DIR", COMFYUI_DIR)  # where HeartMuLa's checkpoints are
    client = {"image": mk.image, "music": mk.music, "video": mk.video}[cfg.kind](
        model_id, sink=mk.records.NullSink()
    )
    before = gpu_used_mb()

    found = _media_check.check(client, real_out)

    after = gpu_used_mb()
    assert found["error"] is None, found
    assert found["bytes"] > 1000
    assert str(found["mime"]).startswith(MIMES[cfg.kind])
    if cfg.kind == "image":
        assert (found["width"], found["height"]) == (256, 256)
    if cfg.kind == "music":
        assert found["duration_s"] is not None
        assert 1 <= found["duration_s"] <= 11
    if found["peak_vram_gb"] is not None and cfg.capabilities.vram_gb is not None:
        assert found["peak_vram_gb"] <= cfg.capabilities.vram_gb + 0.5  # the catalog's figure still holds
    if before is not None and after is not None:
        assert after - before < MARGIN_MB, (before, after)
