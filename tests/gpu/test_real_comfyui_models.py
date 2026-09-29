"""Every installed ComfyUI entry of the packaged catalog runs its tiny job (`hone-models models check`):
an output of the right kind, and the GPU memory given back afterwards. Slow (the video models take
minutes); run by hand when ComfyUI, a workflow or the catalog changes.

Run: scripts/gpu-lock.sh uv run pytest -m "gpu and slow" tests/gpu/test_real_comfyui_models.py
ComfyUI as for AC-30 (tests/gpu/test_ac30_real_generation.py): a running server is used and freed, else
`HONE_COMFYUI_START` starts one per entry and stops it afterwards.
"""

from pathlib import Path

import pytest
from comfyui_real import COMFYUI_DIR, gpu_used_mb, use_real_comfyui

import hone_models as mk
from hone_models import _media_check

pytestmark = [pytest.mark.gpu, pytest.mark.comfyui, pytest.mark.slow]
MIMES = {"image": "image/", "music": "audio/", "video": "video/"}
MARGIN_MB = 512
# Entries whose ComfyUI node fails on the reference machine for a reason outside hone-models (D-073).
KNOWN_FAILURES = {
    "heartmula-3b": "the HeartMuLa node's loader fails with transformers 5 (HeartCodec buffer shapes)",
    "heartmula-rl-3b": "the HeartMuLa node's loader fails with transformers 5 (HeartCodec buffer shapes)",
}


def installed_entries() -> list[object]:
    """The packaged ComfyUI entries with a workflow whose files are on this machine."""
    reg = mk.registry.load()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HONE_COMFYUI_DIR", COMFYUI_DIR)
        checker = mk.catalog.Checker()
        found = [m.id for m in reg.models.values() if m.provider == "comfyui" and m.workflow]
        ids = sorted(m for m in found if checker.installed(reg.get(m)) == "yes")
    return [
        pytest.param(m, marks=pytest.mark.xfail(reason=KNOWN_FAILURES[m], strict=False))
        if m in KNOWN_FAILURES
        else m
        for m in ids
    ]


@pytest.mark.parametrize("model_id", installed_entries())
def test_tiny_job(model_id: str, gpu_lock, real_out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_real_comfyui(monkeypatch)
    cfg = mk.registry.load().get(model_id)
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
