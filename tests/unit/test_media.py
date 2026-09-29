"""The media client: input checks, seeds, cost, sessions, factories, records; and FakeMedia."""

from pathlib import Path

import pytest

import hone_models as mk
from hone_models.errors import CapabilityError, ConfigError
from hone_models.providers.common import error_kind
from hone_models.testing import FakeMedia
from media_fixtures import registry

HOSTED = """
[models."test-sora"]
provider = "comfyui"
kind = "video"
[models."test-sora".capabilities]
sizes = ["720x1280", "1280x720"]
durations_s = [4, 8, 12]
price = { per_output_second = 0.10 }
commercial_use = true

[models."test-gpt-image"]
provider = "comfyui"
kind = "image"
defaults = { seed = 11, size = "256x256" }
[models."test-gpt-image".capabilities]
price = { per_image = 0.04 }
license = "proprietary"
"""


def spans(client: mk.MediaClient) -> list[dict]:
    return client.sink.spans  # type: ignore[attr-defined]  # a MemorySink


@pytest.fixture
def reg(tmp_path: Path) -> mk.registry.Registry:
    path = tmp_path / "priced.toml"
    path.write_text(HOSTED)
    return mk.registry.load([path])


def test_factories_check_the_kind() -> None:
    reg = registry()
    assert (mk.image("test-image", registry=reg).kind, mk.music("test-music", registry=reg).kind) == (
        "image",
        "music",
    )
    with pytest.raises(ConfigError, match=r"is a music model; mk.video\(\) takes video models"):
        mk.video("test-music", registry=reg)
    with pytest.raises(ConfigError, match="is a chat model"):
        mk.image("gemma4-12b")


def test_inputs_listed_from_the_entry() -> None:
    reg = registry()
    assert mk.image("test-image", registry=reg).inputs == ["references", "size", "steps"]
    assert mk.video("test-video", registry=reg).inputs == ["duration_s", "image", "negative", "size"]


def test_sizes_and_durations_declared_by_the_entry(reg: mk.registry.Registry, tmp_path: Path) -> None:
    sora = FakeMedia.like("test-sora", registry=reg)
    with pytest.raises(CapabilityError, match=r"durations of \[4.0, 8.0, 12.0\] s, not 5"):
        sora.generate("a wave", duration_s=5, out=tmp_path / "c.mp4")
    with pytest.raises(CapabilityError, match="sizes"):
        sora.generate("a wave", size="1024x1024", duration_s=4, out=tmp_path / "c.mp4")
    with pytest.raises(ConfigError, match="WxH"):
        sora.generate("a wave", size="big", out=tmp_path / "c.mp4")
    assert sora.calls == []
    with pytest.raises(CapabilityError, match="at most 5"):
        FakeMedia.like("test-video", registry=registry()).generate("x", duration_s=6, out=tmp_path / "v.mp4")


def test_seed_default_random_and_recorded(reg: mk.registry.Registry, tmp_path: Path) -> None:
    img = FakeMedia.like("test-gpt-image", registry=reg)
    assert img.generate("a", out=tmp_path / "a.png").seed == 11  # defaults.seed
    fake = FakeMedia()
    seeds = {fake.generate("b", out=tmp_path / f"b{i}.png").seed for i in range(3)}
    assert len(seeds) == 3  # random when neither given nor a default
    assert [s["attributes"]["gen_ai.request.seed"] for s in spans(fake)] == [c[2] for c in fake.calls]


def test_naive_cost_per_image_and_per_second(reg: mk.registry.Registry, tmp_path: Path) -> None:
    img = FakeMedia.like("test-gpt-image", registry=reg)
    r = img.generate("two cats", n=2, out=tmp_path / "cats.png")
    assert [f.path.name for f in r.files] == ["cats_1.png", "cats_2.png"]
    assert (r.files[0].width, r.files[0].height) == (256, 256)  # the entry's default size
    assert (r.cost_usd, r.cost_estimated, r.license, r.commercial_use) == (0.08, True, "proprietary", None)
    attrs = spans(img)[0]["attributes"]
    assert (attrs["hone.models.cost_usd"], attrs["hone.models.media.cost_estimated"]) == (0.08, True)
    clip = FakeMedia.like("test-sora", registry=reg).generate("a wave", duration_s=4, out=tmp_path / "c.mp4")
    assert clip.cost_usd is None or clip.cost_usd == pytest.approx(0.1)  # 1 s packaged clip when ffprobe runs
    song = FakeMedia(kind="music").generate("hum", duration_s=2, out=tmp_path / "s.wav")
    assert (song.cost_usd, song.cost_estimated) == (None, False)


def test_failed_job_and_no_output(tmp_path: Path) -> None:
    fake = FakeMedia(kind="video")
    fake.fail_next("Your request was rejected by the safety system")
    r = fake.generate("x", out=tmp_path / "v.mp4")
    assert (r.error_kind, r.path, r.cost_usd) == ("refused", None, None)
    assert spans(fake)[0]["status"]["code"] == "error"
    assert fake.generate("y", out=tmp_path / "v.mp4").files[0].mime == "video/mp4"


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("torch.OutOfMemoryError: CUDA out of memory", "out_of_memory"),
        ("blocked by moderation", "refused"),
        ("ValueError: steps must be positive", "invalid_input"),
        ("something else", "failed"),
    ],
)
def test_error_kinds(message: str, kind: str) -> None:
    assert error_kind(message) == kind


def test_file_inputs_as_strings_and_recorded(tmp_path: Path) -> None:
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"not really a png")
    fake = FakeMedia()
    fake.generate("x", references=[str(ref)], image=str(ref), out=tmp_path / "o.png")
    inputs = fake.calls[0][1]
    assert inputs["references"] == [ref]
    assert inputs["image"] == ref
    recorded = spans(fake)[0]["attributes"]["hone.models.media.inputs"]
    assert recorded["image"]["bytes"] == 16
    with pytest.raises(ConfigError, match="does not exist"):
        fake.generate("x", image=str(tmp_path / "missing.png"), out=tmp_path / "o.png")


def test_sessions_do_not_nest(tmp_path: Path) -> None:
    fake = FakeMedia(kind="music")
    with fake.session() as song:
        song.generate("a", out=tmp_path / "a.wav")
        with pytest.raises(ConfigError, match="already open"), fake.session():
            pass
    assert spans(fake)[0]["attributes"]["hone.models.media.session"] is True


def test_fake_media_like_checks_kind() -> None:
    with pytest.raises(ConfigError, match="not an image, music or video model"):
        FakeMedia.like("kokoro-82m")
    with pytest.raises(ConfigError, match="kind must be one of"):
        FakeMedia(kind="speech")


def test_text_inputs_hashed_when_capture_is_off(tmp_path: Path) -> None:
    store = tmp_path / "spans.db"
    sink = mk.records.SqliteSpanSink(store, capture_content=False)
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"x")
    FakeMedia(kind="music", sink=sink).generate(
        "secret style", lyrics="[verse]\nsecret words", duration_s=1, references=[ref], out=tmp_path / "s.wav"
    )
    sink.close()
    (span,) = mk.records.read_spans(store)
    inputs = span["attributes"]["hone.models.media.inputs"]
    assert set(inputs["lyrics"]) == {"sha256", "len"}
    assert inputs["duration_s"] == 1
    assert inputs["references"][0]["path"] == str(ref)  # file records stay readable
    assert set(span["attributes"]["hone.models.media.prompt"]) == {"sha256", "len"}
    assert b"secret" not in store.read_bytes()
