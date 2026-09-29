"""AC-26: hosted images (generation, edit with references, b64 and url answers) and video (polled, then
downloaded; failed; refused; timeout) through respx: files written, no bytes and no API key in the SQLite
file, a naive cost, and sizes or durations the model does not declare refused before any request."""

import base64
import json
from pathlib import Path

import httpx
import pytest
import respx

import hone_models as mk
from hone_models.errors import CapabilityError, ModelTimeout
from hone_models.providers import openai_media
from media_fixtures import png_file

pytestmark = pytest.mark.e2e
API = "https://api.openai.com/v1"
KEY = "sk-PLANTED-hosted-0123456789abcdef"
HOSTED = Path(__file__).parents[1] / "fixtures" / "hosted" / "models.toml"
VIDEO_ID = "video_68d7512d07848190b3e45da0ecbebcde"
VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42 not a real video, only bytes to store"


@pytest.fixture(autouse=True)
def hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setattr(openai_media, "POLL_S", 0.0)
    monkeypatch.setattr("hone_models._http.BACKOFF_S", 0.0)


@pytest.fixture
def db(isolated: Path) -> Path:
    return isolated / "spans.db"


def client(kind: str, model_id: str, db: Path) -> mk.MediaClient:
    factory = {"image": mk.image, "video": mk.video}[kind]
    return factory(model_id, registry=mk.registry.load([HOSTED]), sink=mk.records.SqliteSpanSink(db))


def stored(db: Path, *forbidden: bytes) -> list[dict]:
    """The spans in the SQLite file, after checking that none of `forbidden` (and never the key) is in it."""
    for f in db.parent.glob("spans.db*"):
        data = f.read_bytes()
        assert KEY.encode() not in data
        for chunk in forbidden:
            assert chunk not in data
    return mk.records.read_spans(db)


def test_ac26_image_generation_b64(db: Path, recorded, tmp_path: Path) -> None:
    answer = recorded("openai_images_b64.json")
    png = base64.b64decode(answer["data"][0]["b64_json"])
    img = client("image", "test-gpt-image", db)
    with respx.mock(base_url=API, assert_all_mocked=True) as mock:
        route = mock.post("/images/generations").respond(json=answer)
        r = img.generate("a lighthouse at dusk", size="1024x1024", quality="low", out=tmp_path / "a.png")
    sent = json.loads(route.calls[0].request.content)
    assert sent == {
        "model": "gpt-image-1.5",
        "prompt": "a lighthouse at dusk",
        "size": "1024x1024",
        "quality": "low",
    }
    assert route.calls[0].request.headers["Authorization"] == f"Bearer {KEY}"
    assert r.error is None
    assert r.path == tmp_path / "a.png"
    assert r.files[0].path.read_bytes() == png
    assert (r.files[0].width, r.files[0].height, r.files[0].mime) == (16, 16, "image/png")
    assert (r.cost_usd, r.cost_estimated) == (0.04, True)
    (span,) = stored(db, answer["data"][0]["b64_json"][:40].encode(), png[8:40])
    attrs = span["attributes"]
    assert span["name"] == "hone.models.image"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["hone.models.media.revised_prompt"].startswith("A lighthouse on a rocky coast")
    assert attrs["hone.models.media.outputs"][0]["sha256"] == r.files[0].sha256
    assert attrs["hone.models.cost_usd"] == 0.04
    assert attrs["hone.models.media.cost_estimated"] is True


def test_ac26_image_edit_with_references_and_url_answers(db: Path, recorded, tmp_path: Path) -> None:
    refs = [png_file(tmp_path / "front.png"), png_file(tmp_path / "side.png", 12, 12)]
    img = client("image", "test-gpt-image", db)
    with respx.mock(assert_all_mocked=True) as mock:
        edit = mock.post(f"{API}/images/edits").respond(json=recorded("openai_images_url.json"))
        one = mock.get("https://files.example.com/img-1.png").respond(
            content=png_file(tmp_path / "1.png", 4, 4).read_bytes()
        )
        mock.get("https://files.example.com/img-2.png").respond(
            content=png_file(tmp_path / "2.png", 6, 6).read_bytes()
        )
        r = img.generate(
            "the same girl, from behind", references=refs, n=2, out=tmp_path / "out" / "back.png"
        )
    body = edit.calls[0].request.content
    assert edit.calls[0].request.headers["Content-Type"].startswith("multipart/form-data")
    assert body.count(b'name="image[]"') == 2
    assert b'filename="front.png"' in body
    assert b'name="model"\r\n\r\ngpt-image-1.5' in body
    assert b'name="n"\r\n\r\n2' in body
    assert "Authorization" not in one.calls[0].request.headers  # the key never goes to another host
    assert [f.path.name for f in r.files] == ["back_1.png", "back_2.png"]
    assert [(f.width, f.height) for f in r.files] == [(4, 4), (6, 6)]
    assert r.cost_usd == 0.08
    (span,) = stored(db, b"sig=abc123")
    assert [i["path"] for i in span["attributes"]["hone.models.media.inputs"]["references"]] == [
        str(p) for p in refs
    ]


def test_ac26_image_refused_by_moderation_is_a_result(db: Path, recorded, tmp_path: Path) -> None:
    img = client("image", "test-gpt-image", db)
    with respx.mock(base_url=API) as mock:
        mock.post("/images/generations").respond(400, json=recorded("openai_images_moderation.json"))
        r = img.generate("something not allowed", out=tmp_path / "x.png")
    assert (r.files, r.error_kind) == ([], "refused")
    assert "rejected by the safety system" in (r.error or "")
    (span,) = stored(db)
    assert span["status"]["code"] == "error"
    assert span["attributes"]["hone.models.media.error_kind"] == "refused"


def test_ac26_video_polled_then_downloaded(db: Path, recorded, tmp_path: Path) -> None:
    clip = client("video", "test-sora", db)
    start = png_file(tmp_path / "shot.png")
    with respx.mock(base_url=API, assert_all_mocked=True) as mock:
        submit = mock.post("/videos").respond(json=recorded("openai_video_queued.json"))
        mock.get(f"/videos/{VIDEO_ID}").mock(
            side_effect=[
                httpx.Response(200, json=recorded("openai_video_queued.json")),
                httpx.Response(200, json=recorded("openai_video_in_progress.json")),
                httpx.Response(200, json=recorded("openai_video_completed.json")),
            ]
        )
        content = mock.get(f"/videos/{VIDEO_ID}/content").respond(content=VIDEO_BYTES)
        r = clip.generate("slow push-in", image=start, duration_s=8, out=tmp_path / "clips" / "01.mp4")
    body = submit.calls[0].request.content
    assert b'name="input_reference"; filename="shot.png"' in body
    assert b'name="seconds"\r\n\r\n8' in body
    assert b'name="size"\r\n\r\n720x1280' in body
    assert content.called
    assert r.path == tmp_path / "clips" / "01.mp4"
    assert r.files[0].path.read_bytes() == VIDEO_BYTES
    assert (r.job_id, r.error) == (VIDEO_ID, None)
    assert (r.cost_usd, r.cost_estimated) == (0.8, True)  # 8 s x 0.10 (a naive estimate)
    (span,) = stored(db, VIDEO_BYTES)
    assert span["name"] == "hone.models.video"
    assert span["attributes"]["hone.models.media.job_id"] == VIDEO_ID
    progress = [e["attributes"]["percent"] for e in span["events"] if e["name"] == "progress"]
    assert progress == [0, 45, 100]


def test_ac26_video_failed_is_a_result(db: Path, recorded, tmp_path: Path) -> None:
    clip = client("video", "test-sora", db)
    with respx.mock(base_url=API, assert_all_mocked=True) as mock:
        submit = mock.post("/videos").respond(json=recorded("openai_video_queued.json"))
        mock.get(f"/videos/{VIDEO_ID}").respond(json=recorded("openai_video_failed.json"))
        r = clip.generate("a storm at sea", out=tmp_path / "x.mp4")
    assert json.loads(submit.calls[0].request.content)["seconds"] == "4"  # the entry's default
    assert (r.files, r.job_id, r.error_kind) == ([], VIDEO_ID, "failed")
    assert "Video generation failed" in (r.error or "")
    assert r.cost_usd is None
    assert not (tmp_path / "x.mp4").exists()
    (span,) = stored(db)
    assert span["status"]["code"] == "error"


def test_ac26_video_timeout_deletes_the_job(db: Path, recorded, tmp_path: Path) -> None:
    clip = client("video", "test-sora", db)
    with respx.mock(base_url=API, assert_all_mocked=True) as mock:
        mock.post("/videos").respond(json=recorded("openai_video_queued.json"))
        mock.get(f"/videos/{VIDEO_ID}").respond(json=recorded("openai_video_in_progress.json"))
        delete = mock.delete(f"/videos/{VIDEO_ID}").respond(json={"id": VIDEO_ID, "deleted": True})
        with pytest.raises(ModelTimeout, match=f"{VIDEO_ID}.*cancelled"):
            clip.generate("a storm at sea", timeout_s=0.05, out=tmp_path / "x.mp4")
    assert delete.called
    (span,) = stored(db)
    assert any(e["name"] == "cancelled" for e in span["events"])


def test_ac26_undeclared_size_or_duration_fails_before_any_request(db: Path, tmp_path: Path) -> None:
    with respx.mock(assert_all_mocked=True) as mock:
        with pytest.raises(CapabilityError, match="not '512x512'"):
            client("image", "test-gpt-image", db).generate("x", size="512x512", out=tmp_path / "x.png")
        with pytest.raises(CapabilityError, match=r"durations of \[4\.0, 8\.0, 12\.0\] s, not 5"):
            client("video", "test-sora", db).generate("x", duration_s=5, out=tmp_path / "x.mp4")
    assert mock.calls.call_count == 0
    assert not db.exists() or stored(db) == []
