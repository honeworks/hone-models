"""The hosted images and video provider's edge cases: retries, polling, refusals, answers it cannot use."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

import hone_models as mk
from hone_models.errors import ConfigError, ProviderError
from hone_models.providers import openai_media
from media_fixtures import LeaseRecorder, png_file

API = "https://api.openai.com/v1"
VIDEO_ID = "video_68d7512d07848190b3e45da0ecbebcde"
HOSTED = Path(__file__).parents[1] / "fixtures" / "hosted" / "models.toml"


@pytest.fixture(autouse=True)
def hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_media, "POLL_S", 0.0)
    monkeypatch.setattr("hone_models._http.BACKOFF_S", 0.0)


def make(
    kind: str, model_id: str = "", extra: str | None = None, tmp_path: Path | None = None
) -> mk.MediaClient:
    registry = mk.registry.load([HOSTED])
    if extra is not None:
        assert tmp_path is not None
        (tmp_path / "m.toml").write_text(
            f'[models.m]\nprovider = "openai_compatible"\nkind = "{kind}"\n{extra}\n'
        )
        registry = mk.registry.load([HOSTED, tmp_path / "m.toml"])
        model_id = "m"
    factory = {"image": mk.image, "video": mk.video, "music": mk.music}[kind]
    return factory(model_id, registry=registry, sink=mk.records.MemorySink())


def span(client: mk.MediaClient) -> dict:
    (only,) = client.sink.spans  # type: ignore[attr-defined]
    return only


def test_inputs_per_kind() -> None:
    assert make("image", "test-gpt-image").inputs == ["background", "n", "quality", "references", "size"]
    assert make("video", "test-sora").inputs == ["duration_s", "image", "size"]


def test_music_is_not_a_hosted_kind(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="makes images and video, not music"):
        make("music", extra='base_url = "https://api.openai.com/v1"', tmp_path=tmp_path).generate(
            "x", out=tmp_path / "x.wav"
        )


def test_missing_base_url_or_key_fail_before_any_request(tmp_path: Path, monkeypatch) -> None:
    with respx.mock(assert_all_mocked=True) as mock:
        with pytest.raises(ConfigError, match="needs base_url"):
            make("image", extra="", tmp_path=tmp_path).generate("x", out=tmp_path / "x.png")
        monkeypatch.delenv("OPENAI_API_KEY")
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            make("image", "test-gpt-image").generate("x", out=tmp_path / "x.png")
    assert mock.calls.call_count == 0


def test_submit_retries_a_429_once_then_succeeds(recorded, tmp_path: Path) -> None:
    img = make("image", "test-gpt-image")
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        mock.post("/images/generations").mock(
            side_effect=[
                httpx.Response(429, json=recorded("openai_rate_limit.json")),
                httpx.Response(200, json=recorded("openai_images_b64.json")),
            ]
        )
        r = img.generate("x", out=tmp_path / "x.png")
    assert r.error is None
    assert r.path is not None
    assert [e["name"] for e in span(img)["events"]] == ["retry"]


def test_other_client_errors_raise(tmp_path: Path) -> None:
    img = make("image", "test-gpt-image")
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        mock.post("/images/generations").respond(401, json={"error": {"message": "Incorrect API key"}})
        with pytest.raises(ProviderError, match="HTTP 401"):
            img.generate("x", out=tmp_path / "x.png")


def test_extra_inputs_go_into_the_edit_form_and_output_format_names_the_suffix(
    recorded, tmp_path: Path
) -> None:
    img = make(
        "image", extra='base_url = "https://api.openai.com/v1"\ninputs = ["output_format"]', tmp_path=tmp_path
    )
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        edit = mock.post("/images/edits").respond(json=recorded("openai_images_b64.json"))
        r = img.generate(
            "x", references=[str(png_file(tmp_path / "a.png"))], output_format="jpeg", out=tmp_path / "o"
        )
    assert b'name="output_format"\r\n\r\njpeg' in edit.calls[0].request.content
    assert r.path == tmp_path / "o.jpg"


def test_unusable_image_answers(tmp_path: Path) -> None:
    img = make("image", "test-gpt-image")
    with respx.mock(assert_all_mocked=True) as mock:
        mock.post(f"{API}/images/generations").respond(json={"data": [{"revised_prompt": "x"}]})
        with pytest.raises(ProviderError, match="neither b64_json nor url"):
            img.generate("x", out=tmp_path / "x.png")
        mock.post(f"{API}/images/generations").respond(
            json={"data": [{"url": "https://files.example.com/a.png?sig=s"}]}
        )
        mock.get("https://files.example.com/a.png").respond(403)
        with pytest.raises(
            ProviderError, match="could not fetch an image the provider returned: HTTP 403"
        ) as err:
            img.generate("x", out=tmp_path / "x.png")
        assert "sig=" not in str(err.value)
        mock.get("https://files.example.com/a.png").mock(side_effect=httpx.ReadError("reset"))
        with pytest.raises(ProviderError, match="ReadError"):
            img.generate("x", out=tmp_path / "x.png")
        mock.post(f"{API}/images/generations").respond(json={"data": []})
        assert img.generate("x", out=tmp_path / "x.png").error_kind == "no_output"


def test_a_local_server_takes_a_lease(recorded, tmp_path: Path, monkeypatch) -> None:
    recorder = LeaseRecorder()
    monkeypatch.setattr(mk.gpu, "GPU", recorder)
    img = make(
        "image",
        extra='base_url = "http://127.0.0.1:8000/v1"\ncapabilities = { vram_gb = 5.0 }',
        tmp_path=tmp_path,
    )
    with respx.mock(base_url="http://127.0.0.1:8000/v1") as mock:
        mock.post("/images/generations").respond(json=recorded("openai_images_b64.json"))
        img.generate("x", out=tmp_path / "x.png")
    assert recorder.calls == [("m", 5.0)]


def interrupt(request: httpx.Request) -> httpx.Response:
    raise KeyboardInterrupt


def video_routes(mock: respx.MockRouter, recorded, polls: Any) -> respx.Route:
    mock.post("/videos").respond(json=recorded("openai_video_queued.json"))
    mock.get(f"/videos/{VIDEO_ID}").mock(side_effect=polls)
    return mock.delete(f"/videos/{VIDEO_ID}").respond(json={"deleted": True})


def test_video_refused_by_moderation_is_a_result(recorded, tmp_path: Path) -> None:
    clip = make("video", "test-sora")
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        video_routes(mock, recorded, [httpx.Response(200, json=recorded("openai_video_moderation.json"))])
        r = clip.generate("x", out=tmp_path / "x.mp4")
    assert (r.error_kind, r.job_id) == ("refused", VIDEO_ID)
    assert r.error == "moderation_blocked: Your request was blocked by our moderation system."


def test_polling_tolerates_transient_errors(recorded, tmp_path: Path) -> None:
    clip = make("video", "test-sora")
    busy = [
        httpx.Response(503),
        httpx.ConnectError("down"),
        httpx.Response(429),
        httpx.Response(200, text="<html>"),
    ]
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        delete = video_routes(
            mock, recorded, [*busy, httpx.Response(200, json=recorded("openai_video_completed.json"))]
        )
        mock.get(f"/videos/{VIDEO_ID}/content").respond(content=b"video")
        r = clip.generate("x", out=tmp_path / "x.mp4")
    assert r.error is None
    assert r.path is not None
    assert not delete.called


def test_five_failed_polls_in_a_row_give_up_with_the_job_id(recorded, tmp_path: Path) -> None:
    clip = make("video", "test-sora")
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        submit = mock.post("/videos").respond(json=recorded("openai_video_queued.json"))
        delete = video_routes(mock, recorded, [httpx.Response(502)] * 5)
        with pytest.raises(ProviderError, match=f"lost track of video job {VIDEO_ID} after 5 failed polls"):
            clip.generate("x", out=tmp_path / "x.mp4")
    assert delete.called
    assert submit.call_count == 1  # submitted once, never again


def test_a_client_error_while_polling_raises_and_deletes(recorded, tmp_path: Path) -> None:
    clip = make("video", "test-sora")
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        delete = video_routes(mock, recorded, [httpx.Response(404, json={"error": {"message": "not found"}})])
        with pytest.raises(ProviderError, match="HTTP 404"):
            clip.generate("x", out=tmp_path / "x.mp4")
    assert delete.called


def test_an_interrupt_while_waiting_deletes_the_job(recorded, tmp_path: Path) -> None:
    clip = make("video", "test-sora")
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        delete = video_routes(mock, recorded, interrupt)
        with pytest.raises(KeyboardInterrupt):
            clip.generate("x", out=tmp_path / "x.mp4")
    assert delete.called


def test_a_failed_delete_is_logged(recorded, tmp_path: Path, caplog) -> None:
    clip = make("video", "test-sora")
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        video_routes(mock, recorded, interrupt)
        mock.delete(f"/videos/{VIDEO_ID}").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(KeyboardInterrupt):
            clip.generate("x", out=tmp_path / "x.mp4")
    assert f"could not delete video job {VIDEO_ID}" in caplog.text


def test_video_answers_it_cannot_use(recorded, tmp_path: Path) -> None:
    clip = make("video", "test-sora")
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        mock.post("/videos").respond(json={"status": "queued"})
        with pytest.raises(ProviderError, match="no job id"):
            clip.generate("x", out=tmp_path / "x.mp4")
        video_routes(mock, recorded, [httpx.Response(200, json={"id": VIDEO_ID, "status": "cancelled"})])
        r = clip.generate("x", out=tmp_path / "x.mp4")
    assert (r.error, r.error_kind) == ("the video job ended with status 'cancelled'", "failed")


def test_fractional_seconds_are_sent_as_given(recorded, tmp_path: Path) -> None:
    clip = make("video", extra='base_url = "https://api.openai.com/v1"', tmp_path=tmp_path)
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        submit = mock.post("/videos").respond(json=recorded("openai_video_queued.json"))
        mock.get(f"/videos/{VIDEO_ID}").respond(json=recorded("openai_video_failed.json"))
        clip.generate("x", duration_s=2.5, out=tmp_path / "x.mp4")
    assert json.loads(submit.calls[0].request.content)["seconds"] == "2.5"
