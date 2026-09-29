"""The ComfyUI provider's edge cases: workflows, older servers, polling, server start, unload, the fake."""

import json
import sys
from pathlib import Path

import httpx
import pytest
import respx

import hone_models as mk
from hone_models import _comfyui_loaded, _gpu_room
from hone_models.errors import CapabilityError, ConfigError, ProviderError
from hone_models.providers import _comfyui_server, _comfyui_workflow, comfyui
from hone_models.testing import FakeComfyUI
from media_fixtures import COMFYUI_FIXTURES, LeaseRecorder, png_file, registry


@pytest.fixture(autouse=True)
def lease(monkeypatch: pytest.MonkeyPatch) -> LeaseRecorder:
    recorder = LeaseRecorder()
    monkeypatch.setattr(mk.gpu, "GPU", recorder)
    return recorder


def entry(tmp_path: Path, extra: str, workflow: object | None = None) -> mk.registry.Registry:
    if workflow is not None:
        (tmp_path / "wf.json").write_text(json.dumps(workflow))
    path = tmp_path / "m.toml"
    path.write_text(f'[models.m]\nprovider = "comfyui"\nkind = "image"\n{extra}\n')
    return mk.registry.load([path])


def test_workflow_problems_are_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no workflow yet for"):
        _comfyui_workflow.load(entry(tmp_path, "").get("m"))
    with pytest.raises(ConfigError, match="cannot read"):
        _comfyui_workflow.load(entry(tmp_path, 'workflow = "missing.json"').get("m"))
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ConfigError, match="not JSON"):
        _comfyui_workflow.load(entry(tmp_path, 'workflow = "bad.json"').get("m"))
    ui_format = {"nodes": [], "links": []}
    with pytest.raises(ConfigError, match="API format"):
        _comfyui_workflow.load(entry(tmp_path, 'workflow = "wf.json"', ui_format).get("m"))


def test_mapping_problems_are_config_errors(tmp_path: Path) -> None:
    flow = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}}
    cfg = entry(
        tmp_path, 'workflow = "wf.json"\ninputs = { prompt = "2.text", steps = "1.steps" }', flow
    ).get("m")
    workflow, _ = _comfyui_workflow.load(cfg)
    with pytest.raises(ConfigError, match="has no node '2'"):
        _comfyui_workflow.fill(workflow, cfg, {"prompt": "x"})
    with pytest.raises(ConfigError, match="has no input 'steps' on node '1' \\(CLIPTextEncode\\)"):
        _comfyui_workflow.fill(workflow, cfg, {"steps": 3})
    bare = entry(tmp_path, 'workflow = "wf.json"', flow).get("m")
    with pytest.raises(ConfigError, match="maps no `prompt`"):
        _comfyui_workflow.fill(workflow, bare, {"prompt": "x", "seed": 1})
    assert _comfyui_workflow.fill(workflow, bare, {"prompt": "", "seed": 1}) == workflow  # nothing mapped


def test_more_references_than_slots(tmp_path: Path) -> None:
    flow = {"1": {"class_type": "LoadImage", "inputs": {"image": ""}}}
    cfg = entry(tmp_path, 'workflow = "wf.json"\ninputs = { references = ["1.image"] }', flow).get("m")
    with pytest.raises(CapabilityError, match="1 reference slots, not 2"):
        _comfyui_workflow.fill(flow, cfg, {"references": ["a", "b"]})


def test_a_file_input_mapped_to_slots_is_optional(tmp_path: Path) -> None:
    flow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "3": {"class_type": "Latent", "inputs": {"start_image": ["1", 0], "end_image": ["2", 0]}},
    }
    inputs = 'inputs = { image = ["1.image"], source = "2.image" }'
    cfg = entry(tmp_path, f'workflow = "wf.json"\n{inputs}', flow).get("m")
    left_out = _comfyui_workflow.fill(flow, cfg, {})
    assert "1" not in left_out  # a slot: removed, with the link to it
    assert left_out["3"]["inputs"] == {"end_image": ["2", 0]}
    assert left_out["2"]["inputs"]["image"] == "placeholder.png"  # one path: required, kept as it is
    given = _comfyui_workflow.fill(flow, cfg, {"image": "hone/a.png"})
    assert given["1"]["inputs"]["image"] == "hone/a.png"
    assert given["3"] == flow["3"]


def test_video_outputs_carry_flags_next_to_the_files() -> None:
    # SaveVideo's history entry: {"images": [...files], "animated": [true]}
    video = {"filename": "clip_00001_.mp4", "subfolder": "hone", "type": "output"}
    outputs = {"58": {"images": [video], "animated": [True]}}
    assert comfyui._output_files(outputs, ["58"]) == [video]
    assert comfyui._output_files(outputs, None) == [video]


def test_older_server_cancel_and_default_outputs(tmp_path: Path) -> None:
    img = mk.image("test-image", registry=registry())
    with FakeComfyUI(jobs_api=False) as server:
        server.queue("hang")
        with pytest.raises(mk.errors.ModelTimeout):
            img.generate("x", timeout_s=0.2, out=tmp_path / "x.png")
        paths = [q["path"] for q in server.requests]
        assert paths[-3:] == ["/queue", "/interrupt", "/free"]  # it was running: interrupted
        server.queue("no_output")
        r = img.generate("y", out=tmp_path / "y.png")
    assert (r.error_kind, r.files) == ("no_output", [])


def test_pending_job_on_older_server_is_dequeued() -> None:
    with respx.mock(base_url="http://c") as mock:
        mock.post("/api/jobs/j1/cancel").respond(404)
        mock.get("/queue").respond(json={"queue_running": [], "queue_pending": [[1, "j1"]]})
        delete = mock.post("/queue").respond(json={})
        comfyui._cancel("http://c", "j1")
    assert json.loads(delete.calls.last.request.content) == {"delete": ["j1"]}


def test_cancel_never_raises() -> None:
    with respx.mock(base_url="http://c") as mock:
        mock.post("/api/jobs/j1/cancel").mock(side_effect=httpx.ConnectError("gone"))
        comfyui._cancel("http://c", "j1")  # logged only


def test_polling_tolerates_a_few_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comfyui, "POLL_S", 0.0)
    with respx.mock(base_url="http://c") as mock:
        mock.get("/history/j1").mock(
            side_effect=[httpx.Response(500), httpx.Response(200, json={"j1": {"a": 1}})]
        )
        assert comfyui._wait("http://c", "j1", float("inf")) == {"a": 1}
        mock.get("/history/j2").respond(503)
        mock.post("/api/jobs/j2/cancel").respond(json={"cancelled": True})
        with pytest.raises(ProviderError, match="lost ComfyUI job j2"):
            comfyui._wait("http://c", "j2", float("inf"))


def test_interrupted_job_and_validation_messages() -> None:
    entry_ = {"status": {"messages": [["execution_interrupted", {}], ["bad"]]}, "outputs": {}}
    outcome = comfyui._outcome("http://c", "j", entry_, registry().get("test-image"), Path("x.png"))
    assert (outcome.error, outcome.error_kind) == ("the job was interrupted", "failed")
    assert comfyui._node_errors(httpx.Response(400, text="plain")) == "plain"
    body = {
        "error": {"message": "Cannot execute because node Foo does not exist.", "details": "Node ID '#3'"}
    }
    assert comfyui._node_errors(httpx.Response(400, json=body)).endswith("does not exist.: Node ID '#3'")
    outputs = {
        "9": {"images": [{"filename": "a.png", "type": "output"}, {"filename": "p.png", "type": "temp"}]}
    }
    assert [f["filename"] for f in comfyui._output_files(outputs, None)] == ["a.png"]


def test_transport_failures_are_provider_errors(tmp_path: Path) -> None:
    img = mk.image("test-image", registry=registry())
    with FakeComfyUI() as server, respx.mock(assert_all_called=False) as mock:
        mock.post(f"{server.url}/prompt").mock(side_effect=httpx.ConnectError("reset"))
        mock.route(host="127.0.0.1").pass_through()
        with pytest.raises(ProviderError, match="did not take the job"):
            img.generate("x", out=tmp_path / "x.png")
    with FakeComfyUI() as server, respx.mock(assert_all_called=False) as mock:
        mock.post(f"{server.url}/upload/image").respond(500)
        mock.route(host="127.0.0.1").pass_through()
        with pytest.raises(ProviderError, match="could not upload: HTTP 500"):
            img.generate("x", references=[png_file(tmp_path / "r.png", 3, 3)], out=tmp_path / "x.png")


def test_unload_frees_and_clears_the_list(tmp_path: Path) -> None:
    with FakeComfyUI() as server:
        _comfyui_loaded.add(server.url, "test-image", "j1")
        mk.unload("test-image", registry=registry())
        assert server.frees == 1
        assert _comfyui_loaded.read() == {}
    with pytest.raises(ProviderError, match="did not free"):
        mk.unload("test-image", registry=registry())  # no server any more


def test_release_hook_is_registered(tmp_path: Path) -> None:
    with FakeComfyUI() as server:
        mk.image("test-image", registry=registry()).generate("x", out=tmp_path / "x.png")
        hook = _gpu_room._HOOKS.pop(f"comfyui:{server.url}")
        _comfyui_loaded.add(server.url, "test-image", "j1")
        hook()
        assert _comfyui_loaded.read() == {}


def test_server_start_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONE_COMFYUI_START", raising=False)
    with monkeypatch.context() as m:
        m.setattr(_comfyui_server, "healthy", lambda url: False)  # no DNS lookup for the remote name
        with (
            pytest.raises(ProviderError, match="remote: not started here"),
            _comfyui_server.running("http://gpu-box:1"),
        ):
            pass
    with (
        pytest.raises(ProviderError, match="HONE_COMFYUI_START is not set"),
        _comfyui_server.running("http://127.0.0.1:1"),
    ):
        pass
    monkeypatch.setenv("HONE_COMFYUI_START", str(tmp_path / "missing-start.sh"))
    with pytest.raises(ConfigError, match="cannot be run"), _comfyui_server.running("http://127.0.0.1:1"):
        pass
    monkeypatch.setenv("HONE_COMFYUI_START", f"{sys.executable} -c 'raise SystemExit(4)'")
    with (
        pytest.raises(ProviderError, match="exited with code 4"),
        _comfyui_server.running("http://127.0.0.1:1"),
    ):
        pass
    monkeypatch.setenv("HONE_COMFYUI_START", f"{sys.executable} -c 'import time; time.sleep(30)'")
    monkeypatch.setenv("HONE_COMFYUI_START_S", "0.3")
    with pytest.raises(ProviderError, match="did not answer"), _comfyui_server.running("http://127.0.0.1:1"):
        pass


def test_fake_comfyui_edges() -> None:
    with FakeComfyUI() as server:
        assert httpx.get(f"{server.url}/view", params={"filename": "nope"}).status_code == 404
        assert httpx.get(f"{server.url}/nothing").status_code == 404
        assert httpx.get(f"{server.url}/api/jobs/unknown").status_code == 404
        assert httpx.get(f"{server.url}/history").json() == {}
        with pytest.raises(ValueError, match="unknown outcome"):
            server.queue("explode")
        broken = json.loads((COMFYUI_FIXTURES / "music.json").read_text())
        del broken["109"]
        refused = httpx.post(f"{server.url}/prompt", json={"prompt": broken})
        assert refused.status_code == 400
        assert refused.json()["node_errors"]["3"]["errors"][0]["extra_info"]["input_name"] == "seed"
        server.queue("ok", run_s=5)
        job = httpx.post(f"{server.url}/prompt", json={"prompt": {}}).json()["prompt_id"]
        assert httpx.get(f"{server.url}/api/jobs/{job}").json()["status"] == "in_progress"
        assert httpx.post(f"{server.url}/api/jobs/{job}/cancel").json() == {"cancelled": True}
        assert httpx.get(f"{server.url}/api/jobs/{job}").json()["status"] == "cancelled"
