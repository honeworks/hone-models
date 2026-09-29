"""AC-17: the contract checks pass for every shape hone-models provides (design §10), with fake transports."""

import json
from importlib.metadata import entry_points

import pytest
import respx
from hone_flow.testing.contracts import check_gpu_lease
from hone_lens.testing.contracts import check_replayer

import hone_models as mk
from hone_models.testing import FakeOllama, check_record_sink
from select_contracts import check_decision_client, check_embedder, check_machine_probe, check_text_client

pytestmark = pytest.mark.e2e
OLLAMA = "http://127.0.0.1:11434"
OPENAI = "https://api.openai.com/v1"
JEV = "https://api.typesafe.ai/v1"
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
DECISION = {
    "q1": {"answer": "yes", "confidence": 0.95, "rationale": "It says blue."},
    "q2": {"answer": "blue", "confidence": 0.9, "rationale": "Blue."},
    "q3": {"answer": 3, "confidence": 0.6, "rationale": "Plain."},
}


def ollama_chat(request):
    """JSON when a schema is sent, plain text otherwise."""
    body = json.loads(request.content)
    schema = json.dumps(body.get("format"))
    content = "OK" if "format" not in body else json.dumps(DECISION if "q1" in schema else {"ok": True})
    return respx.MockResponse(json={"model": "m", "message": {"content": content}, "done_reason": "stop"})


def openai_chat(request):
    body = json.loads(request.content)
    content = '{"ok": true}' if "response_format" in body else "OK"
    message = {"role": "assistant", "content": content}
    return respx.MockResponse(json={"model": "m", "choices": [{"message": message, "finish_reason": "stop"}]})


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0000000000000000000")
    monkeypatch.setenv("TYPESAFE_API_KEY", "tsk-test-000000000000")


def test_ac17_text_clients(keys) -> None:
    with respx.mock() as mock:
        mock.post(f"{OLLAMA}/api/chat").mock(side_effect=ollama_chat)
        mock.post(f"{OPENAI}/chat/completions").mock(side_effect=openai_chat)
        for model in ("gemma4-12b", "gpt-4.1-mini"):
            sink = mk.records.MemorySink()
            llm = mk.text(model, sink=sink)
            check_text_client(llm)
            assert llm.complete([{"role": "user", "content": "x"}], schema=SCHEMA).parsed == {"ok": True}
            assert sink.spans[1]["trace_id"] == "a" * 32  # the checker's traceparent reached the span


def test_ac17_decision_clients(keys) -> None:
    jev = {
        "answers": [
            {"id": "q1", "p_yes": 0.97},
            {"id": "q2", "probabilities": {"blue": 0.9, "red": 0.1}},
            {"id": "q3", "score": 3.1},
        ]
    }
    with respx.mock() as mock:
        mock.post(f"{OLLAMA}/api/chat").mock(side_effect=ollama_chat)
        mock.post(f"{JEV}/decide").respond(json=jev)
        for model in ("gemma4-12b", "jev"):
            check_decision_client(mk.decision(model, sink=mk.records.NullSink()))
        answers = mk.decision("jev", sink=mk.records.NullSink()).decide("x", {"q1": mk.YesNo("?")})
        assert answers["q1"]["calibrated"] is True


def test_ac17_embedders(keys) -> None:
    with respx.mock() as mock:
        mock.post(f"{OLLAMA}/api/embed").respond(json={"embeddings": [[1.0, 2.0], [3.0, 4.0]]})
        rows = [{"index": 1, "embedding": [0.0, 5.0]}, {"index": 0, "embedding": [2.0, 0.0]}]
        mock.post(f"{OPENAI}/embeddings").respond(json={"data": rows})
        check_embedder(mk.embedder("ollama:tiny-embed", sink=mk.records.NullSink()))
        check_embedder(mk.embedder("openai:text-embedding-3-small", sink=mk.records.NullSink()))


def test_ac17_gpu_leases(isolated, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mk.gpu.GPU, "memory", lambda: (8192, 0))
    check_gpu_lease(mk.gpu)  # module-level lease
    check_gpu_lease(mk.gpu.GPU)
    check_gpu_lease(mk.gpu.NullGpuLease())
    check_gpu_lease(mk.gpu.FileLockGpuLease(isolated / "gpu.lock"))


def test_ac17_machine_probe(isolated, monkeypatch: pytest.MonkeyPatch) -> None:
    """hone-select's `MachineProbe` checker against `mk.machine.MACHINE` and the entry point's factory."""
    monkeypatch.setenv("HONE_GPU_LOCK", str(isolated / "gpu.lock"))  # never the machine-wide lock
    gpu = {"index": 0, "name": "Fake GPU", "memory_total_mb": 8192, "memory_used_mb": 0, "utilization_pct": 0}
    monkeypatch.setattr(mk.machine.MACHINE, "gpus", lambda: [gpu])  # no real GPU readings
    monkeypatch.setattr(mk.machine.MACHINE, "processes", dict)
    monkeypatch.setattr(mk.gpu.GPU, "memory", lambda: None)
    loaded = {"models": [{"name": "x:latest", "size": 1, "size_vram": 1}]}
    with FakeOllama(responder=lambda path, body: loaded if path == "/api/ps" else None):
        check_machine_probe(mk.machine.MACHINE)
        (ep,) = [ep for ep in entry_points(group="hone.machine_probes") if ep.name == "hone_models"]
        check_machine_probe(ep.load()(gpus=lambda: None, processes=dict, sink=mk.records.NullSink()))


def test_ac17_record_sinks(isolated) -> None:
    sqlite = mk.records.SqliteSpanSink(isolated / "s.db")
    check_record_sink(sqlite, lambda: mk.records.read_spans(isolated / "s.db"))
    jsonl = mk.records.JsonlSpanSink(isolated / "s.jsonl")
    check_record_sink(jsonl, lambda: [json.loads(x) for x in (isolated / "s.jsonl").read_text().splitlines()])
    memory = mk.records.MemorySink()
    check_record_sink(memory, lambda: memory.spans)


def test_ac17_replayer() -> None:
    """hone-lens's checker: its example span (JSON-string attributes, sections without roles) and ours."""
    with FakeOllama():
        sink = mk.records.MemorySink()
        check_replayer(mk.replay.Replayer(sink=sink))
        prompt = mk.Prompt("p", "1", {"system": "Be brief.", "format_example": "A storm.", "brief": "Fish."})
        mk.text("gemma4-12b", sink=sink).complete(prompt)
        check_replayer(mk.replay.Replayer(sink=sink), sink.spans[-1])
    assert sink.spans[-1]["attributes"]["hone.lens.finding_id"] == "F-0001"  # the checker's trace context


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("hone.text_clients", mk.text),
        ("hone.decision_clients", mk.decision),
        ("hone.embedders", mk.embedder),
        ("hone.gpu_leases", mk.gpu.GPU),
        ("hone.replayers", mk.replay.Replayer),
        ("hone.machine_probes", mk.machine.Machine),
    ],
)
def test_ac17_entry_points(group, expected) -> None:
    (ep,) = [ep for ep in entry_points(group=group) if ep.name == "hone_models"]
    assert ep.load() is expected
