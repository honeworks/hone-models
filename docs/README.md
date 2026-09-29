# hone-models documentation

| Guide | What it covers |
|---|---|
| [concepts.md](concepts.md) | clients, the registry, results vs. exceptions, records |
| [registry.md](registry.md) | model ids, TOML files, capabilities, selection, ad-hoc ids |
| [text-and-structured-output.md](text-and-structured-output.md) | `mk.text`, params, images, schemas, the structured pipeline |
| [prompts.md](prompts.md) | `mk.Prompt` sections, versions and character spans |
| [decisions.md](decisions.md) | `mk.decision`, questions and answers, Jev, logprob calibration |
| [embeddings.md](embeddings.md) | `mk.embedder` |
| [speech.md](speech.md) | `mk.speech`, voices, long narration, `FakeSpeech` |
| [gpu-and-sessions.md](gpu-and-sessions.md) | `mk.gpu.lease`, `mk.gpu.status`, `mk.machine`, `mk.session`, `mk.unload` |
| [records-and-replay.md](records-and-replay.md) | span attributes, sinks, content capture, secrets, replay |
| [cli.md](cli.md) | `hone-models models ...` and `hone-models calls ...` |
| [testing.md](testing.md) | `FakeOllama`, `FakeSpeech`, the record sink contract, bringing your own client |

Every ` ```python ` block in these pages is executed by the test suite against the packaged
`FakeOllama` server (blocks marked ` ```python no-run ` need a real server or a hosted API key and are not run).
