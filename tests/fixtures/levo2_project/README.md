A fake SongGeneration (LeVo 2) project folder for tests/integration/test_levo2_command.py: stand-ins for
torch, numpy, omegaconf, the project's `generate.py` and `codeclm.utils.offload_profiler`, found first
because the adapter puts the project folder at the front of `sys.path`. The fake `generate_lowmem`
writes `audios/<idx>.flac` holding, as JSON, what it was given.
