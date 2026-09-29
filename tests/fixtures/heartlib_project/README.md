A fake heartlib project folder for tests/integration/test_heartmula_command.py: stand-ins for torch, numpy,
tokenizers, transformers, soundfile and heartlib's pipeline module, found first because the test entry
puts this folder on `PYTHONPATH`. The fake pipeline records what it was given; the fake `soundfile.write`
writes that record, as JSON, as the song.
