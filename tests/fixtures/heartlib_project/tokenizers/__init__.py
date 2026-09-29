"""Fake tokenizers."""

from types import SimpleNamespace

Tokenizer = SimpleNamespace(from_file=lambda path: f"tokenizer:{path}")
