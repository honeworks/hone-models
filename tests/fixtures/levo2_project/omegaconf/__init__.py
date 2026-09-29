"""Fake omegaconf: remembers the registered resolvers."""

from types import SimpleNamespace

resolvers: dict = {}
OmegaConf = SimpleNamespace(register_new_resolver=resolvers.__setitem__, load=lambda path: {})
