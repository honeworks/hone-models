# pyright: reportAttributeAccessIssue=false
"""Fake numpy: records the seed."""

from types import SimpleNamespace

import torch

random = SimpleNamespace(seed=lambda seed: torch.seeds.__setitem__("numpy", seed))
