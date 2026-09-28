"""hone-models: one interface for local and hosted AI models.

import hone_models as mk
llm = mk.text("gemma4-12b")
r = llm.complete([{"role": "user", "content": "Write a haiku about rain."}])
"""

__version__ = "0.1.0"

from . import errors, gpu, records, registry, replay
from ._tracing import current_trace
from .decision import Choice, DecisionClient, ScoreQ, YesNo, decision
from .embeddings import Embedder, embedder
from .ports import PORTS_VERSION, RecordSink
from .prompt import Prompt, Section
from .runtime.session import session, unload
from .speech import SpeechClient, SpeechResult, SpeechSegment, speech
from .text import TextClient, TextResult, text

__all__ = [
    "PORTS_VERSION",
    "Choice",
    "DecisionClient",
    "Embedder",
    "Prompt",
    "RecordSink",
    "ScoreQ",
    "Section",
    "SpeechClient",
    "SpeechResult",
    "SpeechSegment",
    "TextClient",
    "TextResult",
    "YesNo",
    "__version__",
    "current_trace",
    "decision",
    "embedder",
    "errors",
    "gpu",
    "records",
    "registry",
    "replay",
    "session",
    "speech",
    "text",
    "unload",
]
