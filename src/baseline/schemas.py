from dataclasses import dataclass, field
from typing import Literal

from src.agent.state import HopRecord
from src.generation.citations import Citation


@dataclass(frozen=True)
class PipelineResult:
    """A tagged answer from either the naive or full RAGnarok pipeline, for side-by-side display."""

    pipeline: Literal["naive", "ragnarok"]
    answer: str
    citations: list[Citation]
    hop_path: list[HopRecord]
    context_texts: list[str] = field(default_factory=list)
