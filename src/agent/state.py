from pydantic import BaseModel, ConfigDict, Field

from src.config import AGENT_MAX_HOPS
from src.retrieval.schemas import RetrievedChunk


class HopRecord(BaseModel):
    """One resolved hop in a multi-hop trace: the symbol followed, where, and why."""

    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    reason: str


class AgentState(BaseModel):
    """Mutable state threaded through the LangGraph multi-hop agent graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    context: list[RetrievedChunk] = Field(default_factory=list)
    hop_context: list[RetrievedChunk] = Field(default_factory=list)
    hop_path: list[HopRecord] = Field(default_factory=list)
    hops_remaining: int = AGENT_MAX_HOPS
    pending_symbol: str | None = None
    pending_reason: str | None = None
    answer: str | None = None
