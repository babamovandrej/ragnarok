from collections.abc import Mapping
from dataclasses import dataclass

from src.ingestion.schemas import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    """A Chunk paired with a retrieval-stage relevance score; higher is always more relevant."""

    chunk: Chunk
    score: float


def chunk_from_chroma_record(chunk_id: str, text: str, metadata: Mapping[str, str | int]) -> Chunk:
    """Reassemble a Chunk from what Chroma stores: the id, document text, and flat metadata."""
    return Chunk(
        chunk_id=chunk_id,
        file_path=metadata["file_path"],
        start_line=metadata["start_line"],
        end_line=metadata["end_line"],
        symbol_name=metadata["symbol_name"],
        qualified_name=metadata["qualified_name"],
        symbol_type=metadata["symbol_type"],
        language=metadata["language"],
        text=text,
    )
