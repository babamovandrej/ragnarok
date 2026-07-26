from dataclasses import dataclass
from typing import cast

from chromadb import Metadata
from tree_sitter_language_pack import SupportedLanguage

from src.ingestion.schemas import Chunk, SymbolType


@dataclass(frozen=True)
class RetrievedChunk:
    """A Chunk paired with a retrieval-stage relevance score; higher is always more relevant."""

    chunk: Chunk
    score: float


def chunk_from_chroma_record(chunk_id: str, text: str, metadata: Metadata) -> Chunk:
    """Reassemble a Chunk from what Chroma stores: the id, document text, and flat metadata.

    Chroma's metadata values are typed as the wide `str | int | float | bool` union; the
    conversions below are safe no-ops in practice since our own indexer only ever writes
    str/int values, narrowed here to the precise types `Chunk` declares. `imports` reads via
    `.get` with a "" default (unlike the other, schema-version-gated fields) since it's a
    purely additive prompt-context enhancement (task 10) — an index built before it existed
    should degrade to no import context, not be rejected as incompatible.
    """
    return Chunk(
        chunk_id=chunk_id,
        file_path=str(metadata["file_path"]),
        start_line=int(metadata["start_line"]),
        end_line=int(metadata["end_line"]),
        symbol_name=str(metadata["symbol_name"]),
        qualified_name=str(metadata["qualified_name"]),
        symbol_type=cast(SymbolType, metadata["symbol_type"]),
        language=cast(SupportedLanguage, metadata["language"]),
        repo_sha=str(metadata["repo_sha"]),
        text=text,
        imports=str(metadata.get("imports", "")),
    )
