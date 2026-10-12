import logging
import re
from pathlib import Path

import chromadb
from chromadb.api.types import IncludeEnum, Metadata

from src.config import (
    AGENT_HOP_CANDIDATE_LIMIT,
    DEFAULT_RETRIEVAL_TOP_K,
    MIN_CONTEXT_RELEVANCE_SCORE,
)
from src.ingestion.embedder import Embedder
from src.retrieval.dense import query_dense
from src.retrieval.hybrid import fuse_rrf
from src.retrieval.rerank import Reranker
from src.retrieval.schemas import RetrievedChunk, chunk_from_chroma_record
from src.retrieval.sparse import query_sparse

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SYMBOL_TYPE_PRIORITY = {"class": 0, "interface": 0, "struct": 0, "type": 0, "function": 1}
_LOWEST_PRIORITY = 2


def retrieve_hybrid(
    query: str,
    collection: chromadb.Collection,
    bm25_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[RetrievedChunk]:
    """Run dense+sparse retrieval, RRF fusion, and reranking for one query."""
    query_embedding = embedder.embed_query(query)
    dense = query_dense(collection, query_embedding, top_k=top_k)
    sparse = query_sparse(bm25_dir, collection, query, top_k=top_k)
    fused = fuse_rrf(dense, sparse, top_k=top_k)
    return reranker.rerank(query, fused)


def find_symbol_definition(
    symbol_name: str, collection: chromadb.Collection
) -> RetrievedChunk | None:
    """Look up the chunk whose symbol_name exactly matches symbol_name, or None if absent."""
    result = collection.get(
        where={"symbol_name": symbol_name},
        include=[IncludeEnum.documents, IncludeEnum.metadatas],
        limit=1,
    )
    if not result["ids"]:
        logger.info("No chunk found defining symbol %s", symbol_name)
        return None
    documents, metadatas = result["documents"], result["metadatas"]
    assert documents is not None and metadatas is not None
    chunk_id, text, metadata = result["ids"][0], documents[0], metadatas[0]
    return RetrievedChunk(chunk=chunk_from_chroma_record(chunk_id, text, metadata), score=1.0)


def _is_inside(definition: Metadata, context: list[RetrievedChunk]) -> bool:
    """True if definition's line range lies within a chunk already in context."""
    return any(
        retrieved.chunk.file_path == definition["file_path"]
        and retrieved.chunk.start_line <= int(definition["start_line"])
        and int(definition["end_line"]) <= retrieved.chunk.end_line
        for retrieved in context
    )


def referenced_symbols(
    context: list[RetrievedChunk],
    collection: chromadb.Collection,
    limit: int = AGENT_HOP_CANDIDATE_LIMIT,
) -> list[str]:
    """Names the context's code refers to that a lookup could usefully fetch.

    A name qualifies when the repo defines it exactly once (a by-name lookup can't tell
    several definitions apart) and that definition isn't already in context, whether as a
    retrieved chunk or nested inside one. Earlier context chunks contribute first; within a
    chunk, types come before functions and functions before methods, since a bare
    method-like name in code is more often a local variable than a reference.
    """
    metadatas = collection.get(include=[IncludeEnum.metadatas])["metadatas"]
    assert metadatas is not None
    definitions: dict[str, list[Metadata]] = {}
    for metadata in metadatas:
        definitions.setdefault(str(metadata["symbol_name"]), []).append(metadata)

    candidates: list[str] = []
    for retrieved in context:
        priorities: dict[str, int] = {}
        for name in _IDENTIFIER.findall(retrieved.chunk.text):
            matches = definitions.get(name, [])
            if name in priorities or name in candidates or len(matches) != 1:
                continue
            if _is_inside(matches[0], context):
                continue
            symbol_type = str(matches[0]["symbol_type"])
            priorities[name] = _SYMBOL_TYPE_PRIORITY.get(symbol_type, _LOWEST_PRIORITY)
        candidates.extend(sorted(priorities, key=priorities.__getitem__))
    return candidates[:limit]


class RelevantReferences:
    """Finds the definitions a question's context refers to that are worth fetching.

    referenced_symbols finds every definition the retrieved code mentions, and most are
    incidental: a type in a signature, a base class. The reranker scores each definition
    against the question, and only those at or above MIN_CONTEXT_RELEVANCE_SCORE are
    returned, most relevant first. Scores are kept between calls, because the agent asks
    again after every hop and most candidates repeat.
    """

    def __init__(self, question: str, collection: chromadb.Collection, reranker: Reranker) -> None:
        self._question = question
        self._collection = collection
        self._reranker = reranker
        self._scored: dict[str, RetrievedChunk] = {}

    def __call__(self, context: list[RetrievedChunk]) -> list[RetrievedChunk]:
        definitions = []
        for name in referenced_symbols(context, self._collection):
            found = find_symbol_definition(name, self._collection)
            if found is not None:
                definitions.append(found)

        unscored = [d for d in definitions if d.chunk.chunk_id not in self._scored]
        for scored in self._reranker.rerank(self._question, unscored):
            self._scored[scored.chunk.chunk_id] = scored

        candidates = [self._scored[d.chunk.chunk_id] for d in definitions]
        relevant = [c for c in candidates if c.score >= MIN_CONTEXT_RELEVANCE_SCORE]
        logger.info(
            "%d of %d referenced definition(s) are relevant to the question",
            len(relevant),
            len(candidates),
        )
        return sorted(relevant, key=lambda candidate: candidate.score, reverse=True)
