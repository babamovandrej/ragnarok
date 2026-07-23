import logging
from pathlib import Path

import chromadb

from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.ingestion.embedder import Embedder
from src.retrieval.dense import query_dense
from src.retrieval.hybrid import fuse_rrf
from src.retrieval.rerank import Reranker
from src.retrieval.schemas import RetrievedChunk, chunk_from_chroma_record
from src.retrieval.sparse import query_sparse

logger = logging.getLogger(__name__)


def retrieve_hybrid(
    query: str,
    collection: chromadb.Collection,
    bm25_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[RetrievedChunk]:
    """Run dense+sparse retrieval, RRF fusion, and reranking for one query."""
    query_embedding = embedder.embed([query])[0]
    dense = query_dense(collection, query_embedding, top_k=top_k)
    sparse = query_sparse(bm25_dir, collection, query, top_k=top_k)
    fused = fuse_rrf(dense, sparse, top_k=top_k)
    return reranker.rerank(query, fused)


def find_symbol_definition(
    symbol_name: str, collection: chromadb.Collection
) -> RetrievedChunk | None:
    """Look up the chunk whose symbol_name exactly matches symbol_name, or None if absent."""
    result = collection.get(
        where={"symbol_name": symbol_name}, include=["documents", "metadatas"], limit=1
    )
    if not result["ids"]:
        logger.info("No chunk found defining symbol %s", symbol_name)
        return None
    chunk_id, text, metadata = result["ids"][0], result["documents"][0], result["metadatas"][0]
    return RetrievedChunk(chunk=chunk_from_chroma_record(chunk_id, text, metadata), score=1.0)
