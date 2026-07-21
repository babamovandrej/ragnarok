import logging
from pathlib import Path

import bm25s
import chromadb

from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.retrieval.schemas import RetrievedChunk, chunk_from_chroma_record

logger = logging.getLogger(__name__)


def query_sparse(
    bm25_dir: Path,
    collection: chromadb.Collection,
    query_text: str,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[RetrievedChunk]:
    """Return the top_k chunks from the bm25s index at bm25_dir, ranked by BM25 relevance."""
    retriever = bm25s.BM25.load(str(bm25_dir), load_corpus=True)
    num_docs = retriever.scores["num_docs"]
    query_tokens = bm25s.tokenize([query_text], show_progress=False)
    results, scores = retriever.retrieve(
        query_tokens, k=min(top_k, num_docs), show_progress=False
    )

    chunk_ids = [str(match["text"]) for match in results[0]]
    fetched = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    chunks_by_id = {
        chunk_id: chunk_from_chroma_record(chunk_id, text, metadata)
        for chunk_id, text, metadata in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"]
        )
    }
    retrieved = [
        RetrievedChunk(chunk=chunks_by_id[chunk_id], score=float(score))
        for chunk_id, score in zip(chunk_ids, scores[0])
        if chunk_id in chunks_by_id
    ]
    logger.info("Sparse retrieval returned %d candidate(s) for top_k=%d", len(retrieved), top_k)
    return retrieved
