import logging

import chromadb
import numpy as np
from chromadb.api.types import IncludeEnum

from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.retrieval.schemas import RetrievedChunk, chunk_from_chroma_record

logger = logging.getLogger(__name__)


def query_dense(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[RetrievedChunk]:
    """Return the top_k chunks in collection ranked by cosine similarity to query_embedding.

    Assumes collection was created with metadata={"hnsw:space": "cosine"}, so Chroma's
    returned distance is cosine distance and 1 - distance recovers cosine similarity.
    """
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
    )
    documents, metadatas, distances = result["documents"], result["metadatas"], result["distances"]
    assert documents is not None and metadatas is not None and distances is not None
    matches = zip(result["ids"][0], documents[0], metadatas[0], distances[0])
    retrieved = [
        RetrievedChunk(
            chunk=chunk_from_chroma_record(chunk_id, text, metadata), score=1.0 - distance
        )
        for chunk_id, text, metadata, distance in matches
    ]
    logger.info("Dense retrieval returned %d candidate(s) for top_k=%d", len(retrieved), top_k)
    return retrieved
