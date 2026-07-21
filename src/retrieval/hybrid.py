import logging

from src.config import DEFAULT_RETRIEVAL_TOP_K, HYBRID_RRF_K
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


def fuse_rrf(
    *ranked_lists: list[RetrievedChunk],
    rrf_k: int = HYBRID_RRF_K,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[RetrievedChunk]:
    """Fuse ranked retrieval lists via Reciprocal Rank Fusion, deduped by chunk_id."""
    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}
    for ranked_list in ranked_lists:
        for rank, retrieved in enumerate(ranked_list, start=1):
            chunk_id = retrieved.chunk.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks_by_id.setdefault(chunk_id, retrieved.chunk)

    fused = [
        RetrievedChunk(chunk=chunks_by_id[chunk_id], score=score)
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda retrieved: retrieved.score, reverse=True)
    logger.info(
        "RRF fusion of %d ranked list(s) produced %d unique chunk(s)",
        len(ranked_lists),
        len(fused),
    )
    return fused[:top_k]
