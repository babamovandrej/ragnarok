import logging
import time

from sentence_transformers import CrossEncoder

from src.config import RERANK_MODEL_ID, RERANK_MODEL_REVISION
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Loads the pinned cross-encoder model once and reorders candidates by relevance to a query."""

    def __init__(self, device: str | None = None) -> None:
        logger.info(
            "Loading rerank model %s@%s (device=%s)",
            RERANK_MODEL_ID,
            RERANK_MODEL_REVISION,
            device or "auto",
        )
        started = time.monotonic()
        self._model: CrossEncoder = CrossEncoder(
            RERANK_MODEL_ID, revision=RERANK_MODEL_REVISION, device=device
        )
        logger.info("Loaded rerank model in %.1fs", time.monotonic() - started)

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Reorder candidates by the cross-encoder's relevance score for each (query, text) pair."""
        if not candidates:
            return []
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self._model.predict(pairs)
        reranked = [
            RetrievedChunk(chunk=candidate.chunk, score=float(score))
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda retrieved: retrieved.score, reverse=True)
        return reranked
