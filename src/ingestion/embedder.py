import logging
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
)

logger = logging.getLogger(__name__)


class Embedder:
    """Loads the pinned code-embedding model once and embeds chunk texts in batches."""

    def __init__(self, device: str | None = None, batch_size: int | None = None) -> None:
        self._batch_size = batch_size if batch_size is not None else EMBEDDING_BATCH_SIZE
        logger.info(
            "Loading embedding model %s@%s (device=%s)",
            EMBEDDING_MODEL_ID,
            EMBEDDING_MODEL_REVISION,
            device or "auto",
        )
        started = time.monotonic()
        self._model: SentenceTransformer = SentenceTransformer(
            EMBEDDING_MODEL_ID,
            revision=EMBEDDING_MODEL_REVISION,
            trust_remote_code=True,
            device=device,
        )
        self._model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
        logger.info("Loaded embedding model in %.1fs", time.monotonic() - started)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of chunk texts, returning one dense vector per text."""
        logger.info("Embedding %d text(s) in batches of %d", len(texts), self._batch_size)
        started = time.monotonic()
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        logger.info("Embedded %d text(s) in %.1fs", len(texts), time.monotonic() - started)
        return embeddings
