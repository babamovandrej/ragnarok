import logging
from pathlib import Path

from src.baseline.schemas import PipelineResult
from src.config import DEFAULT_RETRIEVAL_TOP_K, GENERATION_MAX_TOKENS_DEFAULT
from src.generation.llm import LLMBackend
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import load_chroma_collection
from src.retrieval.dense import query_dense

logger = logging.getLogger(__name__)


def answer_naive(
    question: str,
    cache_dir: Path,
    embedder: Embedder,
    backend: LLMBackend,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> PipelineResult:
    """Answer with single dense top-k retrieval only: no BM25, rerank, hops, or citation checks."""
    collection = load_chroma_collection(cache_dir)
    query_embedding = embedder.embed([question])[0]
    context = query_dense(collection, query_embedding, top_k=top_k)
    passages = "\n\n".join(retrieved.chunk.text for retrieved in context)
    prompt = f"Context:\n{passages}\n\nQuestion: {question}\nAnswer:"
    answer = backend.generate(prompt, max_tokens=GENERATION_MAX_TOKENS_DEFAULT)
    logger.info("Naive pipeline answered from %d dense-only chunk(s)", len(context))
    return PipelineResult(
        pipeline="naive",
        answer=answer,
        citations=[],
        hop_path=[],
        context_texts=[retrieved.chunk.text for retrieved in context],
    )
