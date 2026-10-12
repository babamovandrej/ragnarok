import logging
from pathlib import Path

from src.baseline.schemas import PipelineResult
from src.config import DEFAULT_RETRIEVAL_TOP_K, GENERATION_MAX_TOKENS_DEFAULT
from src.generation.llm import GenerationBackend
from src.generation.prompts import BLOCK_SEPARATOR, select_blocks_that_fit
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import load_chroma_collection
from src.retrieval.dense import query_dense

logger = logging.getLogger(__name__)

_PROMPT = "Context:\n{passages}\n\nQuestion: {question}\nAnswer:"


def answer_naive(
    question: str,
    cache_dir: Path,
    embedder: Embedder,
    backend: GenerationBackend,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> PipelineResult:
    """Answer with single dense top-k retrieval only: no BM25, rerank, hops, or citation checks.

    Context is trimmed to the same token budget RAGnarok gets, so an overlong prompt can't
    crash generation and neither pipeline sees more context than the other.
    """
    collection = load_chroma_collection(cache_dir)
    query_embedding = embedder.embed_query(question)
    retrieved = query_dense(collection, query_embedding, top_k=top_k)
    kept = select_blocks_that_fit(
        _PROMPT.format(passages="", question=question),
        [r.chunk.text for r in retrieved],
        backend,
        GENERATION_MAX_TOKENS_DEFAULT,
    )
    context = [retrieved[index] for index in kept]
    passages = BLOCK_SEPARATOR.join(r.chunk.text for r in context)
    prompt = _PROMPT.format(passages=passages, question=question)
    answer = backend.generate(prompt, max_tokens=GENERATION_MAX_TOKENS_DEFAULT)
    logger.info("Naive pipeline answered from %d dense-only chunk(s)", len(context))
    return PipelineResult(
        pipeline="naive",
        answer=answer,
        citations=[],
        hop_path=[],
        context_texts=[retrieved.chunk.text for retrieved in context],
    )
