import logging
from functools import partial
from pathlib import Path

from src.agent.graph import build_graph, decide_next_hop
from src.agent.state import AgentState
from src.agent.tools import RelevantReferences, find_symbol_definition, retrieve_hybrid
from src.baseline.naive_rag import answer_naive
from src.baseline.schemas import PipelineResult
from src.config import AGENT_MAX_HOPS, DEFAULT_RETRIEVAL_TOP_K
from src.generation.citations import verify_citations
from src.generation.llm import GenerationBackend
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import BM25_DIRNAME, load_chroma_collection
from src.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)


def answer_ragnarok(
    question: str,
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    backend: GenerationBackend,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    max_hops: int = AGENT_MAX_HOPS,
) -> PipelineResult:
    """Answer with hybrid+rerank plus up to max_hops agent hops; citations verified."""
    collection = load_chroma_collection(cache_dir)
    bm25_dir = cache_dir / BM25_DIRNAME
    graph = build_graph(
        retrieve=partial(
            retrieve_hybrid,
            collection=collection,
            bm25_dir=bm25_dir,
            embedder=embedder,
            reranker=reranker,
            top_k=top_k,
        ),
        find_symbol=partial(find_symbol_definition, collection=collection),
        decide_more_hops=partial(
            decide_next_hop, RelevantReferences(question, collection, reranker)
        ),
        backend=backend,
    )
    try:
        result = graph.invoke(AgentState(question=question, hops_remaining=max_hops))
    except Exception:
        logger.exception("RAGnarok pipeline failed to generate an answer for %r", question)
        raise
    context = [*result["context"], *result.get("hop_context", [])]
    citations = verify_citations(result["answer"], [c.chunk for c in context])
    hop_path = result.get("hop_path", [])
    logger.info("RAGnarok pipeline answered with %d hop(s)", len(hop_path))
    return PipelineResult(
        pipeline="ragnarok",
        answer=result["answer"],
        citations=citations,
        hop_path=hop_path,
        context_texts=[c.chunk.text for c in context],
    )


def compare(
    question: str,
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    backend: GenerationBackend,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> tuple[PipelineResult, PipelineResult]:
    """Run question through both pipelines in parallel intent; returns (naive, ragnarok)."""
    naive_result = answer_naive(question, cache_dir, embedder, backend, top_k=top_k)
    ragnarok_result = answer_ragnarok(question, cache_dir, embedder, reranker, backend, top_k=top_k)
    return naive_result, ragnarok_result
