import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.baseline.naive_rag import answer_naive
from src.baseline.schemas import PipelineResult
from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.eval.qa_dataset import QAItem
from src.eval.ragas_runner import run_ragas_eval
from src.generation.llm import GenerationBackend
from src.ingestion.embedder import Embedder
from src.pipeline import answer_ragnarok
from src.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

AblationPipeline = Literal["naive", "hybrid_rerank", "multi_hop"]


@dataclass(frozen=True)
class AblationRow:
    """One pipeline's aggregate RAGAS scores for the naive/hybrid+rerank/multi-hop table."""

    pipeline: AblationPipeline
    faithfulness: float
    context_precision: float
    context_recall: float


def _run_pipeline(
    pipeline: AblationPipeline,
    question: str,
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    backend: GenerationBackend,
    top_k: int,
) -> PipelineResult:
    """Answer one question through the requested ablation arm."""
    if pipeline == "naive":
        return answer_naive(question, cache_dir, embedder, backend, top_k=top_k)
    if pipeline == "hybrid_rerank":
        return answer_ragnarok(
            question, cache_dir, embedder, reranker, backend, top_k=top_k, max_hops=0
        )
    return answer_ragnarok(question, cache_dir, embedder, reranker, backend, top_k=top_k)


def run_ablation(
    qa_items: list[QAItem],
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    backend: GenerationBackend,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[AblationRow]:
    """Run the same Q&A set through naive / hybrid+rerank (0 hops) / full multi-hop pipelines."""
    rows = []
    for pipeline in ("naive", "hybrid_rerank", "multi_hop"):
        results = [
            _run_pipeline(pipeline, item.question, cache_dir, embedder, reranker, backend, top_k)
            for item in qa_items
        ]
        scored = run_ragas_eval(
            qa_items, [r.answer for r in results], [r.context_texts for r in results], backend
        )
        table = scored.to_pandas()
        rows.append(
            AblationRow(
                pipeline=pipeline,
                faithfulness=float(table["faithfulness"].mean()),
                context_precision=float(table["llm_context_precision_without_reference"].mean()),
                context_recall=float(table["context_recall"].mean()),
            )
        )
        logger.info("Ablation arm %s scored on %d item(s)", pipeline, len(qa_items))
    return rows
