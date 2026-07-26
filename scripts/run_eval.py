import argparse
import logging
import sys
from pathlib import Path

from ragas.dataset_schema import EvaluationResult

from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.eval.ablation import run_ablation
from src.eval.metrics import idk_rate
from src.eval.qa_dataset import QAItem, load_qa_dataset
from src.eval.ragas_runner import run_ragas_eval
from src.generation.cpu_backend import CPUBackend
from src.generation.llm import GenerationBackend
from src.ingestion.embedder import Embedder
from src.pipeline import answer_ragnarok
from src.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_INVALID_ARGUMENT = 2

DEFAULT_QA_PATH = Path("data/eval/qa_starlette.jsonl")


def run_pipeline_eval(
    qa_items: list[QAItem],
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    backend: GenerationBackend,
    top_k: int,
) -> tuple[EvaluationResult, float]:
    """Answer every QA item with the full RAGnarok pipeline; score with RAGAS + adversarial IDK."""
    results = [
        answer_ragnarok(item.question, cache_dir, embedder, reranker, backend, top_k=top_k)
        for item in qa_items
    ]
    scored = run_ragas_eval(
        qa_items, [r.answer for r in results], [r.context_texts for r in results], backend
    )
    adversarial_answers = [
        r.answer for item, r in zip(qa_items, results) if item.category == "adversarial"
    ]
    return scored, idk_rate(adversarial_answers)


def build_parser() -> argparse.ArgumentParser:
    """Construct the run_eval CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Score RAGnarok against the hand-labeled starlette Q&A set "
        "(run scripts/build_index.py first)."
    )
    parser.add_argument("cache_dir", type=Path, help="Index directory produced by build_index.py.")
    parser.add_argument(
        "--qa-path", type=Path, default=DEFAULT_QA_PATH, help="Hand-labeled Q&A JSONL file."
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help="Context chunks after rerank."
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the naive/hybrid+rerank/multi-hop ablation table instead of a single score.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.cache_dir.is_dir():
        logger.error("No index found at %s; run scripts/build_index.py first.", args.cache_dir)
        return EXIT_INVALID_ARGUMENT
    if not args.qa_path.is_file():
        logger.error("No Q&A dataset found at %s.", args.qa_path)
        return EXIT_INVALID_ARGUMENT

    qa_items = load_qa_dataset(args.qa_path)
    embedder, reranker, backend = Embedder(), Reranker(), CPUBackend()

    if args.ablation:
        rows = run_ablation(qa_items, args.cache_dir, embedder, reranker, backend, top_k=args.top_k)
        print(f"{'pipeline':<15}{'faithfulness':>14}{'ctx_precision':>15}{'ctx_recall':>12}")
        for row in rows:
            print(
                f"{row.pipeline:<15}{row.faithfulness:>14.3f}"
                f"{row.context_precision:>15.3f}{row.context_recall:>12.3f}"
            )
        return EXIT_SUCCESS

    scored, adversarial_idk_rate = run_pipeline_eval(
        qa_items, args.cache_dir, embedder, reranker, backend, args.top_k
    )
    table = scored.to_pandas()
    print(table.to_string(index=False))
    print(f"\nAggregate faithfulness:      {table['faithfulness'].mean():.3f}")
    precision_col = table["llm_context_precision_without_reference"].mean()
    print(f"Aggregate context precision: {precision_col:.3f}")
    print(f"Aggregate context recall:    {table['context_recall'].mean():.3f}")
    print(f"Adversarial IDK rate:        {adversarial_idk_rate:.0%}")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
