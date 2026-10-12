"""Deterministic retrieval benchmark over the hand-labeled Q&A set.

Scores each retrieval configuration by whether it surfaces the files a question's answer
lives in (`expected_files`), so no LLM judge is involved. Also reports the reranker's best
score per question, which is what calibrates MIN_CONTEXT_RELEVANCE_SCORE: in-repo questions
should clear the threshold, adversarial "not in this repo" questions shouldn't.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.eval.qa_dataset import QAItem, load_qa_dataset
from src.eval.retrieval import file_recall, hit_at_k, ranked_files, reciprocal_rank
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import BM25_DIRNAME, load_chroma_collection
from src.retrieval.dense import query_dense
from src.retrieval.hybrid import fuse_rrf
from src.retrieval.rerank import Reranker
from src.retrieval.schemas import RetrievedChunk
from src.retrieval.sparse import query_sparse

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_INVALID_ARGUMENT = 2

DEFAULT_QA_PATH = Path("data/qa_starlette.jsonl")

CONFIGS = ("dense (naive baseline)", "bm25", "hybrid (RRF)", "hybrid + rerank")


def retrieve_all_configs(
    question: str, cache_dir: Path, embedder: Embedder, reranker: Reranker, top_k: int
) -> tuple[dict[str, list[RetrievedChunk]], float]:
    """Run every configuration for one question; returns results and the best rerank score.

    The last configuration mirrors agent.tools.retrieve_hybrid, the pipeline's real path.
    """
    collection = load_chroma_collection(cache_dir)
    dense = query_dense(collection, embedder.embed_query(question), top_k=top_k)
    sparse = query_sparse(cache_dir / BM25_DIRNAME, collection, question, top_k=top_k)
    fused = fuse_rrf(dense, sparse, top_k=top_k)
    reranked = reranker.rerank(question, fused)
    results = dict(zip(CONFIGS, [dense, sparse, fused, reranked]))
    return results, reranked[0].score if reranked else float("-inf")


def summarize(per_item: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Average each configuration's metrics over items."""
    return {
        config: {
            metric: statistics.fmean(item[config][metric] for item in per_item)
            for metric in per_item[0][config]
        }
        for config in CONFIGS
    }


def run(
    qa_items: list[QAItem], cache_dir: Path, embedder: Embedder, reranker: Reranker, top_k: int
) -> dict[str, object]:
    """Benchmark every configuration over qa_items; returns a JSON-serializable report."""
    scored: dict[str, list[dict[str, dict[str, float]]]] = {"single_hop": [], "multi_hop": []}
    best_scores: dict[str, list[float]] = {"in_repo": [], "adversarial": []}
    started = time.monotonic()
    for item in qa_items:
        results, best_score = retrieve_all_configs(
            item.question, cache_dir, embedder, reranker, top_k
        )
        if item.category == "adversarial":
            best_scores["adversarial"].append(best_score)
            logger.info("%s done: best rerank score %.2f", item.id, best_score)
            continue
        best_scores["in_repo"].append(best_score)
        metrics = {}
        for config, retrieved in results.items():
            files = ranked_files(retrieved)
            metrics[config] = {
                "hit@1": hit_at_k(files, item.expected_files, 1),
                "hit@3": hit_at_k(files, item.expected_files, 3),
                "mrr": reciprocal_rank(files, item.expected_files),
                f"file_recall@{top_k}": file_recall(files, item.expected_files),
            }
        scored[item.category].append(metrics)
        logger.info(
            "%s done: correct file ranked first by %s",
            item.id,
            [config for config in CONFIGS if metrics[config]["hit@1"]] or "none",
        )

    return {
        "top_k": top_k,
        "seconds_per_question": (time.monotonic() - started) / len(qa_items),
        "single_hop": summarize(scored["single_hop"]),
        "multi_hop": summarize(scored["multi_hop"]),
        "best_rerank_score": best_scores,
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the eval_retrieval CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval configurations against the hand-labeled Q&A set "
        "(run scripts/build_index.py first)."
    )
    parser.add_argument("cache_dir", type=Path, help="Index directory produced by build_index.py.")
    parser.add_argument(
        "--qa-path", type=Path, default=DEFAULT_QA_PATH, help="Hand-labeled Q&A JSONL file."
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help="Chunks kept per config."
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], help="Force the embedding/rerank device."
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

    embedder, reranker = Embedder(device=args.device), Reranker(device=args.device)
    report = run(load_qa_dataset(args.qa_path), args.cache_dir, embedder, reranker, args.top_k)
    print(json.dumps(report, indent=2))
    return EXIT_SUCCESS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
