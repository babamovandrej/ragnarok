"""Answer a few labeled questions with both pipelines and save the answers side by side.

A spot check that the whole chain works with real models: retrieval, the hop decisions, the
generated answer and its citations. Each question is written to the output file as soon as
it is answered, so an interrupted run keeps what it finished.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.eval.qa_dataset import QAItem, load_qa_dataset
from src.generation.cpu_backend import CPUBackend
from src.generation.llm import GenerationBackend
from src.ingestion.embedder import Embedder
from src.pipeline import compare
from src.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_INVALID_ARGUMENT = 2

DEFAULT_QA_PATH = Path("data/qa_starlette.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/sample_answers.json")
DEFAULT_IDS = ("single_hop_10", "single_hop_04", "multi_hop_03", "multi_hop_08", "adversarial_02")


def answer_item(
    item: QAItem,
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    backend: GenerationBackend,
) -> dict[str, object]:
    """Run one labeled question through both pipelines; returns a JSON-serializable record."""
    started = time.monotonic()
    naive, ragnarok = compare(item.question, cache_dir, embedder, reranker, backend)
    return {
        "id": item.id,
        "category": item.category,
        "question": item.question,
        "expected_files": item.expected_files,
        "grounding_notes": item.grounding_notes,
        "seconds": round(time.monotonic() - started, 1),
        "naive_answer": naive.answer,
        "ragnarok_answer": ragnarok.answer,
        "citations": [
            {
                "span": f"{citation.file_path}#L{citation.start_line}-L{citation.end_line}",
                "verified": citation.verified,
            }
            for citation in ragnarok.citations
        ],
        "hops": [
            {"symbol": hop.symbol_name, "file": hop.file_path, "reason": hop.reason}
            for hop in ragnarok.hop_path
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the sample_answers CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Answer a few labeled questions with the naive baseline and RAGnarok "
        "(run scripts/build_index.py first)."
    )
    parser.add_argument("cache_dir", type=Path, help="Index directory produced by build_index.py.")
    parser.add_argument(
        "--qa-path", type=Path, default=DEFAULT_QA_PATH, help="Hand-labeled Q&A JSONL file."
    )
    parser.add_argument(
        "--ids", nargs="+", default=list(DEFAULT_IDS), help="Question ids to answer, in order."
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to write the answers."
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
    items_by_id = {item.id: item for item in load_qa_dataset(args.qa_path)}
    unknown = [question_id for question_id in args.ids if question_id not in items_by_id]
    if unknown:
        logger.error("Unknown question id(s): %s", ", ".join(unknown))
        return EXIT_INVALID_ARGUMENT

    embedder, reranker = Embedder(device=args.device), Reranker(device=args.device)
    backend = CPUBackend()
    records: list[dict[str, object]] = []
    for question_id in args.ids:
        record = answer_item(items_by_id[question_id], args.cache_dir, embedder, reranker, backend)
        records.append(record)
        args.out.write_text(json.dumps(records, indent=2) + "\n")
        logger.info("%s answered in %.0fs", question_id, record["seconds"])
    return EXIT_SUCCESS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    sys.exit(main())
