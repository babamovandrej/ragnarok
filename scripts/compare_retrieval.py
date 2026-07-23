import argparse
import json
import logging
import sys
from pathlib import Path

from src.agent.tools import retrieve_hybrid
from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import BM25_DIRNAME, load_chroma_collection
from src.retrieval.dense import query_dense
from src.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_INVALID_ARGUMENT = 2


def compare_query(
    query: str,
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    top_k: int,
) -> dict[str, list[str]]:
    """Run one query through the dense-only and hybrid+rerank paths against a built index."""
    collection = load_chroma_collection(cache_dir)
    query_embedding = embedder.embed([query])[0]
    dense_only = query_dense(collection, query_embedding, top_k=top_k)
    hybrid_reranked = retrieve_hybrid(
        query, collection, cache_dir / BM25_DIRNAME, embedder, reranker, top_k=top_k
    )

    return {
        "dense_only": [retrieved.chunk.qualified_name for retrieved in dense_only],
        "hybrid_reranked": [retrieved.chunk.qualified_name for retrieved in hybrid_reranked],
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the compare_retrieval CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare dense-only vs hybrid+rerank retrieval on an already-built index "
        "(run scripts/build_index.py first)."
    )
    parser.add_argument("cache_dir", type=Path, help="Index directory produced by build_index.py.")
    parser.add_argument("queries", nargs="+", help="One or more natural-language queries.")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help="Candidates per path."
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.cache_dir.is_dir():
        logger.error("No index found at %s; run scripts/build_index.py first.", args.cache_dir)
        return EXIT_INVALID_ARGUMENT

    embedder = Embedder()
    reranker = Reranker()
    results = {
        query: compare_query(query, args.cache_dir, embedder, reranker, args.top_k)
        for query in args.queries
    }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for query, paths in results.items():
            print(f"\nQuery: {query}")
            print(f"  dense-only:      {paths['dense_only']}")
            print(f"  hybrid+reranked: {paths['hybrid_reranked']}")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
