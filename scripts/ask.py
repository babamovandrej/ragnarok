import argparse
import logging
import sys
from pathlib import Path

from src.config import DEFAULT_RETRIEVAL_TOP_K, GENERATION_MAX_TOKENS_DEFAULT
from src.generation.citations import Citation, verify_citations
from src.generation.cpu_backend import CPUBackend
from src.generation.llm import LLMBackend
from src.generation.prompts import render_prompt
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import BM25_DIRNAME, load_chroma_collection
from src.retrieval.dense import query_dense
from src.retrieval.hybrid import fuse_rrf
from src.retrieval.rerank import Reranker
from src.retrieval.sparse import query_sparse

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_INVALID_ARGUMENT = 2


def answer_question(
    question: str,
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    backend: LLMBackend,
    top_k: int,
    max_tokens: int,
) -> tuple[str, list[Citation]]:
    """Retrieve context for question, generate an answer, and verify its citations."""
    collection = load_chroma_collection(cache_dir)
    query_embedding = embedder.embed([question])[0]

    dense = query_dense(collection, query_embedding, top_k=top_k)
    sparse = query_sparse(cache_dir / BM25_DIRNAME, collection, question, top_k=top_k)
    fused = fuse_rrf(dense, sparse, top_k=top_k)
    context = reranker.rerank(question, fused)

    prompt = render_prompt(question, context)
    answer = backend.generate(prompt, max_tokens=max_tokens)
    citations = verify_citations(answer, [retrieved.chunk for retrieved in context])
    return answer, citations


def build_parser() -> argparse.ArgumentParser:
    """Construct the ask CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Ask a single-hop question against an already-built index "
        "(run scripts/build_index.py first)."
    )
    parser.add_argument("cache_dir", type=Path, help="Index directory produced by build_index.py.")
    parser.add_argument("question", help="A natural-language question about the indexed repo.")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help="Context chunks after rerank."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=GENERATION_MAX_TOKENS_DEFAULT,
        help="Max answer tokens.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.cache_dir.is_dir():
        logger.error("No index found at %s; run scripts/build_index.py first.", args.cache_dir)
        return EXIT_INVALID_ARGUMENT

    answer, citations = answer_question(
        args.question,
        args.cache_dir,
        Embedder(),
        Reranker(),
        CPUBackend(),
        args.top_k,
        args.max_tokens,
    )

    print(f"\nAnswer:\n{answer}\n")
    print("Citations:")
    for citation in citations:
        status = "verified" if citation.verified else "UNVERIFIED"
        print(f"  {citation.file_path}#L{citation.start_line}-L{citation.end_line} [{status}]")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
