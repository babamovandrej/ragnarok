import argparse
import logging
import sys
from functools import partial
from pathlib import Path

from src.agent.graph import build_graph, llm_decide_more_hops
from src.agent.state import AgentState, HopRecord
from src.agent.tools import find_symbol_definition, retrieve_hybrid
from src.config import DEFAULT_RETRIEVAL_TOP_K, GENERATION_MAX_TOKENS_DEFAULT
from src.generation.citations import Citation, verify_citations
from src.generation.cpu_backend import CPUBackend
from src.generation.llm import GenerationBackend
from src.generation.prompts import fit_chunks_to_budget, render_prompt
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import BM25_DIRNAME, load_chroma_collection
from src.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_INVALID_ARGUMENT = 2


def answer_question(
    question: str,
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    backend: GenerationBackend,
    top_k: int,
    max_tokens: int,
) -> tuple[str, list[Citation]]:
    """Retrieve context for question, generate an answer, and verify its citations."""
    collection = load_chroma_collection(cache_dir)
    context = retrieve_hybrid(
        question, collection, cache_dir / BM25_DIRNAME, embedder, reranker, top_k=top_k
    )

    fitted_context = fit_chunks_to_budget(question, context, backend, max_tokens)
    prompt = render_prompt(question, fitted_context)
    answer = backend.generate(prompt, max_tokens=max_tokens)
    citations = verify_citations(answer, [retrieved.chunk for retrieved in context])
    return answer, citations


def answer_question_with_agent(
    question: str,
    cache_dir: Path,
    embedder: Embedder,
    reranker: Reranker,
    backend: GenerationBackend,
    top_k: int,
) -> tuple[str, list[Citation], list[HopRecord]]:
    """Answer question via the multi-hop agent graph; returns the answer, citations, hop path."""
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
        decide_more_hops=partial(llm_decide_more_hops, backend),
        backend=backend,
    )
    result = graph.invoke(AgentState(question=question))
    citations = verify_citations(result["answer"], [c.chunk for c in result["context"]])
    return result["answer"], citations, result.get("hop_path", [])


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
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the multi-hop LangGraph agent instead of single-hop retrieval.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], help="Force the embedding device."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.cache_dir.is_dir():
        logger.error("No index found at %s; run scripts/build_index.py first.", args.cache_dir)
        return EXIT_INVALID_ARGUMENT

    embedder, reranker, backend = Embedder(device=args.device), Reranker(), CPUBackend()
    hop_path: list[HopRecord] = []
    if args.agent:
        answer, citations, hop_path = answer_question_with_agent(
            args.question, args.cache_dir, embedder, reranker, backend, args.top_k
        )
    else:
        answer, citations = answer_question(
            args.question, args.cache_dir, embedder, reranker, backend, args.top_k, args.max_tokens
        )

    if hop_path:
        print("Hop path:")
        for i, hop in enumerate(hop_path, start=1):
            print(f"  {i}. {hop.symbol_name} ({hop.file_path}#L{hop.start_line}-L{hop.end_line})")
            print(f"     {hop.reason}")

    print(f"\nAnswer:\n{answer}\n")
    print("Citations:")
    for citation in citations:
        status = "verified" if citation.verified else "UNVERIFIED"
        print(f"  {citation.file_path}#L{citation.start_line}-L{citation.end_line} [{status}]")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
