import logging

import gradio as gr
from gradio import themes

from scripts.build_index import ingest_repository
from src.baseline.schemas import PipelineResult
from src.config import CACHE_ROOT, DEFAULT_RETRIEVAL_TOP_K, DEMO_REPO_URL, MAX_INDEXABLE_CHUNKS
from src.generation.cpu_backend import CPUBackend
from src.generation.llm import GenerationBackend
from src.ingestion.clone import repo_slug_from_url
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import load_manifest
from src.pipeline import compare
from src.retrieval.rerank import Reranker
from src.ui.compare_view import build_compare_view
from src.ui.ingest_view import build_ingest_view

logger = logging.getLogger(__name__)

DEMO_REPO_SLUG = repo_slug_from_url(DEMO_REPO_URL)

_ModelBundle = tuple[Embedder, Reranker, GenerationBackend]
_models: _ModelBundle | None = None


def _load_models() -> _ModelBundle:
    """Return the (Embedder, Reranker, generation backend), loading them on first use.

    None of the models are repo-specific — every repo's index is queried with the same
    embedder, reranker and LLM — so one shared copy serves every indexed repo.
    """
    global _models
    if _models is None:
        _models = (Embedder(), Reranker(), CPUBackend())
    return _models


def _repo_context(repo_slug: str) -> tuple[str, str] | None:
    """Return (repo_slug, repo_sha) for repo_slug's index, or None if it hasn't been built yet."""
    manifest = load_manifest(CACHE_ROOT / repo_slug)
    return (repo_slug, manifest.repo_sha) if manifest is not None else None


def _compare(
    repo_slug: str, question: str
) -> tuple[PipelineResult, PipelineResult, tuple[str, str] | None]:
    """Compare naive vs. RAGnarok on repo_slug's index."""
    embedder, reranker, backend = _load_models()
    naive_result, ragnarok_result = compare(
        question,
        CACHE_ROOT / repo_slug,
        embedder,
        reranker,
        backend,
        top_k=DEFAULT_RETRIEVAL_TOP_K,
    )
    return naive_result, ragnarok_result, _repo_context(repo_slug)


def _ingest(repo_url: str) -> tuple[str, str]:
    """Ingest repo_url; returns (status_message, repo_slug).

    Failures are caught by ingest_view.py's render_ingestion, not here — mirrors _compare,
    which likewise leaves error handling entirely to compare_view.py's render_comparison.
    """
    result = ingest_repository(repo_url)
    status = f"Indexed {result.repo_slug}@{result.repo_sha[:7]} — {result.chunk_count:,} chunks."
    if result.stats.truncated:
        status += (
            f" **Warning: this repo exceeded the {MAX_INDEXABLE_CHUNKS:,}-chunk indexing cap "
            "— the index is truncated and does not cover the entire repo.**"
        )
    return status, result.repo_slug


with gr.Blocks(title="RAGnarok", theme=themes.Soft()) as demo:
    active_repo = gr.State(value=DEMO_REPO_SLUG)
    with gr.Tabs():
        with gr.TabItem("Compare"):
            build_compare_view(_compare, active_repo)
        with gr.TabItem("Ingest a repo"):
            build_ingest_view(_ingest, active_repo)


def main() -> None:
    """Load the models up front, then launch the Gradio app.

    Loading before launch means the first question doesn't pay the full
    Embedder+Reranker+LLM load latency inside its request.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _load_models()
    demo.launch()


if __name__ == "__main__":
    main()
