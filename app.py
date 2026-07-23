import logging

import gradio as gr

from scripts.build_index import ingest_repository
from src.baseline.schemas import PipelineResult
from src.config import CACHE_ROOT, DEFAULT_RETRIEVAL_TOP_K, DEMO_REPO_URL, MAX_RESIDENT_REPOS
from src.generation.cpu_backend import CPUBackend
from src.ingestion.clone import repo_slug_from_url
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import load_manifest
from src.pipeline import compare
from src.retrieval.rerank import Reranker
from src.ui.compare_view import build_compare_view
from src.ui.ingest_view import build_ingest_view
from src.utils.caching import LRUCache

logger = logging.getLogger(__name__)

DEMO_REPO_SLUG = repo_slug_from_url(DEMO_REPO_URL)
DEMO_CACHE_DIR = CACHE_ROOT / DEMO_REPO_SLUG

_ModelBundle = tuple[Embedder, Reranker, CPUBackend]
_model_cache: LRUCache[_ModelBundle] = LRUCache(capacity=MAX_RESIDENT_REPOS)


def _load_models(cache_key: str) -> _ModelBundle:
    """Return this repo's (Embedder, Reranker, CPUBackend), loading and caching them once."""
    cached = _model_cache.get(cache_key)
    if cached is not None:
        return cached
    models = (Embedder(), Reranker(), CPUBackend())
    _model_cache.put(cache_key, models)
    return models


def _demo_compare(question: str) -> tuple[PipelineResult, PipelineResult]:
    """Compare naive vs. RAGnarok on the pre-warmed demo repo, reusing cached models."""
    embedder, reranker, backend = _load_models(DEMO_REPO_SLUG)
    return compare(
        question, DEMO_CACHE_DIR, embedder, reranker, backend, top_k=DEFAULT_RETRIEVAL_TOP_K
    )


def _demo_repo_context() -> tuple[str, str] | None:
    """Return (repo_slug, repo_sha) for the demo index, or None if it hasn't been built yet."""
    manifest = load_manifest(DEMO_CACHE_DIR)
    return (DEMO_REPO_SLUG, manifest.repo_sha) if manifest is not None else None


def _ingest(repo_url: str) -> str:
    """Ingest repo_url via scripts.build_index.ingest_repository; returns a status message."""
    try:
        result = ingest_repository(repo_url)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", repo_url)
        return f"Ingestion failed: {exc}"
    return f"Indexed {result.repo_slug}@{result.repo_sha[:7]} — {result.chunk_count:,} chunks."


with gr.Blocks(title="RAGnarok", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.TabItem("Compare"):
            build_compare_view(  # type: ignore[no-untyped-call]  # gradio ships no stub for Blocks.render
                _demo_compare, repo_context=_demo_repo_context()
            ).render()
        with gr.TabItem("Ingest a repo"):
            build_ingest_view(_ingest).render()  # type: ignore[no-untyped-call]  # gradio ships no stub for Blocks.render

if __name__ == "__main__":
    demo.launch()
