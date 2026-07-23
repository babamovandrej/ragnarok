import gradio as gr

from scripts.build_index import ingest_repository
from src.baseline.naive_rag import compare
from src.config import CACHE_ROOT, DEFAULT_RETRIEVAL_TOP_K, DEMO_REPO_URL, MAX_RESIDENT_REPOS
from src.generation.cpu_backend import CPUBackend
from src.ingestion.clone import repo_slug_from_url
from src.ingestion.embedder import Embedder
from src.retrieval.rerank import Reranker
from src.ui.compare_view import build_compare_view
from src.ui.ingest_view import build_ingest_view
from src.utils.caching import LRUCache

DEMO_REPO_SLUG = repo_slug_from_url(DEMO_REPO_URL)
DEMO_CACHE_DIR = CACHE_ROOT / DEMO_REPO_SLUG

_model_cache: LRUCache = LRUCache(capacity=MAX_RESIDENT_REPOS)


def _load_models(cache_key: str) -> tuple[Embedder, Reranker, CPUBackend]:
    """Return this repo's (Embedder, Reranker, CPUBackend), loading and caching them once."""
    cached = _model_cache.get(cache_key)
    if cached is not None:
        return cached
    models = (Embedder(), Reranker(), CPUBackend())
    _model_cache.put(cache_key, models)
    return models


def _demo_compare(question: str):
    """Compare naive vs. RAGnarok on the pre-warmed demo repo, reusing cached models."""
    embedder, reranker, backend = _load_models(DEMO_REPO_SLUG)
    return compare(
        question, DEMO_CACHE_DIR, embedder, reranker, backend, top_k=DEFAULT_RETRIEVAL_TOP_K
    )


def _ingest(repo_url: str) -> str:
    """Ingest repo_url via scripts.build_index.ingest_repository; returns a status message."""
    try:
        result = ingest_repository(repo_url)
    except Exception as exc:
        return f"Ingestion failed: {exc}"
    return f"Indexed {result.repo_slug}@{result.repo_sha[:7]} — {result.chunk_count:,} chunks."


with gr.Blocks(title="RAGnarok") as demo:
    with gr.Tabs():
        with gr.TabItem("Compare"):
            build_compare_view(_demo_compare).render()
        with gr.TabItem("Ingest a repo"):
            build_ingest_view(_ingest).render()

if __name__ == "__main__":
    demo.launch()
