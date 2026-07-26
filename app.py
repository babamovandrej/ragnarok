import logging
import os

import gradio as gr
from gradio import themes

from scripts.build_index import ingest_repository
from src.baseline.schemas import PipelineResult
from src.config import (
    CACHE_ROOT,
    DEFAULT_RETRIEVAL_TOP_K,
    DEMO_REPO_URL,
    MAX_INDEXABLE_CHUNKS,
    MAX_RESIDENT_REPOS,
)
from src.generation.cpu_backend import CPUBackend
from src.generation.fallback_backend import FallbackBackend
from src.generation.llm import GenerationBackend
from src.generation.zerogpu_backend import ZeroGPUBackend
from src.ingestion.clone import repo_slug_from_url
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import load_manifest
from src.pipeline import compare
from src.retrieval.rerank import Reranker
from src.ui.compare_view import build_compare_view
from src.ui.ingest_view import build_ingest_view
from src.utils.caching import LRUCache
from src.utils.hf_dataset_io import refresh_if_stale

logger = logging.getLogger(__name__)

DEMO_REPO_SLUG = repo_slug_from_url(DEMO_REPO_URL)

_ModelBundle = tuple[Embedder, Reranker, GenerationBackend]
_model_cache: LRUCache[_ModelBundle] = LRUCache(capacity=MAX_RESIDENT_REPOS)


def _build_generation_backend() -> GenerationBackend:
    """Build the ZeroGPU-primary/CPU-fallback generation backend.

    ZeroGPUBackend's constructor moves the model onto CUDA, which only succeeds on
    actual ZeroGPU Space hardware. Outside of that (local dev, CI), fall back to a
    CPU-only backend so the app still runs — there's nothing to fall back *to* at
    generate-time if the primary backend never finished constructing.
    """
    cpu = CPUBackend()
    try:
        zerogpu = ZeroGPUBackend()
    except Exception:
        logger.warning(
            "ZeroGPU backend unavailable in this environment; using CPU only.", exc_info=True
        )
        return cpu
    return FallbackBackend(primary=zerogpu, fallback=cpu)


def _load_models(cache_key: str) -> _ModelBundle:
    """Return this repo's (Embedder, Reranker, generation backend), loading/caching them once."""
    cached = _model_cache.get(cache_key)
    if cached is not None:
        return cached
    models = (Embedder(), Reranker(), _build_generation_backend())
    _model_cache.put(cache_key, models)
    return models


def _repo_context(repo_slug: str) -> tuple[str, str] | None:
    """Return (repo_slug, repo_sha) for repo_slug's index, or None if it hasn't been built yet."""
    manifest = load_manifest(CACHE_ROOT / repo_slug)
    return (repo_slug, manifest.repo_sha) if manifest is not None else None


def _compare(
    repo_slug: str, question: str
) -> tuple[PipelineResult, PipelineResult, str | None, tuple[str, str] | None]:
    """Compare naive vs. RAGnarok on repo_slug's index, reusing that repo's cached models."""
    embedder, reranker, backend = _load_models(repo_slug)
    naive_result, ragnarok_result = compare(
        question,
        CACHE_ROOT / repo_slug,
        embedder,
        reranker,
        backend,
        top_k=DEFAULT_RETRIEVAL_TOP_K,
    )
    return (
        naive_result,
        ragnarok_result,
        getattr(backend, "last_backend_used", None),
        _repo_context(repo_slug),
    )


def _ingest(repo_url: str) -> tuple[str, str]:
    """Ingest repo_url; returns (status_message, repo_slug).

    Also warms repo_slug's query-time model cache entry (Embedder/Reranker/generation
    backend) so the Compare tab's first question against it doesn't pay full model-load
    latency, and so MAX_RESIDENT_REPOS actually gets exercised with more than one repo —
    previously only the demo repo's key was ever populated.

    Failures are caught by ingest_view.py's render_ingestion, not here — mirrors _compare,
    which likewise leaves error handling entirely to compare_view.py's render_comparison.
    """
    result = ingest_repository(repo_url)
    _load_models(result.repo_slug)
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


def _refresh_demo_repo_if_configured() -> None:
    """Pull the demo repo's latest CI-pushed index from the HF Dataset repo, if configured.

    HF_DATASET_REPO may not be set at all — the Dataset repo/Space secrets this depends on
    may not be provisioned yet — so this is a no-op, not a startup failure, when it's absent.
    Any failure while actually refreshing
    (network, missing dataset, etc.) is likewise caught: this Space must still serve whatever
    index it already has on disk rather than fail to start over a refresh it attempted only
    as a courtesy.
    """
    dataset_repo = os.environ.get("HF_DATASET_REPO")
    if not dataset_repo:
        logger.info("HF_DATASET_REPO not set; skipping demo repo index refresh at startup")
        return
    try:
        refreshed = refresh_if_stale(dataset_repo, CACHE_ROOT / DEMO_REPO_SLUG, DEMO_REPO_SLUG)
    except Exception:
        logger.warning(
            "Failed to refresh demo repo index from dataset repo %s", dataset_repo, exc_info=True
        )
        return
    if refreshed:
        logger.info("Refreshed demo repo index from dataset repo %s", dataset_repo)


def main() -> None:
    """Refresh + pre-warm the demo repo's models, then launch the Gradio app.

    Without pre-warming, the first visitor after every cold start (Spaces sleep after
    inactivity) would pay the full Embedder+Reranker+generation-backend load latency inside
    their request.
    """
    _refresh_demo_repo_if_configured()
    _load_models(DEMO_REPO_SLUG)
    demo.launch()


if __name__ == "__main__":
    main()
