import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s
import chromadb
import numpy as np

from src.config import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION, INDEX_SCHEMA_VERSION
from src.ingestion.schemas import Chunk

logger = logging.getLogger(__name__)

CHROMA_DIRNAME = "chroma"
BM25_DIRNAME = "bm25"
COLLECTION_NAME = "chunks"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class IndexManifest:
    """Records what a built index at a cache_dir actually contains."""

    repo_slug: str
    repo_sha: str
    schema_version: int
    embedding_model_id: str
    embedding_model_revision: str
    chunk_count: int
    embedding_dimension: int

    def is_compatible_with(self, repo_slug: str, repo_sha: str) -> bool:
        """Check whether a cached index still matches the given repo/commit and embedding setup."""
        return (
            self.repo_slug == repo_slug
            and self.repo_sha == repo_sha
            and self.schema_version == INDEX_SCHEMA_VERSION
            and self.embedding_model_id == EMBEDDING_MODEL_ID
            and self.embedding_model_revision == EMBEDDING_MODEL_REVISION
        )


def load_manifest(cache_dir: Path) -> IndexManifest | None:
    """Load a previously written index manifest, or None if this cache was never built."""
    manifest_path = cache_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    return IndexManifest(**json.loads(manifest_path.read_text()))


def _write_manifest(manifest: IndexManifest, path: Path) -> None:
    """Persist an IndexManifest as JSON."""
    path.write_text(json.dumps(asdict(manifest), indent=2))


def build_index(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    cache_dir: Path,
    *,
    repo_slug: str,
    repo_sha: str,
) -> IndexManifest | None:
    """Atomically rebuild the Chroma + bm25s indexes; returns None if there were no chunks."""
    if not chunks:
        logger.warning("No chunks to index for %s; skipping index build", cache_dir)
        return None

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(dir=cache_dir.parent, prefix=f".{cache_dir.name}.tmp-"))
    started = time.monotonic()
    try:
        _build_chroma_collection(chunks, embeddings, staging_dir / CHROMA_DIRNAME)
        _build_bm25_index(chunks, staging_dir / BM25_DIRNAME)
        manifest = IndexManifest(
            repo_slug=repo_slug,
            repo_sha=repo_sha,
            schema_version=INDEX_SCHEMA_VERSION,
            embedding_model_id=EMBEDDING_MODEL_ID,
            embedding_model_revision=EMBEDDING_MODEL_REVISION,
            chunk_count=len(chunks),
            embedding_dimension=int(embeddings.shape[1]),
        )
        _write_manifest(manifest, staging_dir / MANIFEST_FILENAME)
        _swap_in_place(staging_dir, cache_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    logger.info(
        "Indexed %d chunk(s) into %s in %.1fs",
        len(chunks),
        cache_dir,
        time.monotonic() - started,
    )
    return manifest


def _swap_in_place(staging_dir: Path, cache_dir: Path) -> None:
    """Replace cache_dir's contents with staging_dir's; on failure, cache_dir is left untouched."""
    previous_dir = None
    if cache_dir.exists():
        previous_dir = cache_dir.parent / f".{cache_dir.name}.previous-{os.getpid()}"
        cache_dir.rename(previous_dir)
    try:
        staging_dir.rename(cache_dir)
    except BaseException:
        if previous_dir is not None:
            previous_dir.rename(cache_dir)
        raise
    if previous_dir is not None:
        shutil.rmtree(previous_dir, ignore_errors=True)


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Map a Chunk onto the flat metadata dict Chroma stores alongside its vector."""
    return {
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "symbol_name": chunk.symbol_name,
        "qualified_name": chunk.qualified_name,
        "symbol_type": chunk.symbol_type,
        "language": chunk.language,
    }


def _build_chroma_collection(
    chunks: list[Chunk], embeddings: np.ndarray, chroma_dir: Path
) -> None:
    """Create a fresh Chroma collection at chroma_dir populated with the given chunks/embeddings."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=[chunk.text for chunk in chunks],
        metadatas=[_chunk_metadata(chunk) for chunk in chunks],
    )
    logger.info("Built Chroma collection with %d chunk(s) at %s", len(chunks), chroma_dir)


def _build_bm25_index(chunks: list[Chunk], bm25_dir: Path) -> None:
    """Build and persist a bm25s sparse index over chunk texts, keyed by chunk_id."""
    corpus = [chunk.text for chunk in chunks]
    corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    bm25_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(bm25_dir), corpus=[chunk.chunk_id for chunk in chunks])
    logger.info("Built bm25 index with %d chunk(s) at %s", len(chunks), bm25_dir)


def load_chroma_collection(cache_dir: Path) -> chromadb.Collection:
    """Load the persisted Chroma collection for a previously indexed repo."""
    client = chromadb.PersistentClient(path=str(cache_dir / CHROMA_DIRNAME))
    return client.get_collection(COLLECTION_NAME)
