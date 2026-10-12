from pathlib import Path

import numpy as np
from chromadb.api.types import IncludeEnum

from src.ingestion.chunker import chunk_file, is_excluded_path
from src.ingestion.clone import changed_files_since
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import load_chroma_collection
from src.ingestion.schemas import Chunk, IngestionStats
from src.retrieval.schemas import chunk_from_chroma_record


def merge_incremental_chunks(
    cache_dir: Path, repo_dir: Path, since_sha: str, embedder: Embedder, new_sha: str
) -> tuple[list[Chunk], np.ndarray, IngestionStats]:
    """Re-chunk/re-embed only files changed since since_sha; reuse existing chunks/embeddings.

    Reused chunks keep whatever repo_sha they were originally embedded at (read back from
    Chroma metadata); only freshly re-chunked files are stamped with new_sha, so a chunk's
    repo_sha always reflects the commit its own content actually came from.
    """
    changed = changed_files_since(repo_dir, since_sha)
    changed_paths = {c.path for c in changed}
    changed_paths |= {c.previous_path for c in changed if c.previous_path}

    collection = load_chroma_collection(cache_dir)
    existing = collection.get(
        include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.embeddings]
    )
    documents, metadatas = existing["documents"], existing["metadatas"]
    assert documents is not None and metadatas is not None and existing["embeddings"] is not None
    kept_chunks: list[Chunk] = []
    kept_embeddings: list[list[float]] = []
    for chunk_id, text, metadata, embedding in zip(
        existing["ids"], documents, metadatas, existing["embeddings"]
    ):
        if metadata["file_path"] in changed_paths:
            continue
        kept_chunks.append(chunk_from_chroma_record(chunk_id, text, metadata))
        kept_embeddings.append(list(embedding))

    new_chunks: list[Chunk] = []
    for changed_file in changed:
        if changed_file.status.startswith("D") or is_excluded_path(changed_file.path):
            continue
        new_chunks.extend(chunk_file(repo_dir / changed_file.path, repo_dir, new_sha))

    new_embeddings = (
        embedder.embed([chunk.search_text for chunk in new_chunks]) if new_chunks else None
    )

    all_chunks = kept_chunks + new_chunks
    if kept_embeddings and new_embeddings is not None:
        all_embeddings = np.vstack([np.array(kept_embeddings), new_embeddings])
    elif new_embeddings is not None:
        all_embeddings = new_embeddings
    else:
        all_embeddings = np.array(kept_embeddings)

    stats = IngestionStats(
        files_scanned=len(changed),
        files_chunked=len({cf.path for cf in changed if not cf.status.startswith("D")}),
        chunks_created=len(new_chunks),
    )
    return all_chunks, all_embeddings, stats
