from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from src.ingestion.indexer import IndexManifest, build_index, load_manifest
from src.ingestion.schemas import Chunk


def export_chunks_to_parquet(
    chunks: list[Chunk], embeddings: np.ndarray, output_path: Path
) -> None:
    table = pa.table(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "file_path": [c.file_path for c in chunks],
            "start_line": [c.start_line for c in chunks],
            "end_line": [c.end_line for c in chunks],
            "symbol_name": [c.symbol_name for c in chunks],
            "qualified_name": [c.qualified_name for c in chunks],
            "symbol_type": [c.symbol_type for c in chunks],
            "language": [c.language for c in chunks],
            "text": [c.text for c in chunks],
            "embedding": [embedding.tolist() for embedding in embeddings],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)


def push_to_hub(parquet_path: Path, repo_id: str) -> None:
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=parquet_path.name,
        repo_id=repo_id,
        repo_type="dataset",
    )


def latest_commit_sha(repo_id: str) -> str:
    """Return the current commit SHA of a Hugging Face Dataset repo."""
    return HfApi().dataset_info(repo_id).sha


def pull_parquet(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download one parquet file from a Hugging Face Dataset repo into local_dir."""
    downloaded = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=str(local_dir)
    )
    return Path(downloaded)


def rehydrate_index_from_parquet(
    parquet_path: Path, cache_dir: Path, repo_slug: str, repo_sha: str
) -> IndexManifest | None:
    """Rebuild a local Chroma+bm25 index from our parquet schema, never Chroma's own format."""
    rows = pq.read_table(parquet_path).to_pylist()
    chunks = [
        Chunk(
            chunk_id=row["chunk_id"],
            file_path=row["file_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            symbol_name=row["symbol_name"],
            qualified_name=row["qualified_name"],
            symbol_type=row["symbol_type"],
            language=row["language"],
            text=row["text"],
        )
        for row in rows
    ]
    embeddings = np.array([row["embedding"] for row in rows], dtype=np.float32)
    return build_index(chunks, embeddings, cache_dir, repo_slug=repo_slug, repo_sha=repo_sha)


def refresh_if_stale(repo_id: str, cache_dir: Path, repo_slug: str) -> bool:
    """Pull and rehydrate from the Dataset repo if its latest SHA differs from the local cache."""
    remote_sha = latest_commit_sha(repo_id)
    local_manifest = load_manifest(cache_dir)
    if local_manifest is not None and local_manifest.repo_sha == remote_sha:
        return False
    parquet_path = pull_parquet(repo_id, f"{remote_sha}.parquet", cache_dir)
    rehydrate_index_from_parquet(parquet_path, cache_dir, repo_slug, remote_sha)
    return True
