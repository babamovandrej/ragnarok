import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from chromadb.api.types import IncludeEnum

from src.ingestion.indexer import load_chroma_collection, load_manifest
from src.retrieval.schemas import chunk_from_chroma_record
from src.utils.hf_dataset_io import export_chunks_to_parquet, push_to_hub

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_INVALID_ARGUMENT = 2


def export_and_push(cache_dir: Path, repo_id: str) -> Path:
    """Export cache_dir's index to our parquet schema and push it to a HF Dataset repo."""
    manifest = load_manifest(cache_dir)
    if manifest is None:
        raise ValueError(f"No manifest at {cache_dir}; run scripts/build_index.py first.")

    collection = load_chroma_collection(cache_dir)
    records = collection.get(
        include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.embeddings]
    )
    documents, metadatas = records["documents"], records["metadatas"]
    assert documents is not None and metadatas is not None
    chunks = [
        chunk_from_chroma_record(chunk_id, text, metadata)
        for chunk_id, text, metadata in zip(records["ids"], documents, metadatas)
    ]
    embeddings = np.array(records["embeddings"])

    parquet_path = cache_dir / f"{manifest.repo_sha}.parquet"
    export_chunks_to_parquet(chunks, embeddings, parquet_path)
    push_to_hub(parquet_path, repo_id)
    logger.info(
        "Pushed %d chunk(s) for %s@%s to %s",
        len(chunks),
        manifest.repo_slug,
        manifest.repo_sha,
        repo_id,
    )
    return parquet_path


def build_parser() -> argparse.ArgumentParser:
    """Construct the push_index_to_hub CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Export a built index to parquet and push it to a Hugging Face Dataset repo."
    )
    parser.add_argument("cache_dir", type=Path, help="Index directory produced by build_index.py.")
    parser.add_argument("repo_id", help="Target HF Dataset repo, e.g. 'namespace/ragnarok-index'.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.cache_dir.is_dir():
        logger.error("No index found at %s; run scripts/build_index.py first.", args.cache_dir)
        return EXIT_INVALID_ARGUMENT

    export_and_push(args.cache_dir, args.repo_id)
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
