import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from huggingface_hub import HfApi
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
