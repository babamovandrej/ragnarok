import numpy as np
import pyarrow.parquet as pq

from src.ingestion.schemas import Chunk
from src.utils.hf_dataset_io import export_chunks_to_parquet

def test_export_chunks_to_parquet_has_expected_columns(tmp_path):
    chunks = [
        Chunk(
            chunk_id="a.py:1-2:foo",
            file_path="a.py",
            start_line=1,
            end_line=2,
            symbol_name="foo",
            qualified_name="foo",
            symbol_type="function",
            language="python",
            text="def foo():\n    pass",
        )
    ]
    embeddings = np.zeros((1, 4), dtype=np.float32)
    output_path = tmp_path / "chunks.parquet"

    export_chunks_to_parquet(chunks, embeddings, output_path)
    table = pq.read_table(output_path)

    assert table.column_names == [
        "chunk_id",
        "file_path",
        "start_line",
        "end_line",
        "symbol_name",
        "qualified_name",
        "symbol_type",
        "language",
        "text",
        "embedding",
    ]
    assert table.num_rows == 1
    assert table.column("embedding")[0].as_py() == [0.0, 0.0, 0.0, 0.0]
