import pytest

from scripts.build_index import ingest_repository
from src.config import DEMO_REPO_URL
from src.ingestion.indexer import load_chroma_collection


@pytest.mark.integration
def test_full_ingestion_pipeline_against_starlette(tmp_path):
    cache_dir = tmp_path / "starlette"

    result = ingest_repository(DEMO_REPO_URL, cache_dir=cache_dir)

    assert 100 <= result.chunk_count <= 2000
    assert result.embedding_dimension > 0
    assert not result.skipped

    collection = load_chroma_collection(cache_dir)
    assert collection.count() == result.chunk_count

    bm25_dir = cache_dir / "bm25"
    assert (bm25_dir / "params.index.json").exists()
