import numpy as np
import pytest

from src.ingestion.indexer import build_index, load_chroma_collection
from src.ingestion.schemas import Chunk
from src.retrieval.dense import query_dense


def _fake_chunk(name: str) -> Chunk:
    return Chunk(
        chunk_id=f"a.py:{name}",
        file_path="a.py",
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        repo_sha="sha1",
        text=f"def {name}():\n    pass",
    )


@pytest.fixture
def collection(tmp_path):
    chunks = [_fake_chunk("foo"), _fake_chunk("bar"), _fake_chunk("baz")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")
    return load_chroma_collection(tmp_path)


class TestQueryDense:
    def test_returns_top_k_ordered_by_cosine_similarity(self, collection):
        results = query_dense(collection, np.array([1.0, 0.0], dtype=np.float32), top_k=2)

        assert [r.chunk.symbol_name for r in results] == ["foo", "baz"]

    def test_top_result_scores_near_perfect_cosine_similarity(self, collection):
        results = query_dense(collection, np.array([1.0, 0.0], dtype=np.float32), top_k=1)

        assert results[0].score == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_match_scores_near_zero(self, collection):
        results = query_dense(collection, np.array([1.0, 0.0], dtype=np.float32), top_k=3)

        bar = next(r for r in results if r.chunk.symbol_name == "bar")
        assert bar.score == pytest.approx(0.0, abs=1e-6)

    def test_reconstructs_full_chunk_from_metadata(self, collection):
        results = query_dense(collection, np.array([1.0, 0.0], dtype=np.float32), top_k=1)

        chunk = results[0].chunk
        assert chunk.chunk_id == "a.py:foo"
        assert chunk.qualified_name == "foo"
        assert chunk.symbol_type == "function"
        assert chunk.text == "def foo():\n    pass"

    def test_default_top_k_returns_whole_collection(self, collection):
        results = query_dense(collection, np.array([1.0, 0.0], dtype=np.float32))

        assert len(results) == 3
