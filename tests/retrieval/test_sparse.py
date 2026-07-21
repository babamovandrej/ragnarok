import numpy as np
import pytest

from src.ingestion.indexer import BM25_DIRNAME, build_index, load_chroma_collection
from src.ingestion.schemas import Chunk
from src.retrieval.sparse import query_sparse


def _fake_chunk(name: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"a.py:{name}",
        file_path="a.py",
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        text=text,
    )


@pytest.fixture
def index_dir(tmp_path):
    chunks = [
        _fake_chunk("auth", "class AuthenticationMiddleware:\n    pass"),
        _fake_chunk("add", "def add(a, b):\n    return a + b"),
        _fake_chunk("sub", "def subtract(a, b):\n    return a - b"),
    ]
    embeddings = np.zeros((3, 4), dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")
    return tmp_path


class TestQuerySparse:
    def test_returns_exact_symbol_match_as_top_result(self, index_dir):
        collection = load_chroma_collection(index_dir)

        results = query_sparse(
            index_dir / BM25_DIRNAME, collection, "AuthenticationMiddleware", top_k=3
        )

        assert results[0].chunk.symbol_name == "auth"

    def test_clamps_top_k_to_corpus_size(self, index_dir):
        collection = load_chroma_collection(index_dir)

        results = query_sparse(index_dir / BM25_DIRNAME, collection, "add", top_k=100)

        assert len(results) == 3

    def test_scores_are_descending(self, index_dir):
        collection = load_chroma_collection(index_dir)

        results = query_sparse(index_dir / BM25_DIRNAME, collection, "add subtract", top_k=3)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_reconstructs_full_chunk_from_metadata(self, index_dir):
        collection = load_chroma_collection(index_dir)

        results = query_sparse(
            index_dir / BM25_DIRNAME, collection, "AuthenticationMiddleware", top_k=1
        )

        chunk = results[0].chunk
        assert chunk.chunk_id == "a.py:auth"
        assert chunk.text == "class AuthenticationMiddleware:\n    pass"
