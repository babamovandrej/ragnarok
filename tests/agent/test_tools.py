from pathlib import Path

import numpy as np
import pytest

from src.agent.tools import find_symbol_definition, retrieve_hybrid
from src.ingestion.indexer import BM25_DIRNAME, build_index, load_chroma_collection
from src.ingestion.schemas import Chunk


def _fake_chunk(name: str, text: str, file_path: str = "a.py") -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path=file_path,
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        text=text,
    )


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


class _FakeReranker:
    def rerank(self, query, candidates):
        return candidates


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    chunks = [
        _fake_chunk(
            "AuthenticationMiddleware",
            "class AuthenticationMiddleware:\n    def __call__(self, request):\n        pass",
            "starlette/authentication.py",
        ),
        _fake_chunk("add", "def add(a, b):\n    return a + b"),
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")
    return tmp_path


class TestFindSymbolDefinition:
    def test_resolves_a_known_symbol_to_its_exact_defining_chunk(self, cache_dir):
        collection = load_chroma_collection(cache_dir)

        found = find_symbol_definition("AuthenticationMiddleware", collection)

        assert found is not None
        assert found.chunk.file_path == "starlette/authentication.py"
        assert found.chunk.symbol_name == "AuthenticationMiddleware"

    def test_returns_none_for_an_unknown_symbol(self, cache_dir):
        collection = load_chroma_collection(cache_dir)

        assert find_symbol_definition("NoSuchSymbol", collection) is None


class TestRetrieveHybrid:
    def test_returns_reranked_candidates_from_dense_and_sparse_fusion(self, cache_dir):
        collection = load_chroma_collection(cache_dir)

        results = retrieve_hybrid(
            "auth", collection, cache_dir / BM25_DIRNAME, _FakeEmbedder(), _FakeReranker(), top_k=2
        )

        assert {r.chunk.symbol_name for r in results} == {"AuthenticationMiddleware", "add"}
