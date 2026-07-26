import json
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_retrieval import (
    EXIT_INVALID_ARGUMENT,
    EXIT_SUCCESS,
    build_parser,
    compare_query,
    main,
)
from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.ingestion.indexer import build_index
from src.ingestion.schemas import Chunk


def _fake_chunk(name: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path="a.py",
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        repo_sha="sha1",
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
        _fake_chunk("auth", "class AuthenticationMiddleware:\n    pass"),
        _fake_chunk("add", "def add(a, b):\n    return a + b"),
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")
    return tmp_path


class TestCompareQuery:
    def test_returns_both_paths_ranked_by_qualified_name(self, cache_dir):
        result = compare_query("auth", cache_dir, _FakeEmbedder(), _FakeReranker(), top_k=2)

        assert set(result) == {"dense_only", "hybrid_reranked"}
        assert result["dense_only"] == ["auth", "add"]
        assert result["hybrid_reranked"] == ["auth", "add"]


class TestBuildParser:
    def test_parses_cache_dir_and_queries(self):
        args = build_parser().parse_args(["/tmp/cache", "how does auth work?", "another query"])

        assert args.cache_dir == Path("/tmp/cache")
        assert args.queries == ["how does auth work?", "another query"]

    def test_top_k_defaults_from_config(self):
        args = build_parser().parse_args(["/tmp/cache", "q"])

        assert args.top_k == DEFAULT_RETRIEVAL_TOP_K

    def test_json_flag_parses(self):
        args = build_parser().parse_args(["/tmp/cache", "q", "--json"])

        assert args.json is True


class TestMain:
    def test_missing_index_returns_invalid_argument(self, tmp_path):
        exit_code = main([str(tmp_path / "missing"), "q"])

        assert exit_code == EXIT_INVALID_ARGUMENT

    def test_json_output_contains_both_paths_per_query(self, cache_dir, monkeypatch, capsys):
        monkeypatch.setattr("scripts.compare_retrieval.Embedder", _FakeEmbedder)
        monkeypatch.setattr("scripts.compare_retrieval.Reranker", _FakeReranker)

        exit_code = main([str(cache_dir), "auth", "--json"])

        assert exit_code == EXIT_SUCCESS
        payload = json.loads(capsys.readouterr().out)
        assert set(payload["auth"]) == {"dense_only", "hybrid_reranked"}

    def test_human_readable_output_lists_each_query(self, cache_dir, monkeypatch, capsys):
        monkeypatch.setattr("scripts.compare_retrieval.Embedder", _FakeEmbedder)
        monkeypatch.setattr("scripts.compare_retrieval.Reranker", _FakeReranker)

        exit_code = main([str(cache_dir), "auth"])

        assert exit_code == EXIT_SUCCESS
        out = capsys.readouterr().out
        assert "Query: auth" in out
        assert "dense-only:" in out
        assert "hybrid+reranked:" in out
