from pathlib import Path

import numpy as np
import pytest

from scripts.eval_retrieval import (
    CONFIGS,
    EXIT_INVALID_ARGUMENT,
    build_parser,
    main,
    retrieve_all_configs,
    run,
)
from src.config import DEFAULT_RETRIEVAL_TOP_K
from src.eval.qa_dataset import QAItem
from src.ingestion.indexer import build_index
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import RetrievedChunk


def _fake_chunk(name: str, file_path: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path=file_path,
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
    def embed_query(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _AuthFirstReranker:
    def rerank(self, query, candidates):
        scored = [
            RetrievedChunk(chunk=c.chunk, score=5.0 if "auth" in c.chunk.file_path else -5.0)
            for c in candidates
        ]
        return sorted(scored, key=lambda r: r.score, reverse=True)


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    chunks = [
        _fake_chunk("add", "math.py", "def add(a, b):\n    return a + b"),
        _fake_chunk("auth", "auth.py", "class AuthenticationMiddleware:\n    pass"),
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")
    return tmp_path


def _item(category: str, expected: list[str]) -> QAItem:
    return QAItem(
        id=category,
        question="how is authentication handled?",
        category=category,
        expected_files=expected,
        grounding_notes="n",
    )


class TestRetrieveAllConfigs:
    def test_returns_every_config_and_the_best_rerank_score(self, cache_dir):
        results, best = retrieve_all_configs(
            "authentication", cache_dir, _FakeEmbedder(), _AuthFirstReranker(), 2
        )

        assert list(results) == list(CONFIGS)
        assert results[CONFIGS[0]][0].chunk.file_path == "math.py"
        assert results[CONFIGS[-1]][0].chunk.file_path == "auth.py"
        assert best == 5.0


class TestRun:
    def test_scores_in_repo_items_and_records_adversarial_best_scores(self, cache_dir):
        items = [
            _item("single_hop", ["auth.py"]),
            _item("multi_hop", ["auth.py", "math.py"]),
            _item("adversarial", []),
        ]

        report = run(items, cache_dir, _FakeEmbedder(), _AuthFirstReranker(), top_k=2)

        reranked = report["single_hop"][CONFIGS[-1]]
        assert reranked["hit@1"] == 1.0
        assert report["single_hop"][CONFIGS[0]]["hit@1"] == 0.0
        assert report["multi_hop"][CONFIGS[-1]]["file_recall@2"] == 1.0
        assert report["best_rerank_score"] == {"in_repo": [5.0, 5.0], "adversarial": [5.0]}


class TestMain:
    def test_parser_defaults_come_from_config(self):
        args = build_parser().parse_args(["/tmp/cache"])

        assert args.top_k == DEFAULT_RETRIEVAL_TOP_K
        assert args.device is None

    def test_rejects_a_missing_index_directory(self, tmp_path):
        assert main([str(tmp_path / "missing")]) == EXIT_INVALID_ARGUMENT
