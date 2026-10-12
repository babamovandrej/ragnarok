from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.baseline.naive_rag import answer_naive
from src.config import GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS, GENERATION_MAX_TOKENS_DEFAULT
from src.ingestion.indexer import build_index
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
        repo_sha="sha1",
        text=text,
    )


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class _ScriptedBackend:
    """Answers every prompt with a fixed answer, recording the prompts it was given."""

    def __init__(self, answer: str, context_window: int = 100_000) -> None:
        self.answer = answer
        self.context_window = context_window
        self.prompts: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.answer


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    chunks = [
        _fake_chunk(
            "auth", "class AuthenticationMiddleware:\n    pass", "starlette/authentication.py"
        ),
        _fake_chunk("add", "def add(a, b):\n    return a + b"),
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")
    return tmp_path


class TestAnswerNaive:
    def test_never_calls_hybrid_sparse_or_rerank_paths(self, cache_dir):
        backend = _ScriptedBackend("a naive answer with no citation format.")

        with (
            patch("src.agent.tools.query_sparse") as mock_sparse,
            patch("src.agent.tools.fuse_rrf") as mock_fuse,
            patch("src.retrieval.rerank.Reranker.rerank") as mock_rerank,
        ):
            result = answer_naive(
                "how is auth handled?", cache_dir, _FakeEmbedder(), backend, top_k=2
            )

        mock_sparse.assert_not_called()
        mock_fuse.assert_not_called()
        mock_rerank.assert_not_called()
        assert result.pipeline == "naive"
        assert result.citations == []
        assert result.hop_path == []
        assert result.answer == backend.answer
        assert set(result.context_texts) == {
            "class AuthenticationMiddleware:\n    pass",
            "def add(a, b):\n    return a + b",
        }

    def test_drops_context_that_would_overflow_the_models_context_window(self, cache_dir):
        question = "how is auth handled?"
        scaffold_tokens = len(f"Context: Question: {question} Answer:".split())
        auth_tokens = len("class AuthenticationMiddleware: pass".split())
        window = (
            GENERATION_MAX_TOKENS_DEFAULT
            + GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS
            + scaffold_tokens
            + auth_tokens
        )
        backend = _ScriptedBackend("an answer", context_window=window)

        result = answer_naive(question, cache_dir, _FakeEmbedder(), backend, top_k=2)

        assert result.context_texts == ["class AuthenticationMiddleware:\n    pass"]
        assert "def add" not in backend.prompts[0]
        assert (
            backend.count_tokens(backend.prompts[0])
            + GENERATION_MAX_TOKENS_DEFAULT
            + GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS
            <= window
        )
