from pathlib import Path

import numpy as np
import pytest

from src.ingestion.indexer import build_index
from src.ingestion.schemas import Chunk
from src.pipeline import answer_ragnarok, compare


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


class _ScriptedBackend:
    """Answers agent decide-more-hops prompts from a script, and every other prompt with answer."""

    def __init__(self, decisions: list[str], answer: str) -> None:
        self._decisions = iter(decisions)
        self.answer = answer

    def generate(self, prompt: str, max_tokens: int) -> str:
        if "Reply:" in prompt:
            return next(self._decisions, "DONE")
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


class TestAnswerRagnarok:
    def test_returns_a_ragnarok_tagged_result_with_verified_citations(self, cache_dir):
        backend = _ScriptedBackend(["DONE"], "starlette/authentication.py#L1-L2 handles auth.")

        result = answer_ragnarok(
            "how is auth handled?", cache_dir, _FakeEmbedder(), _FakeReranker(), backend, top_k=2
        )

        assert result.pipeline == "ragnarok"
        assert result.citations[0].file_path == "starlette/authentication.py"
        assert result.citations[0].verified is True
        assert result.hop_path == []
        assert result.context_texts != []

    def test_max_hops_zero_disables_hopping(self, cache_dir):
        backend = _ScriptedBackend(["auth"], "an answer")

        result = answer_ragnarok(
            "how is auth handled?",
            cache_dir,
            _FakeEmbedder(),
            _FakeReranker(),
            backend,
            top_k=2,
            max_hops=0,
        )

        assert result.hop_path == []


class TestCompare:
    def test_returns_naive_then_ragnarok_tagged_results(self, cache_dir):
        backend = _ScriptedBackend(["DONE"], "an answer, no citations.")

        naive_result, ragnarok_result = compare(
            "how is auth handled?", cache_dir, _FakeEmbedder(), _FakeReranker(), backend, top_k=2
        )

        assert naive_result.pipeline == "naive"
        assert ragnarok_result.pipeline == "ragnarok"
