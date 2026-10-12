from pathlib import Path

import numpy as np
import pytest

from src.ingestion.indexer import build_index
from src.ingestion.schemas import Chunk
from src.pipeline import answer_ragnarok, compare
from src.retrieval.schemas import RetrievedChunk


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


class _FakeReranker:
    def rerank(self, query, candidates):
        return [RetrievedChunk(chunk=c.chunk, score=5.0) for c in candidates]


class _RerankerWithoutAuth:
    """Leaves `auth` out of the initial retrieval, so the agent has to hop to fetch it."""

    def __init__(self) -> None:
        self._retrieved = False

    def rerank(self, query, candidates):
        kept = [c for c in candidates if self._retrieved or c.chunk.symbol_name != "auth"]
        self._retrieved = True
        return [RetrievedChunk(chunk=c.chunk, score=5.0) for c in kept]


class _ScriptedBackend:
    """Answers every prompt with a fixed answer."""

    context_window = 100_000

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_tokens: int) -> str:
        return self.answer


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    chunks = [
        _fake_chunk(
            "auth", "class AuthenticationMiddleware:\n    pass", "starlette/authentication.py"
        ),
        _fake_chunk("add", "def add(a, b):\n    return auth(a + b)"),
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")
    return tmp_path


class TestAnswerRagnarok:
    def test_returns_a_ragnarok_tagged_result_with_verified_citations(self, cache_dir):
        backend = _ScriptedBackend("starlette/authentication.py#L1-L2 handles auth.")

        result = answer_ragnarok(
            "how is auth handled?", cache_dir, _FakeEmbedder(), _FakeReranker(), backend, top_k=2
        )

        assert result.pipeline == "ragnarok"
        assert result.citations[0].file_path == "starlette/authentication.py"
        assert result.citations[0].verified is True
        assert result.hop_path == []
        assert result.context_texts != []

    def test_hops_to_a_relevant_definition_the_retrieved_code_refers_to(self, cache_dir):
        result = answer_ragnarok(
            "how is auth handled?",
            cache_dir,
            _FakeEmbedder(),
            _RerankerWithoutAuth(),
            _ScriptedBackend("an answer"),
            top_k=2,
        )

        assert [hop.symbol_name for hop in result.hop_path] == ["auth"]
        assert "class AuthenticationMiddleware:\n    pass" in result.context_texts

    def test_max_hops_zero_disables_hopping(self, cache_dir):
        result = answer_ragnarok(
            "how is auth handled?",
            cache_dir,
            _FakeEmbedder(),
            _RerankerWithoutAuth(),
            _ScriptedBackend("an answer"),
            top_k=2,
            max_hops=0,
        )

        assert result.hop_path == []

    def test_logs_and_reraises_when_the_graph_fails_to_answer(self, cache_dir, caplog):
        class _BrokenBackend(_ScriptedBackend):
            def generate(self, prompt: str, max_tokens: int) -> str:
                raise RuntimeError("model crashed")

        backend = _BrokenBackend("unused")

        with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="model crashed"):
            answer_ragnarok(
                "how is auth handled?",
                cache_dir,
                _FakeEmbedder(),
                _FakeReranker(),
                backend,
                top_k=2,
            )

        assert "RAGnarok pipeline failed" in caplog.text


class TestCompare:
    def test_returns_naive_then_ragnarok_tagged_results(self, cache_dir):
        backend = _ScriptedBackend("an answer, no citations.")

        naive_result, ragnarok_result = compare(
            "how is auth handled?", cache_dir, _FakeEmbedder(), _FakeReranker(), backend, top_k=2
        )

        assert naive_result.pipeline == "naive"
        assert ragnarok_result.pipeline == "ragnarok"
