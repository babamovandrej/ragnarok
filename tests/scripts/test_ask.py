from pathlib import Path

import numpy as np
import pytest

from scripts.ask import EXIT_INVALID_ARGUMENT, EXIT_SUCCESS, answer_question, build_parser, main
from src.config import DEFAULT_RETRIEVAL_TOP_K, GENERATION_MAX_TOKENS_DEFAULT
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
        text=text,
    )


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


class _FakeReranker:
    def rerank(self, query, candidates):
        return candidates


class _FakeBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.calls.append((prompt, max_tokens))
        return self.response


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


class TestAnswerQuestion:
    def test_returns_the_generated_answer_and_its_verified_citations(self, cache_dir):
        backend = _FakeBackend("starlette/authentication.py#L1-L2 handles auth.")

        answer, citations = answer_question(
            "how is auth handled?",
            cache_dir,
            _FakeEmbedder(),
            _FakeReranker(),
            backend,
            top_k=2,
            max_tokens=GENERATION_MAX_TOKENS_DEFAULT,
        )

        assert answer == backend.response
        assert citations[0].file_path == "starlette/authentication.py"
        assert citations[0].verified is True

    def test_passes_max_tokens_through_to_the_backend(self, cache_dir):
        backend = _FakeBackend("no citations here")

        answer_question(
            "q", cache_dir, _FakeEmbedder(), _FakeReranker(), backend, top_k=2, max_tokens=64
        )

        assert backend.calls[0][1] == 64


class TestBuildParser:
    def test_parses_cache_dir_and_question(self):
        args = build_parser().parse_args(["/tmp/cache", "how does auth work?"])

        assert args.cache_dir == Path("/tmp/cache")
        assert args.question == "how does auth work?"

    def test_defaults_come_from_config(self):
        args = build_parser().parse_args(["/tmp/cache", "q"])

        assert args.top_k == DEFAULT_RETRIEVAL_TOP_K
        assert args.max_tokens == GENERATION_MAX_TOKENS_DEFAULT


class TestMain:
    def test_missing_index_returns_invalid_argument(self, tmp_path):
        exit_code = main([str(tmp_path / "missing"), "q"])

        assert exit_code == EXIT_INVALID_ARGUMENT

    def test_prints_the_answer_and_its_citation_verification_status(
        self, cache_dir, monkeypatch, capsys
    ):
        monkeypatch.setattr("scripts.ask.Embedder", _FakeEmbedder)
        monkeypatch.setattr("scripts.ask.Reranker", _FakeReranker)
        monkeypatch.setattr(
            "scripts.ask.CPUBackend",
            lambda: _FakeBackend("starlette/authentication.py#L1-L2 handles auth."),
        )

        exit_code = main([str(cache_dir), "how is auth handled?"])

        assert exit_code == EXIT_SUCCESS
        out = capsys.readouterr().out
        assert "Answer:" in out
        assert "starlette/authentication.py#L1-L2 [verified]" in out
