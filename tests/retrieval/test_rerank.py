from unittest.mock import patch

import pytest

from src.config import RERANK_MODEL_ID, RERANK_MODEL_REVISION
from src.ingestion.schemas import Chunk
from src.retrieval.rerank import Reranker
from src.retrieval.schemas import RetrievedChunk


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
        text=text,
    )


@pytest.fixture
def mock_cross_encoder():
    with patch("src.retrieval.rerank.CrossEncoder") as mock_cross_encoder:
        yield mock_cross_encoder


class TestReranker:
    def test_pins_model_id_and_revision_on_load(self, mock_cross_encoder):
        Reranker(device="cpu")

        mock_cross_encoder.assert_called_once_with(
            RERANK_MODEL_ID, revision=RERANK_MODEL_REVISION, device="cpu"
        )

    def test_reorders_candidates_by_cross_encoder_score(self, mock_cross_encoder):
        mock_cross_encoder.return_value.predict.return_value = [0.1, 0.9]
        candidates = [
            RetrievedChunk(chunk=_fake_chunk("irrelevant", "def noop(): pass"), score=1.0),
            RetrievedChunk(
                chunk=_fake_chunk("relevant", "class AuthenticationMiddleware: pass"),
                score=0.5,
            ),
        ]

        reranked = Reranker().rerank("how is auth validated?", candidates)

        assert [r.chunk.symbol_name for r in reranked] == ["relevant", "irrelevant"]
        assert reranked[0].score == pytest.approx(0.9)
        assert reranked[1].score == pytest.approx(0.1)

    def test_passes_query_and_chunk_text_pairs_to_the_model(self, mock_cross_encoder):
        mock_cross_encoder.return_value.predict.return_value = [0.0]
        candidates = [RetrievedChunk(chunk=_fake_chunk("foo", "def foo(): pass"), score=0.0)]

        Reranker().rerank("my query", candidates)

        (pairs,), _ = mock_cross_encoder.return_value.predict.call_args
        assert pairs == [("my query", "def foo(): pass")]

    def test_returns_empty_list_for_no_candidates(self, mock_cross_encoder):
        assert Reranker().rerank("query", []) == []
        mock_cross_encoder.return_value.predict.assert_not_called()


@pytest.mark.integration
def test_real_model_reorders_an_obviously_more_relevant_candidate_first():
    unrelated = _fake_chunk("unrelated", "def add(a, b):\n    return a + b")
    auth_text = "class AuthenticationMiddleware:\n    def __call__(self, request):\n        pass"
    auth = _fake_chunk("auth", auth_text)
    candidates = [
        RetrievedChunk(chunk=unrelated, score=0.0),
        RetrievedChunk(chunk=auth, score=0.0),
    ]

    reranked = Reranker().rerank("how is authentication validated on the request?", candidates)

    assert reranked[0].chunk.symbol_name == "auth"
