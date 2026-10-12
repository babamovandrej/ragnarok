from unittest.mock import patch

import pytest

from src.config import RERANK_INSTRUCTION, RERANK_MODEL_ID, RERANK_MODEL_REVISION
from src.ingestion.schemas import Chunk
from src.retrieval.rerank import Reranker, format_pair
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
        repo_sha="sha1",
        text=text,
    )


@pytest.fixture
def mock_hf():
    with (
        patch("src.retrieval.rerank.AutoTokenizer") as tokenizer,
        patch("src.retrieval.rerank.AutoModelForCausalLM") as model,
    ):
        yield tokenizer, model


class _ScoredReranker(Reranker):
    """A Reranker whose model call is replaced by a fixed score per document text."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        return [self._scores[document] for document in documents]


class TestFormatPair:
    def test_follows_the_model_cards_instruct_query_document_layout(self):
        assert format_pair("q", "d", instruction="i") == "<Instruct>: i\n<Query>: q\n<Document>: d"

    def test_defaults_to_the_configured_code_instruction(self):
        assert format_pair("q", "d").startswith(f"<Instruct>: {RERANK_INSTRUCTION}\n")


class TestReranker:
    def test_pins_model_id_and_revision_on_load(self, mock_hf):
        tokenizer, model = mock_hf

        Reranker(device="cpu")

        tokenizer.from_pretrained.assert_called_once_with(
            RERANK_MODEL_ID, revision=RERANK_MODEL_REVISION, padding_side="left"
        )
        model.from_pretrained.assert_called_once_with(
            RERANK_MODEL_ID, revision=RERANK_MODEL_REVISION
        )

    def test_falls_back_to_the_configured_default_device(self, mock_hf, monkeypatch):
        _, model = mock_hf
        monkeypatch.setattr("src.retrieval.rerank.DEFAULT_DEVICE", "cpu")

        Reranker()

        model.from_pretrained.return_value.to.assert_called_once_with("cpu")

    def test_reorders_candidates_by_score(self):
        irrelevant = _fake_chunk("irrelevant", "def noop(): pass")
        relevant = _fake_chunk("relevant", "class AuthenticationMiddleware: pass")
        candidates = [
            RetrievedChunk(chunk=irrelevant, score=1.0),
            RetrievedChunk(chunk=relevant, score=0.5),
        ]
        reranker = _ScoredReranker({irrelevant.search_text: -4.0, relevant.search_text: 3.0})

        reranked = reranker.rerank("how is auth validated?", candidates)

        assert [r.chunk.symbol_name for r in reranked] == ["relevant", "irrelevant"]
        assert [r.score for r in reranked] == [3.0, -4.0]

    def test_scores_the_query_against_each_chunks_location_prefixed_text(self):
        chunk = _fake_chunk("foo", "def foo(): pass")
        reranker = _ScoredReranker({chunk.search_text: 0.0})

        reranker.rerank("my query", [RetrievedChunk(chunk=chunk, score=0.0)])

        assert reranker.calls == [("my query", ["a.py · foo\ndef foo(): pass"])]

    def test_returns_empty_list_for_no_candidates(self):
        reranker = _ScoredReranker({})

        assert reranker.rerank("query", []) == []
        assert reranker.calls == []


@pytest.mark.integration
class TestRealModel:
    @pytest.fixture(scope="class")
    def reranker(self):
        return Reranker()

    def test_reorders_an_obviously_more_relevant_candidate_first(self, reranker):
        unrelated = _fake_chunk("unrelated", "def add(a, b):\n    return a + b")
        auth_text = (
            "class AuthenticationMiddleware:\n    def __call__(self, request):\n        pass"
        )
        auth = _fake_chunk("auth", auth_text)
        candidates = [
            RetrievedChunk(chunk=unrelated, score=0.0),
            RetrievedChunk(chunk=auth, score=0.0),
        ]

        reranked = reranker.rerank("how is authentication validated on the request?", candidates)

        assert reranked[0].chunk.symbol_name == "auth"
        assert reranked[0].score > reranked[1].score

    def test_batched_scores_match_one_at_a_time_scores(self, reranker):
        """Guards left-padding: a mis-padded batch reads the wrong last-token logits."""
        documents = ["short", "a much longer document " * 20, "def f(): return 1"]

        batched = reranker.score("a query", documents)
        single = [reranker.score("a query", [document])[0] for document in documents]

        assert batched == pytest.approx(single, abs=0.05)
