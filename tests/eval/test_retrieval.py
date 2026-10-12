import pytest

from src.eval.retrieval import file_recall, hit_at_k, ranked_files, reciprocal_rank
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import RetrievedChunk


def _retrieved(file_path: str, name: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=name,
        file_path=file_path,
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        repo_sha="sha1",
        text="",
    )
    return RetrievedChunk(chunk=chunk, score=0.0)


class TestRankedFiles:
    def test_dedupes_files_keeping_first_appearance_order(self):
        retrieved = [_retrieved("a.py", "1"), _retrieved("b.py", "2"), _retrieved("a.py", "3")]

        assert ranked_files(retrieved) == ["a.py", "b.py"]


class TestHitAtK:
    def test_hit_when_an_expected_file_is_within_k(self):
        assert hit_at_k(["x.py", "a.py"], ["a.py"], k=2) == 1.0

    def test_miss_when_the_expected_file_is_beyond_k(self):
        assert hit_at_k(["x.py", "a.py"], ["a.py"], k=1) == 0.0


class TestReciprocalRank:
    def test_is_one_over_the_rank_of_the_first_expected_file(self):
        assert reciprocal_rank(["x.py", "y.py", "a.py"], ["a.py", "y.py"]) == pytest.approx(0.5)

    def test_is_zero_when_nothing_expected_was_retrieved(self):
        assert reciprocal_rank(["x.py"], ["a.py"]) == 0.0


class TestFileRecall:
    def test_is_the_fraction_of_expected_files_retrieved(self):
        assert file_recall(["a.py", "x.py"], ["a.py", "b.py"]) == pytest.approx(0.5)

    def test_is_zero_for_no_expected_files(self):
        assert file_recall(["a.py"], []) == 0.0
