from src.generation.citations import Citation, verify_citations
from src.ingestion.schemas import Chunk


def _fake_chunk(file_path: str, start_line: int, end_line: int) -> Chunk:
    return Chunk(
        chunk_id="c1",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name="foo",
        qualified_name="foo",
        symbol_type="function",
        language="python",
        repo_sha="sha1",
        text="pass",
    )


class TestVerifyCitations:
    def test_keeps_a_real_citation_verified_and_flags_a_fabricated_one(self):
        chunks = [_fake_chunk("starlette/authentication.py", 10, 20)]
        answer = (
            "Auth is checked in starlette/authentication.py#L10-L20 "
            "and also in starlette/made_up.py#L1-L5."
        )

        real, fake = verify_citations(answer, chunks)

        assert real == Citation("starlette/authentication.py", 10, 20, verified=True)
        assert fake == Citation("starlette/made_up.py", 1, 5, verified=False)

    def test_flags_a_span_outside_the_real_chunks_line_range_as_unverified(self):
        chunks = [_fake_chunk("a.py", 10, 20)]

        citations = verify_citations("see a.py#L1-L5", chunks)

        assert citations == [Citation("a.py", 1, 5, verified=False)]

    def test_returns_empty_list_when_no_citations_are_present(self):
        assert verify_citations("no citations here", []) == []
