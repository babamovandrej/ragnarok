from src.generation.prompts import render_prompt
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import RetrievedChunk


def _fake_chunk(name: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path="starlette/authentication.py",
        start_line=10,
        end_line=20,
        symbol_name=name,
        qualified_name=name,
        symbol_type="class",
        language="python",
        text=text,
    )


class TestRenderPrompt:
    def test_includes_citation_format_and_i_dont_know_instructions(self):
        prompt = render_prompt("question", [])

        assert "file#Lstart-Lend" in prompt
        assert "I don't know" in prompt

    def test_includes_each_chunk_file_span_and_text(self):
        chunk = _fake_chunk("AuthenticationMiddleware", "class AuthenticationMiddleware: ...")

        prompt = render_prompt("how is auth validated?", [RetrievedChunk(chunk=chunk, score=1.0)])

        assert "starlette/authentication.py#L10-L20" in prompt
        assert "class AuthenticationMiddleware: ..." in prompt

    def test_includes_the_question(self):
        prompt = render_prompt("how does auth work?", [])

        assert "how does auth work?" in prompt
