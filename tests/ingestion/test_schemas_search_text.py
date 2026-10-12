from src.ingestion.schemas import Chunk


def _chunk(file_path: str, qualified_name: str, text: str = "code") -> Chunk:
    return Chunk(
        chunk_id="id",
        file_path=file_path,
        start_line=1,
        end_line=2,
        symbol_name=qualified_name,
        qualified_name=qualified_name,
        symbol_type="function",
        language="python",
        repo_sha="sha1",
        text=text,
    )


class TestSearchText:
    def test_prefixes_the_code_with_its_file_path_and_qualified_name(self):
        chunk = _chunk("starlette/responses.py", "RedirectResponse", "class RedirectResponse: ...")

        assert chunk.search_text == (
            "starlette/responses.py · RedirectResponse\nclass RedirectResponse: ..."
        )

    def test_whole_file_chunks_are_prefixed_with_the_path_only(self):
        chunk = _chunk("pyproject.toml", "pyproject.toml", "[project]")

        assert chunk.search_text == "pyproject.toml\n[project]"

    def test_leaves_the_stored_code_untouched(self):
        chunk = _chunk("a.py", "foo", "def foo(): ...")

        assert chunk.text == "def foo(): ..."
