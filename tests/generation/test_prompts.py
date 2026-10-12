from src.config import GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS
from src.generation.prompts import fit_chunks_to_budget, render_prompt
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import RetrievedChunk


class _FakeBackend:
    """A token counter with a small, fixed context window — the real CPUBackend
    counts real tokens, but a whitespace word-count is a good enough, fast proxy here since
    these tests only care about relative sizes, not exact tokenizer output."""

    def __init__(self, context_window: int) -> None:
        self.context_window = context_window

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_tokens: int) -> str:
        raise AssertionError("fit_chunks_to_budget should never call generate")


def _fake_chunk(name: str, text: str, imports: str = "") -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path="starlette/authentication.py",
        start_line=10,
        end_line=20,
        symbol_name=name,
        qualified_name=name,
        symbol_type="class",
        language="python",
        repo_sha="sha1",
        text=text,
        imports=imports,
    )


class TestRenderPrompt:
    def test_includes_citation_format_and_i_dont_know_instructions(self):
        prompt = render_prompt("question", [])

        assert "path/to/file.py#L10-L20" in prompt
        assert "copied exactly" in prompt
        assert "Example of a correctly cited answer" in prompt
        assert "I don't know" in prompt

    def test_includes_each_chunk_file_span_and_text(self):
        chunk = _fake_chunk("AuthenticationMiddleware", "class AuthenticationMiddleware: ...")

        prompt = render_prompt("how is auth validated?", [RetrievedChunk(chunk=chunk, score=1.0)])

        assert "starlette/authentication.py#L10-L20" in prompt
        assert "class AuthenticationMiddleware: ..." in prompt

    def test_includes_a_chunks_imports_when_present(self):
        chunk = _fake_chunk(
            "AuthenticationMiddleware",
            "class AuthenticationMiddleware: ...",
            imports="import typing",
        )

        prompt = render_prompt("how is auth validated?", [RetrievedChunk(chunk=chunk, score=1.0)])

        assert "Imports:\nimport typing" in prompt
        assert prompt.index("Imports:") < prompt.index("class AuthenticationMiddleware")

    def test_omits_the_imports_section_when_a_chunk_has_none(self):
        chunk = _fake_chunk("AuthenticationMiddleware", "class AuthenticationMiddleware: ...")

        prompt = render_prompt("how is auth validated?", [RetrievedChunk(chunk=chunk, score=1.0)])

        assert "Imports:" not in prompt

    def test_includes_the_question(self):
        prompt = render_prompt("how does auth work?", [])

        assert "how does auth work?" in prompt

    def test_ends_with_a_citation_reminder_immediately_before_answer(self):
        prompt = render_prompt("q", [])

        assert prompt.rstrip().endswith("Answer:")
        reminder_index = prompt.index("Reminder:")
        answer_index = prompt.rindex("Answer:")
        assert reminder_index < answer_index


class TestFitChunksToBudget:
    def _three_ranked_chunks(self) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(chunk=_fake_chunk("a", "word " * 5), score=3.0),
            RetrievedChunk(chunk=_fake_chunk("b", "word " * 5), score=2.0),
            RetrievedChunk(chunk=_fake_chunk("c", "word " * 5), score=1.0),
        ]

    def test_keeps_every_chunk_when_the_budget_is_generous(self):
        backend = _FakeBackend(context_window=100_000)
        chunks = self._three_ranked_chunks()

        fitted = fit_chunks_to_budget("q", chunks, backend, max_output_tokens=512)

        assert fitted == chunks

    def test_drops_lowest_ranked_chunks_that_dont_fit_the_budget(self):
        backend = _FakeBackend(context_window=0)
        question = "q"
        chunks = self._three_ranked_chunks()
        scaffold_tokens = backend.count_tokens(render_prompt(question, []))
        block_tokens = backend.count_tokens(render_prompt(question, [chunks[0]])) - scaffold_tokens
        backend.context_window = (
            scaffold_tokens + 2 * block_tokens + GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS
        )

        fitted = fit_chunks_to_budget(question, chunks, backend, max_output_tokens=0)

        assert [r.chunk.symbol_name for r in fitted] == ["a", "b"]

    def test_reserves_room_for_the_chat_template(self):
        backend = _FakeBackend(context_window=0)
        question = "q"
        chunks = self._three_ranked_chunks()
        scaffold_tokens = backend.count_tokens(render_prompt(question, []))
        block_tokens = backend.count_tokens(render_prompt(question, [chunks[0]])) - scaffold_tokens
        backend.context_window = scaffold_tokens + 2 * block_tokens

        fitted = fit_chunks_to_budget(question, chunks, backend, max_output_tokens=0)

        assert len(fitted) < 2

    def test_skips_an_oversized_chunk_instead_of_dropping_everything_after_it(self):
        backend = _FakeBackend(context_window=0)
        question = "q"
        small = self._three_ranked_chunks()
        huge = RetrievedChunk(chunk=_fake_chunk("huge", "word " * 500), score=9.0)
        scaffold_tokens = backend.count_tokens(render_prompt(question, []))
        block_tokens = backend.count_tokens(render_prompt(question, [small[0]])) - scaffold_tokens
        backend.context_window = (
            scaffold_tokens + 3 * block_tokens + GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS
        )

        fitted = fit_chunks_to_budget(question, [huge, *small], backend, max_output_tokens=0)

        assert [r.chunk.symbol_name for r in fitted] == ["a", "b", "c"]

    def test_returns_empty_when_no_budget_remains_after_output_and_scaffold(self):
        backend = _FakeBackend(context_window=1)
        chunks = self._three_ranked_chunks()

        fitted = fit_chunks_to_budget("q", chunks, backend, max_output_tokens=512)

        assert fitted == []

    def test_logs_a_warning_when_trimming_occurs(self, caplog):
        backend = _FakeBackend(context_window=0)
        question = "q"
        chunks = self._three_ranked_chunks()
        scaffold_tokens = backend.count_tokens(render_prompt(question, []))
        block_tokens = backend.count_tokens(render_prompt(question, [chunks[0]])) - scaffold_tokens
        backend.context_window = (
            scaffold_tokens + 2 * block_tokens + GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS
        )

        with caplog.at_level("WARNING"):
            fit_chunks_to_budget(question, chunks, backend, max_output_tokens=0)

        assert "Trimmed context from 3 to 2 chunk" in caplog.text

    def test_does_not_log_a_warning_when_nothing_is_trimmed(self, caplog):
        backend = _FakeBackend(context_window=100_000)
        chunks = self._three_ranked_chunks()

        with caplog.at_level("WARNING"):
            fit_chunks_to_budget("q", chunks, backend, max_output_tokens=512)

        assert "Trimmed" not in caplog.text


class TestExampleCitation:
    def test_example_citation_uses_the_first_context_items_real_location(self):
        chunk = _fake_chunk("AuthenticationMiddleware", "class AuthenticationMiddleware: ...")

        prompt = render_prompt("q", [RetrievedChunk(chunk=chunk, score=1.0)])

        assert "(starlette/authentication.py#L10-L20)." in prompt
        assert "path/to/file.py" not in prompt
