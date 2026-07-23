import gradio as gr
import pytest

from src.agent.state import HopRecord
from src.baseline.schemas import PipelineResult
from src.generation.citations import Citation
from src.ui.compare_view import (
    _format_naive,
    _format_ragnarok,
    _render_citations,
    build_compare_view,
    render_comparison,
)


def _naive_result(answer: str = "a naive answer") -> PipelineResult:
    return PipelineResult(pipeline="naive", answer=answer, citations=[], hop_path=[])


def _ragnarok_result(
    answer: str = "a ragnarok answer",
    citations: list[Citation] | None = None,
    hop_path: list[HopRecord] | None = None,
) -> PipelineResult:
    return PipelineResult(
        pipeline="ragnarok", answer=answer, citations=citations or [], hop_path=hop_path or []
    )


class TestFormatNaive:
    def test_returns_the_raw_answer_with_no_extra_sections(self):
        assert _format_naive(_naive_result("plain answer")) == "plain answer"


class TestRenderCitations:
    def test_no_citations_says_none_found(self):
        assert _render_citations([], repo_context=None) == "_No citations found_"

    def test_lists_each_citation_with_its_own_verified_status(self):
        citations = [
            Citation("a.py", 1, 2, verified=True),
            Citation("b.py", 3, 4, verified=False),
        ]

        rendered = _render_citations(citations, repo_context=None)

        assert "`a.py#L1-L2` — verified" in rendered
        assert "`b.py#L3-L4` — UNVERIFIED" in rendered

    def test_links_citations_to_github_when_repo_context_is_known(self):
        citation = Citation("a.py", 1, 2, verified=True)

        rendered = _render_citations([citation], repo_context=("owner/repo", "sha123"))

        assert "[`a.py#L1-L2`](https://github.com/owner/repo/blob/sha123/a.py#L1-L2)" in rendered


class TestFormatRagnarok:
    def test_includes_the_answer_and_citations_section(self):
        citation = Citation("a.py", 1, 2, verified=True)

        formatted = _format_ragnarok(_ragnarok_result("the answer", [citation]), repo_context=None)

        assert "the answer" in formatted
        assert "verified" in formatted

    def test_falls_back_to_a_plain_hop_list_without_repo_context(self):
        hop = HopRecord(file_path="a.py", symbol_name="foo", start_line=1, end_line=2, reason="r")

        formatted = _format_ragnarok(_ragnarok_result(hop_path=[hop]), repo_context=None)

        assert "**Hop path:** foo" in formatted

    def test_renders_clickable_hop_links_with_repo_context(self):
        hop = HopRecord(file_path="a.py", symbol_name="foo", start_line=1, end_line=2, reason="r")

        formatted = _format_ragnarok(
            _ragnarok_result(hop_path=[hop]), repo_context=("owner/repo", "sha123")
        )

        assert "[`foo`" in formatted
        assert "https://github.com/owner/repo/blob/sha123/a.py#L1-L2" in formatted


class TestRenderComparison:
    def test_returns_formatted_naive_and_ragnarok_columns_in_order(self):
        def fake_compare(question: str) -> tuple[PipelineResult, PipelineResult]:
            return _naive_result(f"naive: {question}"), _ragnarok_result(f"ragnarok: {question}")

        naive_text, ragnarok_text = render_comparison(fake_compare, "how is auth handled?")

        assert naive_text == "naive: how is auth handled?"
        assert "ragnarok: how is auth handled?" in ragnarok_text

    @pytest.mark.parametrize("blank_question", ["", "   "])
    def test_empty_question_short_circuits_without_calling_compare_fn(self, blank_question):
        def unreachable_compare(question: str) -> tuple[PipelineResult, PipelineResult]:
            raise AssertionError("compare_fn should not be called for an empty question")

        naive_text, ragnarok_text = render_comparison(unreachable_compare, blank_question)

        assert "Enter a question" in naive_text
        assert "Enter a question" in ragnarok_text

    def test_backend_failure_renders_a_friendly_inline_message_in_both_columns(self):
        def failing_compare(question: str) -> tuple[PipelineResult, PipelineResult]:
            raise RuntimeError("model failed to load")

        naive_text, ragnarok_text = render_comparison(failing_compare, "a question")

        assert "went wrong" in naive_text
        assert "model failed to load" in naive_text
        assert "went wrong" in ragnarok_text


class TestBuildCompareView:
    def test_returns_a_gradio_blocks_instance(self):
        view = build_compare_view(lambda question: (_naive_result(), _ragnarok_result()))

        assert isinstance(view, gr.Blocks)
