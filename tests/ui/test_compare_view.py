import gradio as gr

from src.agent.state import HopRecord
from src.baseline.schemas import PipelineResult
from src.generation.citations import Citation
from src.ui.compare_view import _format_result, build_compare_view, render_comparison


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


class TestFormatResult:
    def test_naive_result_never_shows_a_verified_badge(self):
        formatted = _format_result(_naive_result("plain answer"))

        assert formatted == "plain answer"
        assert "Verified" not in formatted

    def test_ragnarok_result_with_citations_shows_a_verified_badge(self):
        citation = Citation("a.py", 1, 2, verified=True)

        formatted = _format_result(_ragnarok_result(citations=[citation]))

        assert "**Verified citations**" in formatted

    def test_ragnarok_result_without_citations_does_not_claim_verification(self):
        formatted = _format_result(_ragnarok_result(citations=[]))

        assert "No citations found" in formatted
        assert "Verified citations" not in formatted

    def test_ragnarok_result_with_hops_lists_the_hop_path(self):
        hop = HopRecord(file_path="a.py", symbol_name="foo", start_line=1, end_line=2, reason="r")

        formatted = _format_result(_ragnarok_result(hop_path=[hop]))

        assert "**Hop path:** foo" in formatted


class TestRenderComparison:
    def test_returns_formatted_naive_and_ragnarok_columns_in_order(self):
        def fake_compare(question: str) -> tuple[PipelineResult, PipelineResult]:
            return _naive_result(f"naive: {question}"), _ragnarok_result(f"ragnarok: {question}")

        naive_text, ragnarok_text = render_comparison(fake_compare, "how is auth handled?")

        assert naive_text == "naive: how is auth handled?"
        assert "ragnarok: how is auth handled?" in ragnarok_text


class TestBuildCompareView:
    def test_returns_a_gradio_blocks_instance(self):
        view = build_compare_view(lambda question: (_naive_result(), _ragnarok_result()))

        assert isinstance(view, gr.Blocks)
