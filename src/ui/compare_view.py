from collections.abc import Callable
from functools import partial

import gradio as gr

from src.baseline.schemas import PipelineResult

CompareFn = Callable[[str], tuple[PipelineResult, PipelineResult]]


def render_comparison(compare_fn: CompareFn, question: str) -> tuple[str, str]:
    """Run compare_fn and render both pipeline results as markdown for the two UI columns."""
    naive_result, ragnarok_result = compare_fn(question)
    return _format_result(naive_result), _format_result(ragnarok_result)


def _format_result(result: PipelineResult) -> str:
    """Render one PipelineResult as markdown; only ragnarok ever shows a verified badge."""
    lines = [result.answer]
    if result.pipeline == "ragnarok":
        lines.append("**Verified citations**" if result.citations else "_No citations found_")
        if result.hop_path:
            hops = ", ".join(hop.symbol_name for hop in result.hop_path)
            lines.append(f"**Hop path:** {hops}")
    return "\n\n".join(lines)


def build_compare_view(compare_fn: CompareFn) -> gr.Blocks:
    """Build a two-column Gradio Blocks UI comparing the naive and RAGnarok pipelines."""
    view: gr.Blocks
    with gr.Blocks(title="RAGnarok — naive vs. hybrid+rerank+agent") as view:
        gr.Markdown(
            "# RAGnarok\n\nCompare a naive single-vector baseline against the full "
            "hybrid+rerank+multi-hop pipeline, side by side, on the same question."
        )
        question = gr.Textbox(label="Question")
        ask_button = gr.Button("Ask")
        with gr.Row():
            naive_output = gr.Markdown(label="Naive baseline")
            ragnarok_output = gr.Markdown(label="RAGnarok")
        ask_button.click(
            fn=partial(render_comparison, compare_fn),
            inputs=question,
            outputs=[naive_output, ragnarok_output],
        )
    return view
