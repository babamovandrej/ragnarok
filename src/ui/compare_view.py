from collections.abc import Callable
from functools import partial

import gradio as gr

from src.baseline.schemas import PipelineResult
from src.generation.citations import Citation
from src.ui.hop_viz import render_hop_path

CompareFn = Callable[[str], tuple[PipelineResult, PipelineResult]]
RepoContext = tuple[str, str]

_EMPTY_QUESTION_MESSAGE = "_Enter a question above, then press Ask._"
_PROGRESS_MESSAGE = "Running naive baseline and RAGnarok pipeline (CPU inference can take a while)…"


def render_comparison(
    compare_fn: CompareFn,
    question: str,
    *,
    repo_context: RepoContext | None = None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str]:
    """Run compare_fn and render both pipeline results as markdown for the two UI columns.

    Backend failures are caught and rendered as an inline error message in both columns
    rather than surfacing Gradio's generic error toast.
    """
    if not question.strip():
        return _EMPTY_QUESTION_MESSAGE, _EMPTY_QUESTION_MESSAGE

    progress(0, desc=_PROGRESS_MESSAGE)
    try:
        naive_result, ragnarok_result = compare_fn(question)
    except Exception as exc:
        error_message = (
            f"**Something went wrong answering this question.**\n\n`{exc}`\n\n"
            "Try again, or ask a different question."
        )
        return error_message, error_message
    progress(1, desc="Done")

    return _format_naive(naive_result), _format_ragnarok(ragnarok_result, repo_context)


def _format_naive(result: PipelineResult) -> str:
    """Render the naive-baseline column: just the raw answer, no citations or hops."""
    return result.answer


def _format_ragnarok(result: PipelineResult, repo_context: RepoContext | None) -> str:
    """Render the RAGnarok column: answer, per-citation verification, and hop path."""
    sections = [result.answer, _render_citations(result.citations, repo_context)]
    if repo_context is not None:
        repo_slug, repo_sha = repo_context
        sections.append(render_hop_path(result.hop_path, repo_slug, repo_sha))
    elif result.hop_path:
        hops = ", ".join(hop.symbol_name for hop in result.hop_path)
        sections.append(f"**Hop path:** {hops}")
    return "\n\n".join(sections)


def _render_citations(citations: list[Citation], repo_context: RepoContext | None) -> str:
    """Render each citation's file#Lstart-Lend span with its own verification status."""
    if not citations:
        return "_No citations found_"
    lines = ["**Citations:**"]
    for citation in citations:
        span = f"{citation.file_path}#L{citation.start_line}-L{citation.end_line}"
        badge = "verified" if citation.verified else "UNVERIFIED"
        if repo_context is not None:
            repo_slug, repo_sha = repo_context
            url = f"https://github.com/{repo_slug}/blob/{repo_sha}/{span}"
            lines.append(f"- [`{span}`]({url}) — {badge}")
        else:
            lines.append(f"- `{span}` — {badge}")
    return "\n".join(lines)


def build_compare_view(
    compare_fn: CompareFn, *, repo_context: RepoContext | None = None
) -> gr.Blocks:
    """Build a two-column Gradio Blocks UI comparing the naive and RAGnarok pipelines."""
    view: gr.Blocks
    with gr.Blocks(title="RAGnarok — naive vs. hybrid+rerank+agent") as view:
        gr.Markdown(
            "# RAGnarok\n\nCompare a naive single-vector baseline against the full "
            "hybrid+rerank+multi-hop pipeline, side by side, on the same question."
        )
        question = gr.Textbox(
            label="Question", placeholder="e.g. How does session authentication work?"
        )
        ask_button = gr.Button("Ask", variant="primary")
        with gr.Row():
            naive_output = gr.Markdown(label="Naive baseline")
            ragnarok_output = gr.Markdown(label="RAGnarok")
        ask_button.click(
            fn=partial(render_comparison, compare_fn, repo_context=repo_context),
            inputs=question,
            outputs=[naive_output, ragnarok_output],
        )
    return view
