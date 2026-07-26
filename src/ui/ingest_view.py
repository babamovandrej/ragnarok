import logging
from collections.abc import Callable
from functools import partial

import gradio as gr

logger = logging.getLogger(__name__)

IngestFn = Callable[[str], tuple[str, str]]

_EMPTY_URL_MESSAGE = "_Paste a GitHub repo URL above, then press Ingest._"
_PROGRESS_MESSAGE = "Cloning, chunking, embedding, and indexing the repo — this can take a while…"


def render_ingestion(
    ingest_fn: IngestFn,
    repo_url: str,
    current_repo_slug: str,
    *,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str]:
    """Run ingest_fn for repo_url; returns (status_message, active_repo_slug).

    On success, active_repo_slug becomes the newly ingested repo, so the Compare tab targets
    it. On an empty URL or a failure, active_repo_slug is echoed back unchanged. Backend
    failures are caught here (mirroring compare_view.py's render_comparison) rather than
    relying on ingest_fn's own implementation to stay safe.
    """
    if not repo_url.strip():
        return _EMPTY_URL_MESSAGE, current_repo_slug
    progress(0, desc=_PROGRESS_MESSAGE)
    try:
        status, repo_slug = ingest_fn(repo_url)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", repo_url)
        return f"**Ingestion failed.**\n\n`{exc}`", current_repo_slug
    progress(1, desc="Done")
    return status, repo_slug


def build_ingest_view(ingest_fn: IngestFn, active_repo_state: gr.State) -> gr.Blocks:
    """Build a Gradio Blocks UI: paste a public repo URL, ingest it, see a status message.

    active_repo_state is shared with the Compare tab: a successful ingest here becomes the
    repo Compare targets next, instead of leaving that link implicit.
    """
    view: gr.Blocks
    with gr.Blocks(title="RAGnarok — ingest a repo") as view:
        gr.Markdown(
            "# Ingest a repository\n\nPaste a public GitHub repo URL to index it before "
            "asking questions about it."
        )
        url = gr.Textbox(label="GitHub repo URL", placeholder="https://github.com/owner/repo")
        ingest_button = gr.Button("Ingest", variant="primary")
        status = gr.Markdown()
        ingest_button.click(
            fn=partial(render_ingestion, ingest_fn),
            inputs=[url, active_repo_state],
            outputs=[status, active_repo_state],
        )
    return view
