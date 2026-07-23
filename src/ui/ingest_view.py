from collections.abc import Callable
from functools import partial

import gradio as gr

IngestFn = Callable[[str], str]

_EMPTY_URL_MESSAGE = "_Paste a GitHub repo URL above, then press Ingest._"
_PROGRESS_MESSAGE = "Cloning, chunking, embedding, and indexing the repo — this can take a while…"


def render_ingestion(
    ingest_fn: IngestFn,
    repo_url: str,
    *,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> str:
    """Run ingest_fn for repo_url, guarding against an empty URL and reporting progress."""
    if not repo_url.strip():
        return _EMPTY_URL_MESSAGE
    progress(0, desc=_PROGRESS_MESSAGE)
    result = ingest_fn(repo_url)
    progress(1, desc="Done")
    return result


def build_ingest_view(ingest_fn: IngestFn) -> gr.Blocks:
    """Build a Gradio Blocks UI: paste a public repo URL, ingest it, see a status message."""
    view: gr.Blocks
    with gr.Blocks(title="RAGnarok — ingest a repo") as view:
        gr.Markdown(
            "# Ingest a repository\n\nPaste a public GitHub repo URL to index it before "
            "asking questions about it."
        )
        url = gr.Textbox(label="GitHub repo URL", placeholder="https://github.com/owner/repo")
        ingest_button = gr.Button("Ingest", variant="primary")
        status = gr.Markdown()
        ingest_button.click(fn=partial(render_ingestion, ingest_fn), inputs=url, outputs=status)
    return view
