from collections.abc import Callable

import gradio as gr

IngestFn = Callable[[str], str]


def build_ingest_view(ingest_fn: IngestFn) -> gr.Blocks:
    """Build a Gradio Blocks UI: paste a public repo URL, ingest it, see a status message."""
    view: gr.Blocks
    with gr.Blocks(title="RAGnarok — ingest a repo") as view:
        gr.Markdown(
            "# Ingest a repository\n\nPaste a public GitHub repo URL to index it before "
            "asking questions about it."
        )
        url = gr.Textbox(label="GitHub repo URL", placeholder="https://github.com/owner/repo")
        ingest_button = gr.Button("Ingest")
        status = gr.Markdown()
        ingest_button.click(fn=ingest_fn, inputs=url, outputs=status)
    return view
