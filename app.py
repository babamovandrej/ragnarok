import gradio as gr

from ragnarok import __version__

with gr.Blocks(title="RAGnarok") as demo:
    gr.Markdown(
        f"""
        # RAGnarok (v{__version__})

        A RAG assistant for understanding codebases. Paste a public GitHub
        repo URL, ask a question, get a cited, grounded answer.

        _Still under construction — ingestion, retrieval, and generation
        land in later phases. See `docs/roadmap.md`._
        """
    )

if __name__ == "__main__":
    demo.launch()
