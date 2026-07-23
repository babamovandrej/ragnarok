import gradio as gr

from src.ui.ingest_view import build_ingest_view


class TestBuildIngestView:
    def test_returns_a_gradio_blocks_instance(self):
        view = build_ingest_view(lambda url: f"ingested {url}")

        assert isinstance(view, gr.Blocks)
