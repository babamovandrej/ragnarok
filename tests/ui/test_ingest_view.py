import gradio as gr
import pytest

from src.ui.ingest_view import build_ingest_view, render_ingestion


class TestRenderIngestion:
    def test_runs_ingest_fn_and_returns_its_result(self):
        result = render_ingestion(lambda url: f"ingested {url}", "https://github.com/a/b")

        assert result == "ingested https://github.com/a/b"

    @pytest.mark.parametrize("blank_url", ["", "   "])
    def test_empty_url_short_circuits_without_calling_ingest_fn(self, blank_url):
        def unreachable_ingest(url: str) -> str:
            raise AssertionError("ingest_fn should not be called for an empty URL")

        result = render_ingestion(unreachable_ingest, blank_url)

        assert "Paste a GitHub repo URL" in result


class TestBuildIngestView:
    def test_returns_a_gradio_blocks_instance(self):
        view = build_ingest_view(lambda url: f"ingested {url}")

        assert isinstance(view, gr.Blocks)
