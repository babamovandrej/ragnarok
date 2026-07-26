import gradio as gr
import pytest

from src.ui.ingest_view import build_ingest_view, render_ingestion


class TestRenderIngestion:
    def test_runs_ingest_fn_and_returns_its_status_and_new_active_repo(self):
        status, repo_slug = render_ingestion(
            lambda url: (f"ingested {url}", "owner/new-repo"), "https://github.com/a/b", "owner/old"
        )

        assert status == "ingested https://github.com/a/b"
        assert repo_slug == "owner/new-repo"

    @pytest.mark.parametrize("blank_url", ["", "   "])
    def test_empty_url_short_circuits_without_calling_ingest_fn(self, blank_url):
        def unreachable_ingest(url: str) -> tuple[str, str]:
            raise AssertionError("ingest_fn should not be called for an empty URL")

        status, repo_slug = render_ingestion(unreachable_ingest, blank_url, "owner/old")

        assert "Paste a GitHub repo URL" in status
        assert repo_slug == "owner/old"

    def test_ingest_fn_failure_renders_a_friendly_inline_message_and_keeps_the_active_repo(self):
        def failing_ingest(url: str) -> tuple[str, str]:
            raise RuntimeError("clone failed")

        status, repo_slug = render_ingestion(failing_ingest, "https://github.com/a/b", "owner/old")

        assert "Ingestion failed" in status
        assert "clone failed" in status
        assert repo_slug == "owner/old"

    def test_ingest_fn_failure_is_logged(self, caplog):
        def failing_ingest(url: str) -> tuple[str, str]:
            raise RuntimeError("clone failed")

        with caplog.at_level("ERROR"):
            render_ingestion(failing_ingest, "https://github.com/a/b", "owner/old")

        assert "Ingestion failed" in caplog.text
        assert "https://github.com/a/b" in caplog.text


class TestBuildIngestView:
    def test_returns_a_gradio_blocks_instance(self):
        view = build_ingest_view(
            lambda url: (f"ingested {url}", "owner/repo"), gr.State(value="owner/repo")
        )

        assert isinstance(view, gr.Blocks)
