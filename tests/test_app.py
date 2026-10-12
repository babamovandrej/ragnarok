import gradio as gr
import pytest


def test_demo_is_a_gradio_blocks_app():
    import app

    assert isinstance(app.demo, gr.Blocks)


class _Sentinel:
    def __init__(self, *args, **kwargs) -> None:
        pass


@pytest.fixture
def sentinel_models(monkeypatch):
    import app

    monkeypatch.setattr(app, "Embedder", _Sentinel)
    monkeypatch.setattr(app, "Reranker", _Sentinel)
    monkeypatch.setattr(app, "CPUBackend", _Sentinel)
    monkeypatch.setattr(app, "_models", None)


class TestMain:
    def test_loads_models_before_launching(self, monkeypatch):
        import app

        calls = []
        monkeypatch.setattr(app, "_load_models", lambda: calls.append("load"))
        monkeypatch.setattr(app.demo, "launch", lambda: calls.append("launch"))

        app.main()

        assert calls == ["load", "launch"]


class TestLoadModels:
    def test_loads_embedder_reranker_and_cpu_backend(self, sentinel_models):
        import app

        embedder, reranker, backend = app._load_models()

        assert all(isinstance(model, _Sentinel) for model in (embedder, reranker, backend))

    def test_loads_models_only_once(self, sentinel_models):
        import app

        assert app._load_models() is app._load_models()


class TestCompare:
    def test_returns_both_results_and_the_repo_context(self, monkeypatch):
        import app
        from src.baseline.schemas import PipelineResult

        naive_result = PipelineResult(pipeline="naive", answer="a", citations=[], hop_path=[])
        ragnarok_result = PipelineResult(pipeline="ragnarok", answer="b", citations=[], hop_path=[])

        monkeypatch.setattr(app, "_load_models", lambda: (None, None, None))
        monkeypatch.setattr(app, "compare", lambda *args, **kwargs: (naive_result, ragnarok_result))
        monkeypatch.setattr(app, "_repo_context", lambda repo_slug: (repo_slug, "sha1"))

        result = app._compare("owner/repo", "a question")

        assert result == (naive_result, ragnarok_result, ("owner/repo", "sha1"))

    def test_queries_the_given_repos_cache_dir_not_a_fixed_demo_one(self, monkeypatch):
        import app
        from src.baseline.schemas import PipelineResult

        result = PipelineResult(pipeline="naive", answer="a", citations=[], hop_path=[])
        requested_dirs = []

        def _fake_compare(question, cache_dir, *args, **kwargs):
            requested_dirs.append(cache_dir)
            return result, result

        monkeypatch.setattr(app, "_load_models", lambda: (None, None, None))
        monkeypatch.setattr(app, "compare", _fake_compare)
        monkeypatch.setattr(app, "_repo_context", lambda repo_slug: None)

        app._compare("owner/some-other-repo", "a question")

        assert requested_dirs == [app.CACHE_ROOT / "owner/some-other-repo"]


class TestRepoContext:
    def test_returns_none_when_the_index_has_never_been_built(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "load_manifest", lambda cache_dir: None)

        assert app._repo_context("owner/repo") is None

    def test_returns_slug_and_sha_when_the_index_exists(self, monkeypatch):
        import app

        class _FakeManifest:
            repo_sha = "abc123"

        monkeypatch.setattr(app, "load_manifest", lambda cache_dir: _FakeManifest())

        assert app._repo_context("owner/repo") == ("owner/repo", "abc123")


class TestIngest:
    def test_propagates_ingestion_failures_to_its_caller(self, monkeypatch):
        """_ingest itself doesn't catch — ingest_view.py's render_ingestion does,
        mirroring _compare (no local catch) + render_comparison (catches)."""
        import app

        def _raise(repo_url):
            raise ValueError("bad url")

        monkeypatch.setattr(app, "ingest_repository", _raise)

        with pytest.raises(ValueError, match="bad url"):
            app._ingest("https://github.com/nope/nope")

    @staticmethod
    def _fake_result(*, truncated: bool):
        from pathlib import Path

        from scripts.build_index import IngestionResult
        from src.ingestion.schemas import IngestionStats

        return IngestionResult(
            repo_slug="a/b",
            repo_sha="abcdef1234567",
            cache_dir=Path("/tmp/x"),
            chunk_count=42,
            embedding_dimension=4,
            stats=IngestionStats(truncated=truncated),
            clone_seconds=0.0,
            chunk_seconds=0.0,
            embedding_seconds=0.0,
            index_seconds=0.0,
            total_seconds=0.0,
        )

    def test_reports_success_with_repo_slug_sha_and_chunk_count(self, monkeypatch):
        import app

        monkeypatch.setattr(
            app, "ingest_repository", lambda url: self._fake_result(truncated=False)
        )

        status, repo_slug = app._ingest("https://github.com/a/b")

        assert "Indexed a/b@abcdef1" in status
        assert "42 chunks" in status
        assert "truncated" not in status.lower()
        assert repo_slug == "a/b"

    def test_reports_a_truncation_warning_when_the_chunk_cap_was_hit(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "ingest_repository", lambda url: self._fake_result(truncated=True))

        status, _ = app._ingest("https://github.com/a/b")

        assert "truncated" in status.lower()
        assert f"{app.MAX_INDEXABLE_CHUNKS:,}" in status
