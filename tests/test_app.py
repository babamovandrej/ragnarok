import gradio as gr

from src.utils.caching import LRUCache


def test_demo_is_a_gradio_blocks_app():
    import app

    assert isinstance(app.demo, gr.Blocks)


class _Sentinel:
    pass


class TestLoadModels:
    def test_reuses_cached_models_for_the_same_key(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "Embedder", _Sentinel)
        monkeypatch.setattr(app, "Reranker", _Sentinel)
        monkeypatch.setattr(app, "CPUBackend", _Sentinel)
        monkeypatch.setattr(app, "_model_cache", LRUCache(capacity=2))

        first = app._load_models("repo/a")
        second = app._load_models("repo/a")

        assert first is second

    def test_loads_separately_per_cache_key(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "Embedder", _Sentinel)
        monkeypatch.setattr(app, "Reranker", _Sentinel)
        monkeypatch.setattr(app, "CPUBackend", _Sentinel)
        monkeypatch.setattr(app, "_model_cache", LRUCache(capacity=2))

        first = app._load_models("repo/a")
        second = app._load_models("repo/b")

        assert first is not second


class TestDemoRepoContext:
    def test_returns_none_when_the_demo_index_has_never_been_built(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "load_manifest", lambda cache_dir: None)

        assert app._demo_repo_context() is None

    def test_returns_slug_and_sha_when_the_demo_index_exists(self, monkeypatch):
        import app

        class _FakeManifest:
            repo_sha = "abc123"

        monkeypatch.setattr(app, "load_manifest", lambda cache_dir: _FakeManifest())

        assert app._demo_repo_context() == (app.DEMO_REPO_SLUG, "abc123")


class TestIngest:
    def test_reports_a_failure_as_a_status_message(self, monkeypatch):
        import app

        def _raise(repo_url):
            raise ValueError("bad url")

        monkeypatch.setattr(app, "ingest_repository", _raise)

        result = app._ingest("https://github.com/nope/nope")

        assert "Ingestion failed" in result
        assert "bad url" in result

    def test_logs_the_exception_before_returning_the_status_message(self, monkeypatch, caplog):
        import app

        def _raise(repo_url):
            raise ValueError("bad url")

        monkeypatch.setattr(app, "ingest_repository", _raise)

        with caplog.at_level("ERROR"):
            app._ingest("https://github.com/nope/nope")

        assert "Ingestion failed" in caplog.text
        assert "https://github.com/nope/nope" in caplog.text
