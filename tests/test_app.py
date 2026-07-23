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


class TestIngest:
    def test_reports_a_failure_as_a_status_message(self, monkeypatch):
        import app

        def _raise(repo_url):
            raise ValueError("bad url")

        monkeypatch.setattr(app, "ingest_repository", _raise)

        result = app._ingest("https://github.com/nope/nope")

        assert "Ingestion failed" in result
        assert "bad url" in result
