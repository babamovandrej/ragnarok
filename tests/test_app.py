import gradio as gr
import pytest

from src.generation.fallback_backend import FallbackBackend
from src.utils.caching import LRUCache


def test_demo_is_a_gradio_blocks_app():
    import app

    assert isinstance(app.demo, gr.Blocks)


class TestMain:
    def test_prewarms_the_demo_repo_before_launching(self, monkeypatch):
        import app

        calls = []
        monkeypatch.setattr(app, "_load_models", lambda cache_key: calls.append(cache_key))
        monkeypatch.setattr(app.demo, "launch", lambda: calls.append("launch"))
        monkeypatch.delenv("HF_DATASET_REPO", raising=False)

        app.main()

        assert calls == [app.DEMO_REPO_SLUG, "launch"]

    def test_refreshes_the_demo_repo_index_before_prewarming(self, monkeypatch):
        import app

        calls = []
        monkeypatch.setenv("HF_DATASET_REPO", "owner/dataset")
        monkeypatch.setattr(
            app, "refresh_if_stale", lambda *args, **kwargs: calls.append("refresh") or False
        )
        monkeypatch.setattr(app, "_load_models", lambda cache_key: calls.append(cache_key))
        monkeypatch.setattr(app.demo, "launch", lambda: calls.append("launch"))

        app.main()

        assert calls == ["refresh", app.DEMO_REPO_SLUG, "launch"]


class TestRefreshDemoRepoIfConfigured:
    def test_skips_when_hf_dataset_repo_is_not_set(self, monkeypatch):
        import app

        monkeypatch.delenv("HF_DATASET_REPO", raising=False)
        monkeypatch.setattr(
            app,
            "refresh_if_stale",
            lambda *a, **k: pytest.fail("refresh_if_stale should not be called"),
        )

        app._refresh_demo_repo_if_configured()

    def test_calls_refresh_if_stale_with_the_demo_repos_cache_dir_when_configured(
        self, monkeypatch
    ):
        import app

        calls = []
        monkeypatch.setenv("HF_DATASET_REPO", "owner/dataset")
        monkeypatch.setattr(app, "refresh_if_stale", lambda *args: calls.append(args) or True)

        app._refresh_demo_repo_if_configured()

        assert calls == [("owner/dataset", app.CACHE_ROOT / app.DEMO_REPO_SLUG, app.DEMO_REPO_SLUG)]

    def test_a_refresh_failure_does_not_propagate(self, monkeypatch, caplog):
        import app

        monkeypatch.setenv("HF_DATASET_REPO", "owner/dataset")

        def _raise(*args, **kwargs):
            raise RuntimeError("network unreachable")

        monkeypatch.setattr(app, "refresh_if_stale", _raise)

        with caplog.at_level("WARNING"):
            app._refresh_demo_repo_if_configured()

        assert "Failed to refresh demo repo index" in caplog.text


class _Sentinel:
    def __init__(self, *args, **kwargs) -> None:
        pass


class TestBuildGenerationBackend:
    def test_wraps_zerogpu_as_primary_and_cpu_as_fallback(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "ZeroGPUBackend", _Sentinel)
        monkeypatch.setattr(app, "CPUBackend", _Sentinel)

        backend = app._build_generation_backend()

        assert isinstance(backend, FallbackBackend)
        assert isinstance(backend._primary, _Sentinel)
        assert isinstance(backend._fallback, _Sentinel)

    def test_falls_back_to_cpu_only_when_zerogpu_cannot_be_constructed(self, monkeypatch):
        import app

        def _raise():
            raise RuntimeError("no CUDA GPUs are available")

        monkeypatch.setattr(app, "ZeroGPUBackend", _raise)
        monkeypatch.setattr(app, "CPUBackend", _Sentinel)

        backend = app._build_generation_backend()

        assert isinstance(backend, _Sentinel)

    def test_logs_a_warning_when_zerogpu_cannot_be_constructed(self, monkeypatch, caplog):
        import app

        def _raise():
            raise RuntimeError("no CUDA GPUs are available")

        monkeypatch.setattr(app, "ZeroGPUBackend", _raise)
        monkeypatch.setattr(app, "CPUBackend", _Sentinel)

        with caplog.at_level("WARNING"):
            app._build_generation_backend()

        assert "ZeroGPU backend unavailable" in caplog.text


class TestLoadModels:
    def test_reuses_cached_models_for_the_same_key(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "Embedder", _Sentinel)
        monkeypatch.setattr(app, "Reranker", _Sentinel)
        monkeypatch.setattr(app, "ZeroGPUBackend", _Sentinel)
        monkeypatch.setattr(app, "CPUBackend", _Sentinel)
        monkeypatch.setattr(app, "_model_cache", LRUCache(capacity=2))

        first = app._load_models("repo/a")
        second = app._load_models("repo/a")

        assert first is second

    def test_loads_separately_per_cache_key(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "Embedder", _Sentinel)
        monkeypatch.setattr(app, "Reranker", _Sentinel)
        monkeypatch.setattr(app, "ZeroGPUBackend", _Sentinel)
        monkeypatch.setattr(app, "CPUBackend", _Sentinel)
        monkeypatch.setattr(app, "_model_cache", LRUCache(capacity=2))

        first = app._load_models("repo/a")
        second = app._load_models("repo/b")

        assert first is not second

    def test_constructs_a_fallback_backend_not_a_bare_cpu_backend(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "Embedder", _Sentinel)
        monkeypatch.setattr(app, "Reranker", _Sentinel)
        monkeypatch.setattr(app, "ZeroGPUBackend", _Sentinel)
        monkeypatch.setattr(app, "CPUBackend", _Sentinel)
        monkeypatch.setattr(app, "_model_cache", LRUCache(capacity=2))

        _, _, backend = app._load_models("repo/a")

        assert isinstance(backend, FallbackBackend)


class TestCompare:
    def test_surfaces_the_backend_used_from_the_model_bundle(self, monkeypatch):
        import app
        from src.baseline.schemas import PipelineResult

        class _FakeBackend:
            last_backend_used = "fallback"

        naive_result = PipelineResult(pipeline="naive", answer="a", citations=[], hop_path=[])
        ragnarok_result = PipelineResult(pipeline="ragnarok", answer="b", citations=[], hop_path=[])

        monkeypatch.setattr(app, "_load_models", lambda cache_key: (None, None, _FakeBackend()))
        monkeypatch.setattr(app, "compare", lambda *args, **kwargs: (naive_result, ragnarok_result))
        monkeypatch.setattr(app, "_repo_context", lambda repo_slug: (repo_slug, "sha1"))

        result = app._compare("owner/repo", "a question")

        assert result == (naive_result, ragnarok_result, "fallback", ("owner/repo", "sha1"))

    def test_returns_none_when_the_backend_does_not_report_last_backend_used(self, monkeypatch):
        import app
        from src.baseline.schemas import PipelineResult

        class _FakeBackend:
            pass  # e.g. a bare CPUBackend, used when ZeroGPU couldn't be constructed

        naive_result = PipelineResult(pipeline="naive", answer="a", citations=[], hop_path=[])
        ragnarok_result = PipelineResult(pipeline="ragnarok", answer="b", citations=[], hop_path=[])

        monkeypatch.setattr(app, "_load_models", lambda cache_key: (None, None, _FakeBackend()))
        monkeypatch.setattr(app, "compare", lambda *args, **kwargs: (naive_result, ragnarok_result))

        _, _, backend_used, _ = app._compare("owner/repo", "a question")

        assert backend_used is None

    def test_loads_models_for_the_given_repo_slug_not_a_fixed_demo_slug(self, monkeypatch):
        import app
        from src.baseline.schemas import PipelineResult

        naive_result = PipelineResult(pipeline="naive", answer="a", citations=[], hop_path=[])
        ragnarok_result = PipelineResult(pipeline="ragnarok", answer="b", citations=[], hop_path=[])
        requested_keys = []

        def _fake_load_models(cache_key):
            requested_keys.append(cache_key)
            return (None, None, object())

        monkeypatch.setattr(app, "_load_models", _fake_load_models)
        monkeypatch.setattr(app, "compare", lambda *args, **kwargs: (naive_result, ragnarok_result))

        app._compare("owner/some-other-repo", "a question")

        assert requested_keys == ["owner/some-other-repo"]


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
        """_ingest itself no longer catches — ingest_view.py's render_ingestion does,
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
        monkeypatch.setattr(app, "_load_models", lambda cache_key: None)

        status, repo_slug = app._ingest("https://github.com/a/b")

        assert "Indexed a/b@abcdef1" in status
        assert "42 chunks" in status
        assert "truncated" not in status.lower()
        assert repo_slug == "a/b"

    def test_reports_a_truncation_warning_when_the_chunk_cap_was_hit(self, monkeypatch):
        import app

        monkeypatch.setattr(app, "ingest_repository", lambda url: self._fake_result(truncated=True))
        monkeypatch.setattr(app, "_load_models", lambda cache_key: None)

        status, _ = app._ingest("https://github.com/a/b")

        assert "truncated" in status.lower()
        assert str(app.MAX_INDEXABLE_CHUNKS) in status or f"{app.MAX_INDEXABLE_CHUNKS:,}" in status

    def test_warms_the_model_cache_for_the_newly_ingested_repo(self, monkeypatch):
        """The demo repo is pre-warmed at startup (task 06); ingesting a second repo should
        populate a second _model_cache entry, actually exercising MAX_RESIDENT_REPOS."""
        import app

        monkeypatch.setattr(
            app, "ingest_repository", lambda url: self._fake_result(truncated=False)
        )
        monkeypatch.setattr(app, "_model_cache", LRUCache(capacity=2))
        monkeypatch.setattr(app, "Embedder", _Sentinel)
        monkeypatch.setattr(app, "Reranker", _Sentinel)
        monkeypatch.setattr(app, "ZeroGPUBackend", _Sentinel)
        monkeypatch.setattr(app, "CPUBackend", _Sentinel)

        app._ingest("https://github.com/a/b")

        assert "a/b" in app._model_cache
