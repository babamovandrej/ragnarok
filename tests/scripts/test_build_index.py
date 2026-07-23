import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scripts.build_index import (
    EXIT_CLONE_FAILED,
    EXIT_INCOMPATIBLE_CACHE,
    EXIT_INDEX_BUILD_FAILED,
    EXIT_INTERRUPTED,
    EXIT_INVALID_ARGUMENT,
    EXIT_NO_CHUNKS_PRODUCED,
    EXIT_SUCCESS,
    EXIT_UNEXPECTED_ERROR,
    INDEX_SCHEMA_VERSION,
    EmbeddingFailedError,
    IncompatibleCacheError,
    IngestionResult,
    NoChunksProducedError,
    _ensure_disjoint,
    _merge_incremental_chunks,
    _positive_int,
    _slug_from_local_path,
    _validate_cache_dir,
    build_parser,
    ingest_repository,
    main,
    render_result,
)
from src.ingestion.chunker import collect_chunks
from src.ingestion.clone import GitCommandError, current_sha
from src.ingestion.indexer import build_index
from src.ingestion.schemas import IngestionStats


class _FakeEmbedder:
    def __init__(self, device=None, batch_size=None):
        self.device = device
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.random.default_rng(0).random((len(texts), 4), dtype=np.float32)


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setattr("scripts.build_index.Embedder", _FakeEmbedder)


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _init_local_repo(path: Path, *, with_source: bool = True) -> Path:
    path.mkdir()
    _run("git", "init", "-q", cwd=path)
    _run("git", "config", "user.email", "test@example.com", cwd=path)
    _run("git", "config", "user.name", "Test", cwd=path)
    if with_source:
        (path / "a.py").write_text("def foo():\n    return 1\n")
    else:
        (path / "README.md").write_text("nothing chunkable here\n")
    _run("git", "add", "-A", cwd=path)
    _run("git", "commit", "-q", "-m", "first", cwd=path)
    return path


@dataclass
class _ShallowCloneCall:
    repo_url: str
    dest: Path
    revision: str | None


class _FakeShallowClone:
    """Stands in for clone.shallow_clone: copies a local fixture instead of hitting the network."""

    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls: list[_ShallowCloneCall] = []

    def __call__(
        self,
        repo_url: str,
        dest: Path,
        timeout: int | None = None,
        revision: str | None = None,
    ) -> Path:
        self.calls.append(_ShallowCloneCall(repo_url=repo_url, dest=dest, revision=revision))
        shutil.copytree(self.source, dest)
        return dest


class TestBuildParser:
    def test_requires_repo_url_or_repo_path(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_rejects_both_repo_url_and_repo_path(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["https://github.com/a/b", "--repo-path", "/tmp/x"])

    def test_accepts_repo_url_alone(self):
        args = build_parser().parse_args(["https://github.com/a/b"])
        assert args.repo_url == "https://github.com/a/b"
        assert args.repo_path is None

    def test_cache_dir_and_revision_parse_as_expected(self):
        args = build_parser().parse_args(
            ["https://github.com/a/b", "--cache-dir", "/tmp/out", "--revision", "v1.0"]
        )
        assert args.cache_dir == Path("/tmp/out")
        assert args.revision == "v1.0"

    def test_rejects_invalid_device_choice(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["https://github.com/a/b", "--device", "tpu"])

    def test_rejects_mutually_exclusive_rebuild_and_incremental(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["https://github.com/a/b", "--rebuild", "--incremental"])

    def test_rejects_non_positive_embedding_batch_size(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["https://github.com/a/b", "--embedding-batch-size", "0"])

    def test_rejects_invalid_log_level(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["https://github.com/a/b", "--log-level", "TRACE"])

    def test_dry_run_and_json_flags_parse(self):
        args = build_parser().parse_args(["https://github.com/a/b", "--dry-run", "--json"])
        assert args.dry_run is True
        assert args.json is True


class TestIngestRepository:
    def test_requires_exactly_one_of_repo_url_or_repo_path(self):
        with pytest.raises(ValueError):
            ingest_repository()

    def test_rejects_both_repo_url_and_repo_path(self, tmp_path):
        with pytest.raises(ValueError):
            ingest_repository("https://github.com/a/b", repo_path=tmp_path)

    def test_incremental_without_repo_path_is_rejected(self):
        with pytest.raises(ValueError, match="repo-path"):
            ingest_repository("https://github.com/a/b", incremental=True)

    def test_revision_with_repo_path_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="revision"):
            ingest_repository(repo_path=tmp_path, revision="v1.0")

    def test_keep_clone_with_repo_path_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="keep-clone"):
            ingest_repository(repo_path=tmp_path, keep_clone=True)

    def test_successful_local_ingestion_reports_chunk_count_and_dimension(
        self, tmp_path, fake_embedder
    ):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"

        result = ingest_repository(repo_path=repo, cache_dir=cache_dir)

        assert result.repo_slug == "local/repo"
        assert result.chunk_count == 1
        assert result.embedding_dimension == 4
        assert not result.skipped
        assert not result.dry_run

    def test_dry_run_skips_embedding_and_indexing(self, tmp_path, monkeypatch):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"

        def _boom(*args, **kwargs):
            raise AssertionError("Embedder should not be constructed on a dry run")

        monkeypatch.setattr("scripts.build_index.Embedder", _boom)

        result = ingest_repository(repo_path=repo, cache_dir=cache_dir, dry_run=True)

        assert result.dry_run
        assert result.chunk_count == 1
        assert not cache_dir.exists()

    def test_second_run_skips_when_cache_is_compatible(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"

        ingest_repository(repo_path=repo, cache_dir=cache_dir)
        second = ingest_repository(repo_path=repo, cache_dir=cache_dir)

        assert second.skipped

    def test_rebuild_forces_a_fresh_build_even_if_compatible(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"

        ingest_repository(repo_path=repo, cache_dir=cache_dir)
        second = ingest_repository(repo_path=repo, cache_dir=cache_dir, rebuild=True)

        assert not second.skipped

    def test_incompatible_cache_without_rebuild_raises(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"

        ingest_repository(repo_path=repo, cache_dir=cache_dir)
        (repo / "a.py").write_text("def foo():\n    return 2\n")
        _run("git", "commit", "-q", "-am", "second", cwd=repo)

        with pytest.raises(IncompatibleCacheError):
            ingest_repository(repo_path=repo, cache_dir=cache_dir)

    def test_no_chunks_produced_raises_with_stats_context(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo", with_source=False)
        cache_dir = tmp_path / "cache"

        with pytest.raises(NoChunksProducedError, match="files_scanned"):
            ingest_repository(repo_path=repo, cache_dir=cache_dir)

    def test_cache_dir_overlapping_repo_path_is_rejected(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo")

        with pytest.raises(ValueError, match="overlap"):
            ingest_repository(repo_path=repo, cache_dir=repo)

    def test_unsafe_cache_dir_is_rejected(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo")

        with pytest.raises(ValueError, match="unsafe"):
            ingest_repository(repo_path=repo, cache_dir=Path("/"))

    def test_local_path_that_is_not_a_git_repo_raises_git_command_error(self, tmp_path):
        not_a_repo = tmp_path / "not_a_repo"
        not_a_repo.mkdir()

        with pytest.raises(GitCommandError):
            ingest_repository(repo_path=not_a_repo, cache_dir=tmp_path / "cache")


class TestIncrementalIngestion:
    def test_requires_an_existing_index(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo")

        with pytest.raises(IncompatibleCacheError, match="run a full build first"):
            ingest_repository(repo_path=repo, cache_dir=tmp_path / "cache", incremental=True)

    def test_skips_when_sha_is_unchanged(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"
        ingest_repository(repo_path=repo, cache_dir=cache_dir)

        second = ingest_repository(repo_path=repo, cache_dir=cache_dir, incremental=True)

        assert second.skipped

    def test_reindexes_only_changed_and_added_files(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"
        first = ingest_repository(repo_path=repo, cache_dir=cache_dir)
        assert first.chunk_count == 1

        (repo / "a.py").write_text("def foo():\n    return 2\n")
        (repo / "b.py").write_text("def bar():\n    return 3\n")
        _run("git", "add", "-A", cwd=repo)
        _run("git", "commit", "-q", "-m", "second", cwd=repo)
        new_sha = current_sha(repo)

        second = ingest_repository(repo_path=repo, cache_dir=cache_dir, incremental=True)

        assert not second.skipped
        assert second.repo_sha == new_sha
        assert second.chunk_count == 2


class TestMergeIncrementalChunks:
    def test_reuses_unchanged_files_embeddings_and_reembeds_only_changed_ones(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _run("git", "init", "-q", cwd=repo)
        _run("git", "config", "user.email", "test@example.com", cwd=repo)
        _run("git", "config", "user.name", "Test", cwd=repo)
        (repo / "a.py").write_text("def foo():\n    return 1\n")
        (repo / "b.py").write_text("def bar():\n    return 2\n")
        _run("git", "add", "-A", cwd=repo)
        _run("git", "commit", "-q", "-m", "first", cwd=repo)
        first_sha = current_sha(repo)

        initial_chunks, _ = collect_chunks(repo)
        initial_embeddings = np.array(
            [[float(i), 0.0, 0.0, 0.0] for i in range(len(initial_chunks))], dtype=np.float32
        )
        cache_dir = tmp_path / "cache"
        build_index(
            initial_chunks, initial_embeddings, cache_dir, repo_slug="a/b", repo_sha=first_sha
        )
        original_b_embedding = next(
            embedding
            for chunk, embedding in zip(initial_chunks, initial_embeddings)
            if chunk.file_path == "b.py"
        )

        (repo / "a.py").write_text("def foo():\n    return 999\n")
        _run("git", "add", "-A", cwd=repo)
        _run("git", "commit", "-q", "-m", "second", cwd=repo)

        class _TrackingEmbedder:
            def __init__(self) -> None:
                self.embed_calls: list[list[str]] = []

            def embed(self, texts: list[str]) -> np.ndarray:
                self.embed_calls.append(list(texts))
                return np.array([[100.0, 100.0, 100.0, 100.0] for _ in texts], dtype=np.float32)

        embedder = _TrackingEmbedder()
        merged_chunks, merged_embeddings, stats = _merge_incremental_chunks(
            cache_dir, repo, first_sha, embedder
        )

        merged_by_file = dict(zip((c.file_path for c in merged_chunks), merged_embeddings))
        assert set(merged_by_file) == {"a.py", "b.py"}
        assert np.array_equal(merged_by_file["b.py"], original_b_embedding)
        assert not np.array_equal(merged_by_file["a.py"], original_b_embedding)
        assert stats.chunks_created == 1
        all_embedded_texts = [text for call in embedder.embed_calls for text in call]
        assert not any("bar" in text for text in all_embedded_texts)
        assert any("999" in text for text in all_embedded_texts)

    def test_deleted_file_is_dropped_entirely(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _run("git", "init", "-q", cwd=repo)
        _run("git", "config", "user.email", "test@example.com", cwd=repo)
        _run("git", "config", "user.name", "Test", cwd=repo)
        (repo / "a.py").write_text("def foo():\n    return 1\n")
        (repo / "b.py").write_text("def bar():\n    return 2\n")
        _run("git", "add", "-A", cwd=repo)
        _run("git", "commit", "-q", "-m", "first", cwd=repo)
        first_sha = current_sha(repo)

        initial_chunks, _ = collect_chunks(repo)
        initial_embeddings = np.array(
            [[float(i), 0.0, 0.0, 0.0] for i in range(len(initial_chunks))], dtype=np.float32
        )
        cache_dir = tmp_path / "cache"
        build_index(
            initial_chunks, initial_embeddings, cache_dir, repo_slug="a/b", repo_sha=first_sha
        )

        (repo / "b.py").unlink()
        _run("git", "add", "-A", cwd=repo)
        _run("git", "commit", "-q", "-m", "delete b", cwd=repo)

        merged_chunks, _, stats = _merge_incremental_chunks(
            cache_dir, repo, first_sha, _FakeEmbedder()
        )

        assert {c.file_path for c in merged_chunks} == {"a.py"}
        assert stats.chunks_created == 0


class TestIngestRepositoryWithUrl:
    def test_uses_temporary_workspace_and_cleans_it_up_on_success(
        self, tmp_path, fake_embedder, monkeypatch
    ):
        source = _init_local_repo(tmp_path / "source")
        fake_clone = _FakeShallowClone(source)
        monkeypatch.setattr("scripts.build_index.shallow_clone", fake_clone)

        result = ingest_repository("https://github.com/example/repo", cache_dir=tmp_path / "cache")

        assert result.repo_slug == "example/repo"
        assert result.chunk_count == 1
        assert not fake_clone.calls[0].dest.exists()

    def test_workspace_is_cleaned_up_even_if_embedding_fails(self, tmp_path, monkeypatch):
        source = _init_local_repo(tmp_path / "source")
        fake_clone = _FakeShallowClone(source)
        monkeypatch.setattr("scripts.build_index.shallow_clone", fake_clone)

        def _boom(*args, **kwargs):
            raise RuntimeError("embedding exploded")

        monkeypatch.setattr("scripts.build_index.Embedder", _boom)

        with pytest.raises(EmbeddingFailedError):
            ingest_repository("https://github.com/example/repo", cache_dir=tmp_path / "cache")

        assert not fake_clone.calls[0].dest.exists()

    def test_keep_clone_persists_the_workspace_under_cache_root_clones(
        self, tmp_path, fake_embedder, monkeypatch
    ):
        source = _init_local_repo(tmp_path / "source")
        fake_clone = _FakeShallowClone(source)
        monkeypatch.setattr("scripts.build_index.shallow_clone", fake_clone)
        monkeypatch.setattr("scripts.build_index.CACHE_ROOT", tmp_path / "cache_root")

        ingest_repository(
            "https://github.com/example/repo", cache_dir=tmp_path / "cache", keep_clone=True
        )

        clone_dest = fake_clone.calls[0].dest
        assert clone_dest.exists()
        assert clone_dest == tmp_path / "cache_root" / "clones" / "example/repo" / "repo"
        assert (clone_dest / "a.py").exists()

    def test_revision_is_passed_through_to_shallow_clone(
        self, tmp_path, fake_embedder, monkeypatch
    ):
        source = _init_local_repo(tmp_path / "source")
        fake_clone = _FakeShallowClone(source)
        monkeypatch.setattr("scripts.build_index.shallow_clone", fake_clone)

        ingest_repository(
            "https://github.com/example/repo",
            cache_dir=tmp_path / "cache",
            revision="v1.0",
        )

        assert fake_clone.calls[0].revision == "v1.0"

    def test_device_and_batch_size_are_passed_through_to_embedder(self, tmp_path, monkeypatch):
        source = _init_local_repo(tmp_path / "source")
        fake_clone = _FakeShallowClone(source)
        monkeypatch.setattr("scripts.build_index.shallow_clone", fake_clone)

        captured: dict[str, object] = {}

        class _CapturingEmbedder:
            def __init__(self, device=None, batch_size=None):
                captured["device"] = device
                captured["batch_size"] = batch_size

            def embed(self, texts: list[str]) -> np.ndarray:
                return np.random.default_rng(0).random((len(texts), 4), dtype=np.float32)

        monkeypatch.setattr("scripts.build_index.Embedder", _CapturingEmbedder)

        ingest_repository(
            "https://github.com/example/repo",
            cache_dir=tmp_path / "cache",
            device="cpu",
            embedding_batch_size=16,
        )

        assert captured == {"device": "cpu", "batch_size": 16}


class TestMainExitCodes:
    def test_invalid_url_returns_invalid_argument_exit_code(self):
        assert main(["not-a-github-url"]) == EXIT_INVALID_ARGUMENT

    def test_missing_repo_argument_is_a_system_exit(self):
        with pytest.raises(SystemExit):
            main([])

    def test_clone_of_nonexistent_local_git_repo_returns_clone_failed(self, tmp_path):
        not_a_repo = tmp_path / "not_a_repo"
        not_a_repo.mkdir()

        assert main(["--repo-path", str(not_a_repo)]) == EXIT_CLONE_FAILED

    def test_no_chunks_returns_no_chunks_exit_code(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo", with_source=False)

        assert main(["--repo-path", str(repo), "--cache-dir", str(tmp_path / "cache")]) == (
            EXIT_NO_CHUNKS_PRODUCED
        )

    def test_incompatible_cache_returns_its_own_exit_code(self, tmp_path, fake_embedder):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"
        assert main(["--repo-path", str(repo), "--cache-dir", str(cache_dir)]) == EXIT_SUCCESS

        (repo / "a.py").write_text("def foo():\n    return 2\n")
        _run("git", "commit", "-q", "-am", "second", cwd=repo)

        assert main(["--repo-path", str(repo), "--cache-dir", str(cache_dir)]) == (
            EXIT_INCOMPATIBLE_CACHE
        )

    def test_keyboard_interrupt_returns_interrupted_exit_code(self, monkeypatch):
        def _raise_interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr("scripts.build_index.ingest_repository", _raise_interrupt)

        assert main(["https://github.com/a/b"]) == EXIT_INTERRUPTED

    def test_successful_run_prints_human_summary(self, tmp_path, fake_embedder, capsys):
        repo = _init_local_repo(tmp_path / "repo")

        exit_code = main(["--repo-path", str(repo), "--cache-dir", str(tmp_path / "cache")])

        assert exit_code == EXIT_SUCCESS
        out = capsys.readouterr().out
        assert "Indexed local/repo@" in out
        assert "Chunks:" in out

    def test_successful_run_with_json_flag_prints_valid_json_to_stdout(
        self, tmp_path, fake_embedder, capsys
    ):
        repo = _init_local_repo(tmp_path / "repo")

        exit_code = main(
            ["--repo-path", str(repo), "--cache-dir", str(tmp_path / "cache"), "--json"]
        )

        assert exit_code == EXIT_SUCCESS
        payload = json.loads(capsys.readouterr().out)
        assert payload["repo_slug"] == "local/repo"
        assert payload["chunk_count"] == 1

    def test_embedding_failure_returns_index_build_failed_exit_code(self, tmp_path, monkeypatch):
        repo = _init_local_repo(tmp_path / "repo")

        def _boom(*args, **kwargs):
            raise RuntimeError("embedding exploded")

        monkeypatch.setattr("scripts.build_index.Embedder", _boom)

        exit_code = main(["--repo-path", str(repo), "--cache-dir", str(tmp_path / "cache")])

        assert exit_code == EXIT_INDEX_BUILD_FAILED

    def test_index_build_failure_returns_index_build_failed_exit_code(
        self, tmp_path, fake_embedder, monkeypatch
    ):
        repo = _init_local_repo(tmp_path / "repo")

        def _boom(*args, **kwargs):
            raise RuntimeError("chroma exploded")

        monkeypatch.setattr("scripts.build_index.build_index", _boom)

        exit_code = main(["--repo-path", str(repo), "--cache-dir", str(tmp_path / "cache")])

        assert exit_code == EXIT_INDEX_BUILD_FAILED

    def test_unexpected_exception_returns_unexpected_error_exit_code(self, tmp_path, monkeypatch):
        def _boom(*args, **kwargs):
            raise TypeError("something truly unexpected")

        monkeypatch.setattr("scripts.build_index.ingest_repository", _boom)

        assert main(["https://github.com/a/b"]) == EXIT_UNEXPECTED_ERROR

    def test_json_output_for_a_skipped_run_reports_skipped_true(
        self, tmp_path, fake_embedder, capsys
    ):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"
        main(["--repo-path", str(repo), "--cache-dir", str(cache_dir)])
        capsys.readouterr()

        exit_code = main(["--repo-path", str(repo), "--cache-dir", str(cache_dir), "--json"])

        assert exit_code == EXIT_SUCCESS
        payload = json.loads(capsys.readouterr().out)
        assert payload["skipped"] is True

    def test_dry_run_via_cli_does_not_create_cache_dir(self, tmp_path, capsys):
        repo = _init_local_repo(tmp_path / "repo")
        cache_dir = tmp_path / "cache"

        exit_code = main(
            ["--repo-path", str(repo), "--cache-dir", str(cache_dir), "--dry-run", "--json"]
        )

        assert exit_code == EXIT_SUCCESS
        payload = json.loads(capsys.readouterr().out)
        assert payload["dry_run"] is True
        assert not cache_dir.exists()

    def test_version_flag_reports_schema_version_and_exits_cleanly(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])

        assert exc_info.value.code == 0
        assert str(INDEX_SCHEMA_VERSION) in capsys.readouterr().out


def _make_result(
    *,
    repo_slug: str = "a/b",
    repo_sha: str = "0123456789abcdef",
    cache_dir: Path = Path("/tmp/cache"),
    chunk_count: int = 3,
    embedding_dimension: int = 4,
    stats: IngestionStats | None = None,
    clone_seconds: float = 0.1,
    chunk_seconds: float = 0.1,
    embedding_seconds: float = 1.0,
    index_seconds: float = 0.5,
    total_seconds: float = 1.7,
    skipped: bool = False,
    dry_run: bool = False,
) -> IngestionResult:
    return IngestionResult(
        repo_slug=repo_slug,
        repo_sha=repo_sha,
        cache_dir=cache_dir,
        chunk_count=chunk_count,
        embedding_dimension=embedding_dimension,
        stats=stats
        if stats is not None
        else IngestionStats(files_scanned=5, files_chunked=3, files_skipped=2),
        clone_seconds=clone_seconds,
        chunk_seconds=chunk_seconds,
        embedding_seconds=embedding_seconds,
        index_seconds=index_seconds,
        total_seconds=total_seconds,
        skipped=skipped,
        dry_run=dry_run,
    )


class TestRenderResult:
    def test_dry_run_status_is_labeled_dry_run_not_indexed(self, capsys):
        render_result(_make_result(dry_run=True, embedding_dimension=0), as_json=False)

        assert "Dry run" in capsys.readouterr().out

    def test_skipped_status_is_labeled_up_to_date(self, capsys):
        render_result(_make_result(skipped=True), as_json=False)

        assert "Skipped (already up to date)" in capsys.readouterr().out

    def test_successful_status_is_labeled_indexed(self, capsys):
        render_result(_make_result(), as_json=False)

        assert "Indexed a/b@0123456" in capsys.readouterr().out

    def test_human_summary_includes_all_stats_fields(self, capsys):
        render_result(_make_result(), as_json=False)

        out = capsys.readouterr().out
        assert "Chunks:              3" in out
        assert "Embedding dimension: 4" in out
        assert "Files scanned:       5" in out
        assert "Files chunked:       3" in out
        assert "Files skipped:       2" in out
        assert "Cache:               /tmp/cache" in out

    def test_json_output_has_expected_shape(self, capsys):
        render_result(_make_result(), as_json=True)

        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "repo_slug": "a/b",
            "repo_sha": "0123456789abcdef",
            "cache_dir": "/tmp/cache",
            "skipped": False,
            "dry_run": False,
            "chunk_count": 3,
            "embedding_dimension": 4,
            "files_scanned": 5,
            "files_chunked": 3,
            "files_skipped": 2,
            "decode_failures": 0,
            "parse_failures": 0,
            "durations_seconds": {
                "clone": 0.1,
                "chunk": 0.1,
                "embed": 1.0,
                "index": 0.5,
                "total": 1.7,
            },
        }


class TestHelperFunctions:
    def test_positive_int_accepts_positive_values(self):
        assert _positive_int("1") == 1
        assert _positive_int("42") == 42

    @pytest.mark.parametrize("value", ["0", "-1", "-100"])
    def test_positive_int_rejects_non_positive_values(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int(value)

    def test_slug_from_local_path_sanitizes_the_directory_name(self, tmp_path):
        repo = tmp_path / "my repo!@#"
        repo.mkdir()

        assert _slug_from_local_path(repo) == "local/my_repo___"

    def test_slug_from_local_path_falls_back_to_repo_when_name_is_empty(self):
        assert _slug_from_local_path(Path("/")) == "local/repo"

    def test_slug_from_local_path_resolves_dot_segments_before_sanitizing(self, tmp_path):
        nested = tmp_path / "my-repo"
        nested.mkdir()

        assert _slug_from_local_path(nested / ".") == "local/my-repo"

    def test_validate_cache_dir_rejects_root(self):
        with pytest.raises(ValueError, match="unsafe"):
            _validate_cache_dir(Path("/"))

    def test_validate_cache_dir_rejects_home(self):
        with pytest.raises(ValueError, match="unsafe"):
            _validate_cache_dir(Path.home())

    def test_validate_cache_dir_rejects_cwd(self):
        with pytest.raises(ValueError, match="unsafe"):
            _validate_cache_dir(Path.cwd())

    def test_validate_cache_dir_accepts_a_normal_subdirectory(self, tmp_path):
        assert _validate_cache_dir(tmp_path / "cache") == (tmp_path / "cache").resolve()

    def test_ensure_disjoint_rejects_identical_paths(self, tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            _ensure_disjoint(tmp_path, tmp_path)

    def test_ensure_disjoint_rejects_cache_dir_nested_inside_repo(self, tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            _ensure_disjoint(tmp_path / "repo" / "cache", tmp_path / "repo")

    def test_ensure_disjoint_rejects_repo_nested_inside_cache_dir(self, tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            _ensure_disjoint(tmp_path / "cache", tmp_path / "cache" / "repo")

    def test_ensure_disjoint_accepts_sibling_directories(self, tmp_path):
        _ensure_disjoint(tmp_path / "cache", tmp_path / "repo")
