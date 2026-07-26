import subprocess
from pathlib import Path

import numpy as np

from src.ingestion.chunker import collect_chunks
from src.ingestion.clone import current_sha
from src.ingestion.incremental import merge_incremental_chunks
from src.ingestion.indexer import build_index


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.random.default_rng(0).random((len(texts), 4), dtype=np.float32)


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


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

        initial_chunks, _ = collect_chunks(repo, first_sha)
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
        second_sha = current_sha(repo)

        class _TrackingEmbedder:
            def __init__(self) -> None:
                self.embed_calls: list[list[str]] = []

            def embed(self, texts: list[str]) -> np.ndarray:
                self.embed_calls.append(list(texts))
                return np.array([[100.0, 100.0, 100.0, 100.0] for _ in texts], dtype=np.float32)

        embedder = _TrackingEmbedder()
        merged_chunks, merged_embeddings, stats = merge_incremental_chunks(
            cache_dir, repo, first_sha, embedder, second_sha
        )

        merged_by_file = dict(zip((c.file_path for c in merged_chunks), merged_embeddings))
        assert set(merged_by_file) == {"a.py", "b.py"}
        assert np.array_equal(merged_by_file["b.py"], original_b_embedding)
        assert not np.array_equal(merged_by_file["a.py"], original_b_embedding)
        assert stats.chunks_created == 1
        all_embedded_texts = [text for call in embedder.embed_calls for text in call]
        assert not any("bar" in text for text in all_embedded_texts)
        assert any("999" in text for text in all_embedded_texts)

        merged_by_file_chunk = {c.file_path: c for c in merged_chunks}
        assert merged_by_file_chunk["a.py"].repo_sha == second_sha
        assert merged_by_file_chunk["b.py"].repo_sha == first_sha

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

        initial_chunks, _ = collect_chunks(repo, first_sha)
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
        second_sha = current_sha(repo)

        merged_chunks, _, stats = merge_incremental_chunks(
            cache_dir, repo, first_sha, _FakeEmbedder(), second_sha
        )

        assert {c.file_path for c in merged_chunks} == {"a.py"}
        assert stats.chunks_created == 0
