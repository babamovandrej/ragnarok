from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from src.ingestion.indexer import build_index, load_chroma_collection, load_manifest
from src.ingestion.schemas import Chunk
from src.utils.hf_dataset_io import (
    export_chunks_to_parquet,
    latest_commit_sha,
    pull_parquet,
    refresh_if_stale,
    rehydrate_index_from_parquet,
)


def _fake_chunk(name: str, text: str, file_path: str = "a.py") -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path=file_path,
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        repo_sha="sha1",
        text=text,
    )


def test_export_chunks_to_parquet_has_expected_columns(tmp_path):
    chunks = [
        Chunk(
            chunk_id="a.py:1-2:foo",
            file_path="a.py",
            start_line=1,
            end_line=2,
            symbol_name="foo",
            qualified_name="foo",
            symbol_type="function",
            language="python",
            repo_sha="sha1",
            text="def foo():\n    pass",
        )
    ]
    embeddings = np.zeros((1, 4), dtype=np.float32)
    output_path = tmp_path / "chunks.parquet"

    export_chunks_to_parquet(chunks, embeddings, output_path)
    table = pq.read_table(output_path)

    assert table.column_names == [
        "chunk_id",
        "file_path",
        "start_line",
        "end_line",
        "symbol_name",
        "qualified_name",
        "symbol_type",
        "language",
        "repo_sha",
        "imports",
        "text",
        "embedding",
    ]
    assert table.num_rows == 1
    assert table.column("embedding")[0].as_py() == [0.0, 0.0, 0.0, 0.0]


class TestLatestCommitSha:
    def test_returns_the_dataset_repos_sha(self):
        with patch("src.utils.hf_dataset_io.HfApi") as mock_api_cls:
            mock_api_cls.return_value.dataset_info.return_value.sha = "abc123"

            assert latest_commit_sha("owner/dataset") == "abc123"
            mock_api_cls.return_value.dataset_info.assert_called_once_with("owner/dataset")


class TestPullParquet:
    def test_downloads_the_named_file_into_local_dir(self, tmp_path):
        with patch("src.utils.hf_dataset_io.hf_hub_download") as mock_download:
            mock_download.return_value = str(tmp_path / "abc123.parquet")

            result = pull_parquet("owner/dataset", "abc123.parquet", tmp_path, revision="abc123")

            assert result == tmp_path / "abc123.parquet"
            mock_download.assert_called_once_with(
                repo_id="owner/dataset",
                filename="abc123.parquet",
                repo_type="dataset",
                revision="abc123",
                local_dir=str(tmp_path),
            )


class TestRehydrateIndexFromParquet:
    def test_rebuilds_a_real_chroma_collection_from_the_parquet_schema(self, tmp_path):
        chunks = [_fake_chunk("auth", "class AuthenticationMiddleware:\n    pass")]
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        parquet_path = tmp_path / "abc123.parquet"
        export_chunks_to_parquet(chunks, embeddings, parquet_path)
        cache_dir = tmp_path / "cache"

        manifest = rehydrate_index_from_parquet(parquet_path, cache_dir, "owner/repo", "abc123")

        assert manifest.repo_slug == "owner/repo"
        assert manifest.repo_sha == "abc123"
        assert manifest.chunk_count == 1
        collection = load_chroma_collection(cache_dir)
        assert collection.count() == 1


class TestRefreshIfStale:
    def test_skips_when_local_sha_already_matches_remote(self, tmp_path):
        cache_dir = tmp_path / "cache"
        chunks = [_fake_chunk("auth", "class AuthenticationMiddleware:\n    pass")]
        embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        build_index(chunks, embeddings, cache_dir, repo_slug="owner/repo", repo_sha="abc123")

        with (
            patch("src.utils.hf_dataset_io.latest_commit_sha", return_value="abc123"),
            patch("src.utils.hf_dataset_io.pull_parquet") as mock_pull,
        ):
            refreshed = refresh_if_stale("owner/dataset", cache_dir, "owner/repo")

        assert refreshed is False
        mock_pull.assert_not_called()

    def test_pulls_and_rehydrates_when_remote_sha_differs(self, tmp_path):
        cache_dir = tmp_path / "cache"
        old_chunks = [_fake_chunk("auth", "old text")]
        old_embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        build_index(
            old_chunks, old_embeddings, cache_dir, repo_slug="owner/repo", repo_sha="old-sha"
        )

        new_chunks = [
            _fake_chunk("auth", "new text"),
            _fake_chunk("bar", "def bar(): pass", "b.py"),
        ]
        new_embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        parquet_path = tmp_path / "new-sha.parquet"
        export_chunks_to_parquet(new_chunks, new_embeddings, parquet_path)

        with (
            patch("src.utils.hf_dataset_io.latest_commit_sha", return_value="new-sha"),
            patch("src.utils.hf_dataset_io.pull_parquet", return_value=parquet_path) as mock_pull,
        ):
            refreshed = refresh_if_stale("owner/dataset", cache_dir, "owner/repo")

        assert refreshed is True
        mock_pull.assert_called_once_with(
            "owner/dataset", "new-sha.parquet", cache_dir, revision="new-sha"
        )
        manifest = load_manifest(cache_dir)
        assert manifest.repo_sha == "new-sha"
        assert manifest.chunk_count == 2
