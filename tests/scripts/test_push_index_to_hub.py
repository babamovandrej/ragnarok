from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest

from scripts.push_index_to_hub import (
    EXIT_INVALID_ARGUMENT,
    EXIT_SUCCESS,
    build_parser,
    export_and_push,
    main,
)
from src.ingestion.indexer import build_index
from src.ingestion.schemas import Chunk


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


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    chunks = [_fake_chunk("auth", "class AuthenticationMiddleware:\n    pass")]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    build_index(chunks, embeddings, tmp_path, repo_slug="owner/repo", repo_sha="sha1")
    return tmp_path


class TestExportAndPush:
    def test_exports_a_parquet_named_after_the_manifests_sha_and_pushes_it(self, cache_dir):
        with patch("scripts.push_index_to_hub.push_to_hub") as mock_push:
            parquet_path = export_and_push(cache_dir, "owner/dataset")

        assert parquet_path == cache_dir / "sha1.parquet"
        assert parquet_path.is_file()
        table = pq.read_table(parquet_path)
        assert table.num_rows == 1
        assert table.column("chunk_id")[0].as_py() == "auth"
        mock_push.assert_called_once_with(parquet_path, "owner/dataset")

    def test_raises_without_a_manifest(self, tmp_path):
        with pytest.raises(ValueError, match="No manifest"):
            export_and_push(tmp_path, "owner/dataset")


class TestBuildParser:
    def test_parses_cache_dir_and_repo_id(self):
        args = build_parser().parse_args(["/tmp/cache", "owner/dataset"])

        assert args.cache_dir == Path("/tmp/cache")
        assert args.repo_id == "owner/dataset"


class TestMain:
    def test_missing_index_returns_invalid_argument(self, tmp_path):
        exit_code = main([str(tmp_path / "missing"), "owner/dataset"])

        assert exit_code == EXIT_INVALID_ARGUMENT

    def test_success_calls_export_and_push(self, cache_dir, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "scripts.push_index_to_hub.export_and_push",
            lambda cache_dir, repo_id: calls.append((cache_dir, repo_id)),
        )

        exit_code = main([str(cache_dir), "owner/dataset"])

        assert exit_code == EXIT_SUCCESS
        assert calls == [(cache_dir, "owner/dataset")]
