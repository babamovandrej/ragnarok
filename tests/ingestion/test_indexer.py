import json
import chromadb.errors
import numpy as np
import pytest

from unittest.mock import patch
from tree_sitter_language_pack import SupportedLanguage
from src.config import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION, INDEX_SCHEMA_VERSION
from src.ingestion.schemas import Chunk, SymbolType
from src.ingestion.indexer import (
    MANIFEST_FILENAME,
    IndexManifest,
    build_index,
    load_chroma_collection,
    load_manifest,
)

def _fake_chunk(
    name: str,
    *,
    file_path: str = "a.py",
    start_line: int = 1,
    end_line: int = 2,
    qualified_name: str | None = None,
    symbol_type: SymbolType = "method",
    language: SupportedLanguage = "python",
) -> Chunk:
    return Chunk(
        chunk_id=f"a.py:{name}",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=name,
        qualified_name=qualified_name if qualified_name is not None else f"Example.{name}",
        symbol_type=symbol_type,
        language=language,
        text=f"def {name}():\n    pass",
    )


class TestBuildIndex:
    def test_creates_queryable_chroma_collection_and_bm25_index(self, tmp_path):
        chunks = [_fake_chunk("foo"), _fake_chunk("bar")]
        embeddings = np.random.default_rng(0).random((2, 4), dtype=np.float32)

        manifest = build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")

        assert manifest is not None
        collection = load_chroma_collection(tmp_path)
        assert collection.count() == 2
        assert (tmp_path / "bm25" / "params.index.json").exists()

    def test_manifest_reflects_the_build_inputs(self, tmp_path):
        chunks = [_fake_chunk("foo"), _fake_chunk("bar")]
        embeddings = np.random.default_rng(0).random((2, 4), dtype=np.float32)

        manifest = build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")

        assert manifest is not None
        assert manifest.repo_slug == "a/b"
        assert manifest.repo_sha == "sha1"
        assert manifest.schema_version == INDEX_SCHEMA_VERSION
        assert manifest.embedding_model_id == EMBEDDING_MODEL_ID
        assert manifest.embedding_model_revision == EMBEDDING_MODEL_REVISION
        assert manifest.chunk_count == 2
        assert manifest.embedding_dimension == 4
        assert load_manifest(tmp_path) == manifest

    def test_manifest_file_is_readable_json_with_expected_keys(self, tmp_path):
        chunks = [_fake_chunk("foo")]
        embeddings = np.random.default_rng(0).random((1, 4), dtype=np.float32)

        build_index(chunks, embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")

        payload = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
        assert payload == {
            "repo_slug": "a/b",
            "repo_sha": "sha1",
            "schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model_id": EMBEDDING_MODEL_ID,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "chunk_count": 1,
            "embedding_dimension": 4,
        }

    def test_chunk_metadata_includes_all_chunk_fields(self, tmp_path):
        chunk = _fake_chunk(
            "foo",
            file_path="pkg/mod.py",
            start_line=10,
            end_line=20,
            qualified_name="Example.foo",
            symbol_type="method",
            language="python",
        )
        embeddings = np.random.default_rng(0).random((1, 4), dtype=np.float32)

        build_index([chunk], embeddings, tmp_path, repo_slug="a/b", repo_sha="sha1")

        result = load_chroma_collection(tmp_path).get(ids=[chunk.chunk_id])
        metadatas = result["metadatas"]
        assert metadatas is not None
        metadata = metadatas[0]
        assert metadata == {
            "file_path": "pkg/mod.py",
            "start_line": 10,
            "end_line": 20,
            "symbol_name": "foo",
            "qualified_name": "Example.foo",
            "symbol_type": "method",
            "language": "python",
        }

    def test_is_idempotent_and_drops_stale_chunks(self, tmp_path):
        first_embeddings = np.random.default_rng(0).random((2, 4), dtype=np.float32)
        build_index(
            [_fake_chunk("foo"), _fake_chunk("bar")],
            first_embeddings,
            tmp_path,
            repo_slug="a/b",
            repo_sha="sha1",
        )

        second_embeddings = np.random.default_rng(1).random((2, 4), dtype=np.float32)
        build_index(
            [_fake_chunk("foo"), _fake_chunk("baz")],
            second_embeddings,
            tmp_path,
            repo_slug="a/b",
            repo_sha="sha2",
        )

        collection = load_chroma_collection(tmp_path)
        assert collection.count() == 2
        assert set(collection.get()["ids"]) == {"a.py:foo", "a.py:baz"}
        manifest = load_manifest(tmp_path)
        assert manifest is not None
        assert manifest.repo_sha == "sha2"
        assert manifest.chunk_count == 2

    def test_repeated_builds_do_not_leak_staging_or_previous_directories(self, tmp_path):
        cache_dir = tmp_path / "cache"
        for index in range(3):
            build_index(
                [_fake_chunk(f"foo{index}")],
                np.random.default_rng(index).random((1, 4), dtype=np.float32),
                cache_dir,
                repo_slug="a/b",
                repo_sha=f"sha{index}",
            )

        assert [p.name for p in tmp_path.iterdir()] == ["cache"]

    def test_skips_empty_chunk_list_and_creates_nothing(self, tmp_path):
        result = build_index(
            [], np.empty((0, 4), dtype=np.float32), tmp_path, repo_slug="a/b", repo_sha="sha1"
        )

        assert result is None
        assert not any(tmp_path.iterdir())

    def test_failed_first_build_leaves_no_cache_directory_behind(self, tmp_path):
        cache_dir = tmp_path / "cache"

        with patch("src.ingestion.indexer._build_bm25_index", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                build_index(
                    [_fake_chunk("foo")],
                    np.random.default_rng(0).random((1, 4), dtype=np.float32),
                    cache_dir,
                    repo_slug="a/b",
                    repo_sha="sha1",
                )

        assert not cache_dir.exists()
        assert not any(tmp_path.iterdir())

    def test_preserves_previous_index_on_failed_rebuild(self, tmp_path):
        cache_dir = tmp_path / "cache"
        first_embeddings = np.random.default_rng(0).random((2, 4), dtype=np.float32)
        build_index(
            [_fake_chunk("foo"), _fake_chunk("bar")],
            first_embeddings,
            cache_dir,
            repo_slug="a/b",
            repo_sha="sha1",
        )

        with patch("src.ingestion.indexer._build_bm25_index", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                build_index(
                    [_fake_chunk("baz")],
                    np.random.default_rng(1).random((1, 4), dtype=np.float32),
                    cache_dir,
                    repo_slug="a/b",
                    repo_sha="sha2",
                )

        collection = load_chroma_collection(cache_dir)
        assert set(collection.get()["ids"]) == {"a.py:foo", "a.py:bar"}
        manifest = load_manifest(cache_dir)
        assert manifest is not None
        assert manifest.repo_sha == "sha1"
        assert [p.name for p in tmp_path.iterdir()] == ["cache"]

    def test_failed_rebuild_during_chroma_build_also_preserves_previous_index(self, tmp_path):
        cache_dir = tmp_path / "cache"
        build_index(
            [_fake_chunk("foo")],
            np.random.default_rng(0).random((1, 4), dtype=np.float32),
            cache_dir,
            repo_slug="a/b",
            repo_sha="sha1",
        )

        with patch(
            "src.ingestion.indexer._build_chroma_collection", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError):
                build_index(
                    [_fake_chunk("baz")],
                    np.random.default_rng(1).random((1, 4), dtype=np.float32),
                    cache_dir,
                    repo_slug="a/b",
                    repo_sha="sha2",
                )

        manifest = load_manifest(cache_dir)
        assert manifest is not None
        assert manifest.repo_sha == "sha1"


class TestLoadManifest:
    def test_returns_none_when_never_built(self, tmp_path):
        assert load_manifest(tmp_path) is None

    def test_returns_none_for_nonexistent_directory(self, tmp_path):
        assert load_manifest(tmp_path / "does-not-exist") is None


class TestIndexManifestCompatibility:
    def _manifest(
        self,
        *,
        repo_slug: str = "a/b",
        repo_sha: str = "sha1",
        schema_version: int = INDEX_SCHEMA_VERSION,
        embedding_model_id: str = EMBEDDING_MODEL_ID,
        embedding_model_revision: str = EMBEDDING_MODEL_REVISION,
        chunk_count: int = 1,
        embedding_dimension: int = 4,
    ) -> IndexManifest:
        return IndexManifest(
            repo_slug=repo_slug,
            repo_sha=repo_sha,
            schema_version=schema_version,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=embedding_model_revision,
            chunk_count=chunk_count,
            embedding_dimension=embedding_dimension,
        )

    def test_compatible_when_everything_matches(self):
        assert self._manifest().is_compatible_with("a/b", "sha1")

    def test_incompatible_on_different_sha(self):
        assert not self._manifest().is_compatible_with("a/b", "sha2")

    def test_incompatible_on_different_slug(self):
        assert not self._manifest().is_compatible_with("a/other", "sha1")

    def test_incompatible_on_different_schema_version(self):
        manifest = self._manifest(schema_version=INDEX_SCHEMA_VERSION + 1)
        assert not manifest.is_compatible_with("a/b", "sha1")

    def test_incompatible_on_different_embedding_model_id(self):
        manifest = self._manifest(embedding_model_id="other/model")
        assert not manifest.is_compatible_with("a/b", "sha1")

    def test_incompatible_on_different_embedding_model_revision(self):
        manifest = self._manifest(embedding_model_revision="deadbeef")
        assert not manifest.is_compatible_with("a/b", "sha1")


class TestLoadChromaCollection:
    def test_raises_for_a_cache_dir_that_was_never_built(self, tmp_path):
        with pytest.raises(chromadb.errors.InvalidCollectionException):
            load_chroma_collection(tmp_path)
