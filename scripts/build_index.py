import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import CACHE_ROOT, INDEX_SCHEMA_VERSION
from src.ingestion.chunker import chunk_file, collect_chunks
from src.ingestion.clone import (
    GitCommandError,
    changed_files_since,
    current_sha,
    repo_name_from_url,
    repo_slug_from_url,
    shallow_clone,
)
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import build_index, load_chroma_collection, load_manifest
from src.ingestion.schemas import Chunk, IngestionStats
from src.retrieval.schemas import chunk_from_chroma_record

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_INVALID_ARGUMENT = 2
EXIT_CLONE_FAILED = 3
EXIT_NO_CHUNKS_PRODUCED = 4
EXIT_INDEX_BUILD_FAILED = 5
EXIT_INCOMPATIBLE_CACHE = 6
EXIT_INTERRUPTED = 130


class IngestionError(RuntimeError):
    """Base class for known, expected ingestion pipeline failures."""


class NoChunksProducedError(IngestionError):
    """Raised when a repo yielded zero indexable chunks."""


class EmbeddingFailedError(IngestionError):
    """Raised when the embedding model failed to load or embed chunk texts."""


class IndexBuildFailedError(IngestionError):
    """Raised when writing the Chroma/bm25 index failed."""


class IncompatibleCacheError(IngestionError):
    """Raised when a differently-built index already exists and --rebuild wasn't passed."""


@dataclass(frozen=True)
class IngestionResult:
    """Everything the CLI needs to report about one ingestion run."""

    repo_slug: str
    repo_sha: str
    cache_dir: Path
    chunk_count: int
    embedding_dimension: int
    stats: IngestionStats
    clone_seconds: float
    chunk_seconds: float
    embedding_seconds: float
    index_seconds: float
    total_seconds: float
    skipped: bool = False
    dry_run: bool = False


def _positive_int(value: str) -> int:
    """Parse a CLI integer argument, rejecting anything not greater than zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return parsed


def _slug_from_local_path(repo_path: Path) -> str:
    """Derive a filesystem-safe cache slug for a local repo path, e.g. 'local/my-repo'."""
    name = repo_path.resolve().name
    safe_name = "".join(char if char.isalnum() or char in "_-." else "_" for char in name)
    return f"local/{safe_name or 'repo'}"


def _validate_cache_dir(cache_dir: Path) -> Path:
    """Reject cache paths that resolve to obviously unsafe or shared filesystem locations."""
    resolved = cache_dir.resolve()
    unsafe = {Path("/"), Path.home().resolve(), Path.cwd().resolve()}
    if resolved in unsafe:
        raise ValueError(f"Refusing to use unsafe cache directory: {resolved}")
    return resolved


def _ensure_disjoint(cache_dir: Path, repo_dir: Path) -> None:
    """Reject configurations where the index cache and the repo directory would collide."""
    cache_resolved = cache_dir.resolve()
    repo_resolved = repo_dir.resolve()
    overlaps = (
        cache_resolved == repo_resolved
        or cache_resolved in repo_resolved.parents
        or repo_resolved in cache_resolved.parents
    )
    if overlaps:
        raise ValueError(
            f"Cache directory {cache_resolved} must not overlap "
            f"with repository path {repo_resolved}"
        )


def _make_workspace(keep_clone: bool, repo_slug: str) -> Path:
    """Create a scratch directory to clone into; temporary unless keep_clone is set."""
    if keep_clone:
        workspace = CACHE_ROOT / "clones" / repo_slug
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace
    return Path(tempfile.mkdtemp(prefix="ragnarok-clone-"))


def _check_existing_index(
    cache_dir: Path, repo_slug: str, repo_sha: str, clone_seconds: float, total_started: float
) -> IngestionResult | None:
    """Return a skip-result if a compatible index already exists at cache_dir, else None."""
    existing = load_manifest(cache_dir)
    if existing is None:
        return None
    if existing.is_compatible_with(repo_slug, repo_sha):
        logger.info("Index for %s@%s is already up to date; skipping", repo_slug, repo_sha)
        return IngestionResult(
            repo_slug=repo_slug,
            repo_sha=repo_sha,
            cache_dir=cache_dir,
            chunk_count=existing.chunk_count,
            embedding_dimension=existing.embedding_dimension,
            stats=IngestionStats(),
            clone_seconds=clone_seconds,
            chunk_seconds=0.0,
            embedding_seconds=0.0,
            index_seconds=0.0,
            total_seconds=time.perf_counter() - total_started,
            skipped=True,
        )
    raise IncompatibleCacheError(
        f"An incompatible index already exists at {cache_dir} "
        f"(built for {existing.repo_slug}@{existing.repo_sha}); pass --rebuild to replace it."
    )


def _merge_incremental_chunks(
    cache_dir: Path, repo_dir: Path, since_sha: str, embedder: Embedder
) -> tuple[list[Chunk], np.ndarray, IngestionStats]:
    """Re-chunk/re-embed only files changed since since_sha; reuse existing chunks/embeddings."""
    changed = changed_files_since(repo_dir, since_sha)
    changed_paths = {c.path for c in changed}
    changed_paths |= {c.previous_path for c in changed if c.previous_path}

    collection = load_chroma_collection(cache_dir)
    existing = collection.get(include=["documents", "metadatas", "embeddings"])
    kept_chunks: list[Chunk] = []
    kept_embeddings: list[list[float]] = []
    for chunk_id, text, metadata, embedding in zip(
        existing["ids"], existing["documents"], existing["metadatas"], existing["embeddings"]
    ):
        if metadata["file_path"] in changed_paths:
            continue
        kept_chunks.append(chunk_from_chroma_record(chunk_id, text, metadata))
        kept_embeddings.append(embedding)

    new_chunks: list[Chunk] = []
    for changed_file in changed:
        if changed_file.status.startswith("D"):
            continue
        new_chunks.extend(chunk_file(repo_dir / changed_file.path, repo_dir))

    new_embeddings = embedder.embed([chunk.text for chunk in new_chunks]) if new_chunks else None

    all_chunks = kept_chunks + new_chunks
    if kept_embeddings and new_embeddings is not None:
        all_embeddings = np.vstack([np.array(kept_embeddings), new_embeddings])
    elif new_embeddings is not None:
        all_embeddings = new_embeddings
    else:
        all_embeddings = np.array(kept_embeddings)

    stats = IngestionStats(
        files_scanned=len(changed),
        files_chunked=len({cf.path for cf in changed if not cf.status.startswith("D")}),
        chunks_created=len(new_chunks),
    )
    return all_chunks, all_embeddings, stats


def ingest_repository(
    repo_url: str | None = None,
    *,
    repo_path: Path | None = None,
    cache_dir: Path | None = None,
    revision: str | None = None,
    device: str | None = None,
    embedding_batch_size: int | None = None,
    rebuild: bool = False,
    incremental: bool = False,
    dry_run: bool = False,
    keep_clone: bool = False,
) -> IngestionResult:
    """Run clone -> chunk -> embed -> index for a repo, or skip if a compatible index exists."""
    if incremental and repo_path is None:
        raise ValueError("--incremental requires --repo-path (an existing local checkout).")
    if repo_url is not None and repo_path is not None:
        raise ValueError("Provide exactly one of repo_url or repo_path.")
    if repo_path is not None and revision is not None:
        raise ValueError("--revision is only supported with repo_url, not repo_path.")
    if repo_path is not None and keep_clone:
        raise ValueError("--keep-clone is only supported with repo_url, not repo_path.")

    total_started = time.perf_counter()
    workspace: Path | None = None

    try:
        if repo_path is not None:
            repo_slug = _slug_from_local_path(repo_path)
            repo_dest = repo_path
            clone_seconds = 0.0
            repo_sha = current_sha(repo_path)
        elif repo_url is not None:
            repo_slug = repo_slug_from_url(repo_url)
            workspace = _make_workspace(keep_clone, repo_slug)
            repo_dest = workspace / repo_name_from_url(repo_url)
            clone_started = time.perf_counter()
            shallow_clone(repo_url, repo_dest, revision=revision)
            clone_seconds = time.perf_counter() - clone_started
            repo_sha = current_sha(repo_dest)
            if keep_clone:
                logger.info("Keeping clone of %s at %s", repo_slug, repo_dest)
        else:
            raise ValueError("Provide exactly one of repo_url or repo_path.")

        resolved_cache_dir = _validate_cache_dir(
            cache_dir if cache_dir is not None else CACHE_ROOT / repo_slug
        )
        _ensure_disjoint(resolved_cache_dir, repo_dest)
        logger.info("Resolved %s to commit %s", repo_slug, repo_sha)

        if incremental:
            existing_manifest = load_manifest(resolved_cache_dir)
            if existing_manifest is None:
                raise IncompatibleCacheError(
                    f"No existing index at {resolved_cache_dir} to update incrementally; "
                    "run a full build first."
                )
            if existing_manifest.repo_sha == repo_sha:
                logger.info("Index for %s@%s is already up to date; skipping", repo_slug, repo_sha)
                return IngestionResult(
                    repo_slug=repo_slug,
                    repo_sha=repo_sha,
                    cache_dir=resolved_cache_dir,
                    chunk_count=existing_manifest.chunk_count,
                    embedding_dimension=existing_manifest.embedding_dimension,
                    stats=IngestionStats(),
                    clone_seconds=clone_seconds,
                    chunk_seconds=0.0,
                    embedding_seconds=0.0,
                    index_seconds=0.0,
                    total_seconds=time.perf_counter() - total_started,
                    skipped=True,
                )

            chunk_started = time.perf_counter()
            try:
                embedder = Embedder(device=device, batch_size=embedding_batch_size)
                chunks, embeddings, stats = _merge_incremental_chunks(
                    resolved_cache_dir, repo_dest, existing_manifest.repo_sha, embedder
                )
            except Exception as exc:
                raise EmbeddingFailedError(
                    f"Incremental update failed for {repo_slug}: {exc}"
                ) from exc
            chunk_seconds = time.perf_counter() - chunk_started
            embedding_seconds = 0.0
            logger.info(
                "Incrementally updated %s: %d new/changed chunk(s), %d reused",
                repo_slug,
                stats.chunks_created,
                len(chunks) - stats.chunks_created,
            )
        else:
            if not dry_run and not rebuild:
                skip_result = _check_existing_index(
                    resolved_cache_dir, repo_slug, repo_sha, clone_seconds, total_started
                )
                if skip_result is not None:
                    return skip_result

            chunk_started = time.perf_counter()
            chunks, stats = collect_chunks(repo_dest)
            chunk_seconds = time.perf_counter() - chunk_started
            logger.info(
                "Chunked %s: %d chunk(s) from %d file(s) (%d skipped)",
                repo_slug,
                len(chunks),
                stats.files_chunked,
                stats.files_skipped,
            )

            if not chunks:
                raise NoChunksProducedError(
                    f"No supported source chunks were produced for {repo_slug} "
                    f"(files_scanned={stats.files_scanned}, files_skipped={stats.files_skipped}, "
                    f"decode_failures={stats.decode_failures}, "
                    f"parse_failures={stats.parse_failures})"
                )

            if dry_run:
                return IngestionResult(
                    repo_slug=repo_slug,
                    repo_sha=repo_sha,
                    cache_dir=resolved_cache_dir,
                    chunk_count=len(chunks),
                    embedding_dimension=0,
                    stats=stats,
                    clone_seconds=clone_seconds,
                    chunk_seconds=chunk_seconds,
                    embedding_seconds=0.0,
                    index_seconds=0.0,
                    total_seconds=time.perf_counter() - total_started,
                    dry_run=True,
                )

            embed_started = time.perf_counter()
            try:
                embedder = Embedder(device=device, batch_size=embedding_batch_size)
                embeddings = embedder.embed([chunk.text for chunk in chunks])
            except Exception as exc:
                raise EmbeddingFailedError(f"Embedding failed for {repo_slug}: {exc}") from exc
            embedding_seconds = time.perf_counter() - embed_started

        if not chunks:
            raise NoChunksProducedError(
                f"Incremental update for {repo_slug} produced zero chunks; "
                "the repo may now be empty of indexable source."
            )

        index_started = time.perf_counter()
        try:
            manifest = build_index(
                chunks, embeddings, resolved_cache_dir, repo_slug=repo_slug, repo_sha=repo_sha
            )
        except Exception as exc:
            raise IndexBuildFailedError(f"Index build failed for {repo_slug}: {exc}") from exc
        if manifest is None:
            raise IndexBuildFailedError(f"Index build for {repo_slug} produced no manifest")
        index_seconds = time.perf_counter() - index_started

        return IngestionResult(
            repo_slug=repo_slug,
            repo_sha=repo_sha,
            cache_dir=resolved_cache_dir,
            chunk_count=manifest.chunk_count,
            embedding_dimension=manifest.embedding_dimension,
            stats=stats,
            clone_seconds=clone_seconds,
            chunk_seconds=chunk_seconds,
            embedding_seconds=embedding_seconds,
            index_seconds=index_seconds,
            total_seconds=time.perf_counter() - total_started,
        )
    finally:
        if workspace is not None and not keep_clone:
            shutil.rmtree(workspace, ignore_errors=True)


def render_result(result: IngestionResult, *, as_json: bool) -> None:
    """Print the final ingestion summary: JSON to stdout, or a human-readable block."""
    if as_json:
        payload = {
            "repo_slug": result.repo_slug,
            "repo_sha": result.repo_sha,
            "cache_dir": str(result.cache_dir),
            "skipped": result.skipped,
            "dry_run": result.dry_run,
            "chunk_count": result.chunk_count,
            "embedding_dimension": result.embedding_dimension,
            "files_scanned": result.stats.files_scanned,
            "files_chunked": result.stats.files_chunked,
            "files_skipped": result.stats.files_skipped,
            "decode_failures": result.stats.decode_failures,
            "parse_failures": result.stats.parse_failures,
            "durations_seconds": {
                "clone": round(result.clone_seconds, 2),
                "chunk": round(result.chunk_seconds, 2),
                "embed": round(result.embedding_seconds, 2),
                "index": round(result.index_seconds, 2),
                "total": round(result.total_seconds, 2),
            },
        }
        print(json.dumps(payload, indent=2))
        return

    if result.skipped:
        status = "Skipped (already up to date)"
    elif result.dry_run:
        status = "Dry run"
    else:
        status = "Indexed"
    print(f"{status} {result.repo_slug}@{result.repo_sha[:7]}\n")
    print(f"Chunks:              {result.chunk_count:,}")
    print(f"Embedding dimension: {result.embedding_dimension}")
    print(f"Files scanned:       {result.stats.files_scanned}")
    print(f"Files chunked:       {result.stats.files_chunked}")
    print(f"Files skipped:       {result.stats.files_skipped}")
    print(f"Decode failures:     {result.stats.decode_failures}")
    print(f"Parse failures:      {result.stats.parse_failures}")
    print(f"Cache:               {result.cache_dir}")
    print(f"Duration:            {result.total_seconds:.2f}s")


def configure_logging(level: str) -> None:
    """Initialize root logging once at CLI startup; basicConfig's default handler targets stderr."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


_EXIT_CODES_HELP = """
exit codes:
  0    success
  1    unexpected error
  2    invalid argument
  3    repository access failed (clone or local path)
  4    no chunks produced
  5    embedding or index build failed
  6    incompatible cache exists (pass --rebuild)
  130  interrupted
"""


def build_parser() -> argparse.ArgumentParser:
    """Construct the build_index CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Ingest a repo into a local Chroma + bm25s index.",
        epilog=_EXIT_CODES_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    repo_group = parser.add_mutually_exclusive_group(required=True)
    repo_group.add_argument(
        "repo_url", nargs="?", help="Public GitHub repo URL to clone and index."
    )
    repo_group.add_argument(
        "--repo-path", type=Path, help="Path to an existing local Git repo to index in place."
    )

    parser.add_argument(
        "--cache-dir", type=Path, help="Override the default CACHE_ROOT/<repo_slug> location."
    )
    parser.add_argument(
        "--revision", help="Branch, tag, or commit SHA to check out (default branch if omitted)."
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], help="Force the embedding device."
    )
    parser.add_argument(
        "--embedding-batch-size", type=_positive_int, help="Override the embedding batch size."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Clone and chunk only; skip embedding/indexing."
    )
    parser.add_argument(
        "--keep-clone", action="store_true", help="Persist the clone instead of deleting it."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit a machine-readable JSON summary on stdout."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"build_index (index schema v{INDEX_SCHEMA_VERSION})",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rebuild", action="store_true", help="Replace an existing index even if compatible."
    )
    mode.add_argument(
        "--incremental",
        action="store_true",
        help="Re-chunk/re-embed only files changed since the cached SHA (requires --repo-path).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        result = ingest_repository(
            args.repo_url,
            repo_path=args.repo_path,
            cache_dir=args.cache_dir,
            revision=args.revision,
            device=args.device,
            embedding_batch_size=args.embedding_batch_size,
            rebuild=args.rebuild,
            incremental=args.incremental,
            dry_run=args.dry_run,
            keep_clone=args.keep_clone,
        )
    except KeyboardInterrupt:
        logger.warning("Ingestion cancelled")
        return EXIT_INTERRUPTED
    except (ValueError, NotImplementedError) as exc:
        logger.error("%s", exc)
        return EXIT_INVALID_ARGUMENT
    except GitCommandError as exc:
        logger.error("%s", exc)
        return EXIT_CLONE_FAILED
    except NoChunksProducedError as exc:
        logger.error("%s", exc)
        return EXIT_NO_CHUNKS_PRODUCED
    except IncompatibleCacheError as exc:
        logger.error("%s", exc)
        return EXIT_INCOMPATIBLE_CACHE
    except (EmbeddingFailedError, IndexBuildFailedError) as exc:
        logger.error("%s", exc)
        return EXIT_INDEX_BUILD_FAILED
    except IngestionError as exc:
        logger.error("%s", exc)
        return EXIT_UNEXPECTED_ERROR
    except Exception:
        logger.exception("Unexpected ingestion failure")
        return EXIT_UNEXPECTED_ERROR

    render_result(result, as_json=args.json)
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
