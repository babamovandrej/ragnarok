# Phase 9 — Incremental re-indexing via GitHub Actions

**Status:** Code complete; workflow never dispatched
**Milestone:** a scheduled/dispatched CI job re-indexes only changed files
and the live Space picks up the refreshed index without a manual redeploy.

## Tasks

1. **Diff-based reindex script** — `scripts/build_index.py --incremental`
   (requires `--repo-path`): `_merge_incremental_chunks` calls
   `ingestion/clone.py`'s `changed_files_since(repo_dir, since_sha)`,
   re-chunks/re-embeds only added/modified files via `chunk_file`, and
   reuses every other chunk's existing embedding read straight back out of
   the current Chroma collection. Deleted files are dropped entirely.
   `ingest_repository(..., incremental=True)` requires an existing
   compatible manifest at `cache_dir` (raises `IncompatibleCacheError` if
   none exists — "run a full build first") and skips (returns
   `skipped=True`) if the current SHA already matches the cached one.
   *Done:* unit test on a 2-commit fixture repo (file A changed, file B
   untouched) shows only A's chunks re-embedded and B's rows
   byte-identical. ✅ `tests/scripts/test_build_index.py`'s
   `TestMergeIncrementalChunks` (direct, with a tracking fake embedder
   proving B's text is never sent for re-embedding, and B's embedding
   array is byte-identical to the original) and `TestIncrementalIngestion`
   (end-to-end through `ingest_repository`, real git commits, fake
   embedder).

2. **GitHub Actions workflow** — `.github/workflows/reindex.yml`
   (scheduled daily + `workflow_dispatch`): clones the demo repo fresh,
   pulls+rehydrates the last-pushed index from the HF Dataset repo,
   incrementally reindexes, then pushes the result back via
   `scripts/push_index_to_hub.py` (new: `export_and_push` reads the built
   Chroma collection, writes our parquet schema, and uploads it via
   `utils/hf_dataset_io.py`'s `push_to_hub`).
   *Done:* `gh workflow run reindex.yml` completes green, HF Dataset repo
   shows a new commit.
   *Not run* — requires `secrets.HF_TOKEN` and `vars.HF_DATASET_REPO`
   configured in this repo's GitHub settings (neither exists yet — the
   "HF namespace/org" open item flagged since Phase 0 is still
   unresolved) and would trigger a real, external, credentialed push;
   deliberately not dispatched in this session (see `DECISIONS.md`). ✅
   `tests/scripts/test_push_index_to_hub.py` covers `export_and_push`
   (real parquet export from a real built index, `push_to_hub` mocked
   out) and the CLI wiring.

3. **Space auto-refresh** — `utils/hf_dataset_io.py`'s
   `refresh_if_stale(repo_id, cache_dir, repo_slug)`: compares a local
   manifest's `repo_sha` against `latest_commit_sha(repo_id)`, and if
   different, `pull_parquet` + `rehydrate_index_from_parquet` rebuild the
   local Chroma+bm25 index from our parquet schema (never Chroma's own
   on-disk format — the Phase 0 decision this finally implements).
   *Done:* restarting the Space after 9.2 picks up the new index with no
   manual redeploy.
   *Not yet wired into `app.py`'s startup* — `app.py` currently loads
   only the locally pre-built demo cache; calling `refresh_if_stale` at
   import time would require a real HF Dataset repo to check against,
   which doesn't exist yet (blocked on the same open namespace decision
   as task 2). ✅ `tests/utils/test_hf_dataset_io.py` covers
   `latest_commit_sha`, `pull_parquet`, `rehydrate_index_from_parquet`
   (real parquet → real rebuilt Chroma collection), and `refresh_if_stale`
   (both the skip-when-fresh and pull-and-rehydrate-when-stale paths),
   all with the network calls (`HfApi`, `hf_hub_download`) mocked out.

## Verification for this phase

Make a small, real commit to a forked copy of the demo repo (or simulate
one on the fixture repo), trigger `reindex.yml`, confirm the Dataset repo
updates, then restart the Space and confirm it serves the refreshed
index. *Not performed* — this requires the HF namespace/token setup noted
above, which is a real external/credentialed prerequisite this session
didn't create (see `DECISIONS.md`'s Phase 9 entry). All other
verification (unit tests for the incremental merge logic, the parquet
export/push script, and the Dataset-repo pull/rehydrate/refresh
functions, `ruff check .`, `pytest -q`) is already green with zero
regressions to the Phase 1-8 suites.
