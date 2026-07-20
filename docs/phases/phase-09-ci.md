# Phase 9 — Incremental re-indexing via GitHub Actions

**Status:** Not started
**Milestone:** a scheduled/dispatched CI job re-indexes only changed files
and the live Space picks up the refreshed index without a manual redeploy.

## Tasks

1. **Diff-based reindex script** — `scripts/build_index.py --incremental`:
   given the last-indexed SHA (stored with the pushed dataset) and current
   SHA, re-chunk/re-embed only changed files, merge into parquet, push.
   *Done:* unit test on a 2-commit fixture repo (file A changed, file B
   untouched) shows only A's chunks re-embedded and B's rows
   byte-identical.

2. **GitHub Actions workflow** — `.github/workflows/reindex.yml`,
   scheduled/dispatch, runs incremental reindex, pushes to the HF Dataset
   repo via a secret token.
   *Done:* `gh workflow run reindex.yml` completes green, HF Dataset repo
   shows a new commit.

3. **Space auto-refresh** — Space startup compares local cached SHA vs the
   Dataset repo's latest, pulls fresh parquet if stale.
   *Done:* restarting the Space after 9.2 picks up the new index with no
   manual redeploy.

## Verification for this phase

Make a small, real commit to a forked copy of the demo repo (or simulate
one on the fixture repo), trigger `reindex.yml`, confirm the Dataset repo
updates, then restart the Space and confirm it serves the refreshed index.
