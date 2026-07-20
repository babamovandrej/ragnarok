# Phase 1 — Repo scaffold, dependency setup, empty module stubs

**Status:** Local verification done (tasks 1-4, 6-8); CI push (task 5)
pending user confirmation to commit/push.
**Milestone:** package imports cleanly, tests run, CI is green, the app
stub serves locally.

## Tasks

1. **Package skeleton** — `ragnarok/` with all subpackages (`ingestion`,
   `retrieval`, `agent`, `generation`, `baseline`, `eval`, `ui`, `utils`),
   each with `__init__.py`.
   *Done:* `python -c "import ragnarok"` succeeds from repo root.

2. **Packaging config** — `pyproject.toml` (hatchling backend,
   `requires-python = ">=3.11,<3.13"`).
   *Done:* `pip install -e .` succeeds in a clean venv.

3. **Dependency lock** — `requirements.txt` with pinned versions (see
   `docs/architecture.md` for the library list and `DECISIONS.md` for
   rationale on each open choice).
   *Done:* `pip install -r requirements.txt` succeeds clean; `pip check`
   reports no conflicts.

4. **Test harness** — `tests/` + `conftest.py` + one `test_import_ok` per
   subpackage.
   *Done:* `pytest -q` passes, 0 failures.

5. **CI skeleton** — `.github/workflows/ci.yml` runs `ruff check` +
   `pytest` on push/PR.
   *Done:* a pushed branch shows a green Actions run.

6. **DECISIONS.md + docs scaffold** — root `DECISIONS.md` seeded with the
   decisions from Phase 0 planning; `docs/roadmap.md` + 10 phase docs.
   *Done:* files exist with the Phase 0 decisions recorded (already done
   as part of this planning session).

7. **gitignore additions** — RAGnarok-specific entries (already added as
   part of Phase 0 planning: `.ragnarok_cache/`, `*.parquet`,
   `data/index_staging/`, `models/`, `*.gguf`, `flagged/`,
   `gradio_cached_examples/`).
   *Done:* an untracked file dropped in `.ragnarok_cache/` never appears
   in `git status`.

8. **App entrypoint stub** — root `app.py`, Gradio Blocks placeholder
   page.
   *Done:* `python app.py` serves a page at `localhost:7860`.

## Verification for this phase

Run, in order: `pip install -e .`, `pip install -r requirements.txt`,
`pytest -q`, `python app.py` (then hit `localhost:7860` in a browser),
push a branch and confirm the Actions run is green.
