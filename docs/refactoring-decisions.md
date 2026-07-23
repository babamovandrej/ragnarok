# Refactoring Decisions

One entry per task from the approved plan, in the order implemented. Each records the
problem, the decision made, the reasoning, alternatives considered, the compatibility
impact, and how it was validated. Commit hashes refer to this session's history on `main`.

## 1. Finalize doc/git-state cleanup — `736d91a`

**Problem:** `DECISIONS.md` and all 11 `docs/phases/*.md` files were deleted in the working
tree but not committed. `README.md`, `docs/architecture.md`, and `docs/roadmap.md` still
linked to them, and `docs/architecture.md`'s repo-structure diagram referenced two files that
never existed (`src/ui/layout.py`, `src/utils/logging.py`) and omitted `scripts/compare_retrieval.py`.

**Decision:** Commit the deletion; scrub every dangling reference from the three docs;
correct the stale repo-structure listing.

**Reasoning:** The user confirmed the deletion was intentional cleanup, not accidental loss.
Leaving the repo in a state where core docs link to nonexistent files is worse than removing
the links, and the durable design record already lives in `architecture.md`/`roadmap.md`.

**Alternatives considered:** Restoring the deleted files (rejected — user confirmed intent
to remove them). Leaving the working-tree state untouched (rejected — doc links would keep
pointing at nothing).

**Compatibility impact:** None; documentation-only.

**Validation:** `grep -rn "DECISIONS.md\|docs/phases"` across the repo returns no hits
post-change.

## 2. Packaging & dependency hygiene — `478577c`

**Problem:** `pyproject.toml` declared `dependencies = []`, so `pip install -e .` alone
installed zero third-party packages; every local setup silently depended on also running
`pip install -r requirements.txt`. Several `requirements.txt` packages (`langchain`,
`langchain-community`, `langchain-openai`, `langchain-text-splitters`, `python-dotenv`)
looked unused since only `langchain-core` is imported directly in `src/eval/ragas_runner.py`.

**Decision:** Populate `pyproject.toml`'s `dependencies` with the 17 packages actually
imported by `src`/`scripts`/`app.py` (verified via an import grep across the whole tree),
plus a `dev` extras group for `ruff`/`pytest`/`pytest-cov`/`coverage`. **Did not** prune
anything from `requirements.txt`.

**Reasoning:** `pip show ragas` confirms it requires `langchain`, `langchain-community`,
`langchain-core`, and `langchain_openai` at install time; `pip show langchain` confirms it
requires `langchain-text-splitters`; `pip show python-dotenv` shows it's required by
`pydantic-settings`, itself required by `langchain-community`. Every package that looked
unused from source-level imports alone is a real transitive dependency of `ragas`. Removing
them from `requirements.txt` wouldn't reduce what gets installed (pip would just re-resolve
them for `ragas` anyway) — it would only remove their pin, hurting reproducibility.

**Alternatives considered:** Pruning the langchain-* packages based on the initial
source-grep finding (rejected once `pip show` revealed they're real transitive deps —
proceeding would have broken `ragas`'s install or silently unpinned versions). Leaving
`pyproject.toml` untouched (rejected — the empty dependency list is a real, user-confirmed
bug).

**Compatibility impact:** `pip install -e .` now installs a working environment standalone;
`pip install -e . && pip install -r requirements.txt` (the documented reproducible path)
is unaffected since `requirements.txt` wasn't touched.

**Validation:** `pip install -e ".[dev]" --dry-run` resolves cleanly against the existing
venv with no conflicts, ending in `Would install ragnarok-0.1.0`. `ruff check .` and
`pytest -q` (295 passed) both still green.

## 3. Static typing enforcement (mypy) — `3632f0c`

**Problem:** No mypy config existed anywhere, mypy wasn't a pinned dependency, and CI never
ran it — the codebase's strong annotation discipline was a convention, not an enforced gate.

**Decision:** Add `[tool.mypy]` with `strict = true`, `python_version = "3.11"`,
`explicit_package_bases = true` (needed since `scripts/` has no `__init__.py` but is imported
as `scripts.build_index` elsewhere), scoped to `src`, `scripts`, `app.py`. Add narrow
`ignore_missing_imports` overrides for `ragas.*`/`spaces`/`bm25s`/`pyarrow.*` (confirmed via
a stub-only run that these four are the *only* packages mypy can't find stubs for), and one
file-scoped override for `src/generation/zerogpu_backend.py` disabling
`disallow_untyped_calls`/`disallow_untyped_decorators`/`warn_return_any` (transformers'
`AutoModel`/`AutoTokenizer` and `spaces.GPU` are fundamentally untyped at that boundary).
Fixed every real finding: chromadb's `IncludeEnum`/`Optional`-result typing (narrowed with
asserts, which also act as genuine runtime guards), `chunk_from_chroma_record`'s widened
parameter type with explicit `str()`/`int()`/`cast()` narrowing, two `Optional` values used
without a check (`HfApi().dataset_info().sha`, llama-cpp's message `content`), a
`TypedDict` (`_AgentStateUpdate`) replacing bare `dict` return types on LangGraph nodes, and
two missing return annotations.

**Reasoning:** `strict = true` surfaced only 29 errors on the first pass and 7 more once
`scripts`/`app.py` were added — a small, fixable set given the codebase's existing
discipline, and worth the small number of narrowly-scoped ignores to get there rather than
settling for a looser config.

**Alternatives considered:** A looser config (`disallow_untyped_defs` only, no `strict`) —
rejected since `strict` was achievable with minimal narrowly-scoped exceptions, and gives
more real protection. Blanket `ignore_missing_imports = true` — rejected in favor of an
explicit module list, so a future untyped-stub gap doesn't silently get bucketed in.

**Compatibility impact:** None on runtime behavior — every fix is either a type-only
annotation, an `assert` that documents/enforces an invariant already true at runtime, or a
`cast`/conversion (`str()`/`int()`) that's a no-op given the actual data. `AGENT_HOP_DECISION_MAX_TOKENS`
(32, matching the prior hardcoded value) was added to `src/config.py` because fixing
`src/agent/graph.py`'s bare-`dict` typing required touching that line anyway.

**Validation:** `mypy` reports `Success: no issues found in 46 source files` (later 48 once
Tasks 6–7 added two new modules). `ruff check .`, `ruff format --check .` (after running
`ruff format .` once to fix 13 pre-existing, unrelated formatting-drift files CI had never
caught), and `pytest -q` (295 passed) all green.

## 4. Config centralization cleanup — `977501d`

**Problem:** `src/generation/zerogpu_backend.py` defined its own `ZEROGPU_MODEL_ID`/
`ZEROGPU_MODEL_REVISION`, inconsistent with every other backend (which pull model IDs from
`src/config.py`). `scripts/build_index.py` caught `NotImplementedError` in its CLI handler
even though nothing raises it.

**Decision:** Move the two ZeroGPU constants into `src/config.py`; drop `NotImplementedError`
from the except clause.

**Reasoning:** Consistency with the established pattern; dead branches are confusing to a
future reader trying to understand what conditions the handler is really for.

**Alternatives considered:** None — straightforward moves with a single correct answer.

**Compatibility impact:** `tests/generation/test_zerogpu_backend.py` updated to import the
constants from `src.config` instead of `src.generation.zerogpu_backend` (they're the same
values, just relocated).

**Validation:** `pytest -q` (295 passed), `ruff check .`, `mypy` all green.

## 5. Deduplicate the retrieval-pipeline assembly logic — `ef81801`

**Problem:** The dense→sparse→RRF→rerank sequence was hand-rolled three times: the canonical
`src/agent/tools.py::retrieve_hybrid`, and duplicated inline in `scripts/ask.py::answer_question`
and `scripts/compare_retrieval.py::compare_query`.

**Decision:** Both scripts now call `retrieve_hybrid` directly.
`scripts/compare_retrieval.py` still calls `query_dense` separately, since it specifically
needs the dense-only result as a comparison baseline (something `retrieve_hybrid` doesn't
expose) — accepting one extra, cheap single-query embed/dense-query as the cost of not
duplicating the canonical pipeline function.

**Reasoning:** Three independent copies of the same four-step pipeline is a correctness risk
(they can silently drift apart) with no benefit; the existing `retrieve_hybrid` was already
the right shared abstraction, just not reused.

**Alternatives considered:** Extracting a *new* shared helper (rejected — one already
existed and was fit for purpose). Making `compare_query` avoid the redundant dense
embed/query by threading its result into `retrieve_hybrid` (rejected — `retrieve_hybrid`'s
signature doesn't accept a pre-computed dense result, and changing it for one diagnostic
script's benefit would complicate the interface for its two other, more important callers).

**Compatibility impact:** None observable — existing tests exercise both scripts end-to-end
against a real (fake-embedder-backed) index rather than mocking the internal pipeline steps,
so they passed unchanged.

**Validation:** `pytest -q tests/scripts/test_ask.py tests/scripts/test_compare_retrieval.py`
(17 passed), full suite (295 passed), `ruff check .`, `mypy` all green.

## 6. Resolve `src/baseline` naming/layering smell — `048dd7e`

**Problem:** `src/baseline/naive_rag.py` held not just the naive baseline (`answer_naive`)
but the full production orchestration (`answer_ragnarok`, `compare`), importing from
`src.agent` to do it — a "baseline" package depending on the "agent" package is backwards
layering, and the module name no longer matched most of its contents.

**Decision:** New `src/pipeline.py` holds `answer_ragnarok` and `compare()`.
`src/baseline/naive_rag.py` now contains only `answer_naive` with zero cross-package
dependencies. Updated all call sites (`app.py`, `src/eval/ablation.py`,
`scripts/run_eval.py`) and split the test file accordingly (`tests/baseline/test_naive_rag.py`
keeps naive-only tests; a new `tests/test_pipeline.py` holds the `answer_ragnarok`/`compare`
tests).

**Reasoning:** `src.pipeline` sitting above both `src.baseline` and `src.agent` matches its
actual role — orchestrating both arms — without forcing either lower-level package to
depend on the other.

**Alternatives considered:** Renaming `naive_rag.py` in place without extracting the
orchestration logic (rejected — doesn't fix the `baseline → agent` dependency direction,
just the name). A generic `src/orchestration.py` (rejected in favor of `src/pipeline.py`,
matching the domain vocabulary already used throughout the codebase — "pipeline" is how
`PipelineResult` and the rest of the codebase already refer to this concept).

**Compatibility impact:** Two tests that imported `PipelineResult` via `naive_rag`'s
transitive re-export were updated to import it from its real home,
`src.baseline.schemas` — required once `strict` mypy's `no_implicit_reexport` was enabled
in Task 3 and caught this exact pattern in `src/eval/ablation.py`.

**Validation:** Full suite (295 passed at time of commit), `ruff check .`, `mypy` all green.

## 7. Move embedded business logic out of `scripts/build_index.py` — `3420608`

**Problem:** `scripts/build_index.py` held real business logic — `_validate_cache_dir`,
`_ensure_disjoint`, `_merge_incremental_chunks` — that belonged with the rest of the
ingestion pipeline, not embedded in a CLI script.

**Decision:** `validate_cache_dir`/`ensure_disjoint` move to `src/ingestion/indexer.py`
(next to the other cache-dir I/O it already owns: `load_manifest`, `build_index`,
`load_chroma_collection`). `merge_incremental_chunks` moves to a new
`src/ingestion/incremental.py`, since it combines chunking, diffing, and embedding concerns
distinct from `indexer.py`'s read/write responsibilities. Leading underscores dropped since
these are now real public APIs of their new modules, called externally by
`scripts/build_index.py`.

**Reasoning:** `scripts/build_index.py --incremental` and `app.py` both need this logic;
keeping it in a script module made it awkward to reuse and signaled (correctly) that the
script was doing more than argparse + orchestration.

**Alternatives considered:** A single `src/ingestion/incremental.py` holding all three
functions (rejected — cache-path validation and incremental chunk-merging are distinct
concerns; splitting them keeps each module narrowly focused). Moving `ingest_repository`
itself out of the script too (considered, rejected as scope creep beyond what the approved
plan named — `ingest_repository` is genuinely the CLI's orchestration function, and a larger
move wasn't part of the reviewed task list).

**Compatibility impact:** Test coverage moved with the code:
`TestMergeIncrementalChunks` → `tests/ingestion/test_incremental.py`;
`validate_cache_dir`/`ensure_disjoint` cases → `tests/ingestion/test_indexer.py`.
`tests/scripts/test_build_index.py` keeps only CLI-level tests.

**Validation:** Full suite (295 passed), `ruff check .`, `mypy` all green.
`docs/architecture.md`'s incremental-indexing section and repo-structure diagram updated
to reference the new module.

## 8. Error-handling polish pass — `7355a8f`

**Problem:** `app.py::_ingest` caught `Exception` from `ingest_repository` and returned a
status string to the Gradio UI without logging anything first — a failed ingestion in the
deployed Space left no server-side trace to debug from.

**Decision:** Add `logger.exception(...)` before converting to the user-facing message.
Add one-line comments on the three call sites that deliberately catch broad
`Exception`/`BaseException` (`src/ingestion/chunker.py`'s per-file chunking guard,
`src/ingestion/indexer.py`'s staging-dir cleanup and cache-swap rollback), explaining why
each is intentional.

**Reasoning:** A silent `except Exception` in a user-facing entry point is a real
operational gap — the next person debugging a failed Space deployment would have nothing
to go on beyond the status string shown to the end user. The broad-except comments make
already-correct code self-explanatory instead of looking like an oversight to a future
reviewer.

**Alternatives considered:** Narrowing `_ingest`'s except clause to specific exception types
(rejected — `ingest_repository` can raise several different `IngestionError` subclasses plus
arbitrary lower-level errors; a Gradio callback's job is to never crash the UI thread
regardless of cause, so `Exception` is the right scope here, it just needed logging).

**Compatibility impact:** None on behavior; added a new `logger = logging.getLogger(__name__)`
to `app.py` (it had none before).

**Validation:** New test asserting the exception is logged
(`tests/test_app.py::TestIngest::test_logs_the_exception_before_returning_the_status_message`,
added in Task 10). Full suite green.

## 9. UI/UX polish for the Gradio Compare/Ingest experience — `44351ce`

**Problem:** See the UI/UX section of [`code-audit.md`](code-audit.md) — unused hop-link
rendering, aggregate-only citation display, no empty-input guards, no error handling, no
loading feedback, no theming.

**Decision:**
- Wire `src/ui/hop_viz.py::render_hop_path` into the RAGnarok column (clickable GitHub
  file/line links) when `(repo_slug, repo_sha)` are known; fall back to the prior
  plain-text hop list otherwise.
- Render each citation individually with its own verified/UNVERIFIED status and a GitHub
  link when repo context is available.
- Add an empty-question/empty-URL guard in both `compare_view.py` and `ingest_view.py`.
- Wrap the compare callback in `try`/`except` so a backend failure renders a friendly inline
  message in both columns instead of Gradio's generic error toast.
- Add a `gr.Progress` call in both views describing what's running.
- Apply `gr.themes.Soft()` on the outer `Blocks` in `app.py`.
- `app.py` resolves `(repo_slug, repo_sha)` via `load_manifest(DEMO_CACHE_DIR)`, falling back
  to `None` (no links, matching prior behavior) if the demo index hasn't been built yet.

**Reasoning:** `render_hop_path` and per-citation detail were the two most concrete,
already-half-built pieces of unfinished UX in the repo — finishing them was higher value
than any cosmetic change. Error/empty-state handling and progress feedback directly serve
users hitting a CPU-bound app where requests can take a long time and occasionally fail.

**Alternatives considered:** Connecting the Ingest tab's result to the Compare tab's repo
selection (so Compare could target a freshly-ingested repo, not just the fixed demo) — the
approved plan explicitly flagged this as a larger behavioral change needing separate
confirmation; **not implemented**, left as a documented limitation. Custom CSS/HTML theming
beyond a built-in Gradio theme (rejected — `gr.themes.Soft()` meets "polished, not garish"
without introducing custom CSS to maintain).

**Compatibility impact:** `build_compare_view`'s signature gained an optional
`repo_context` keyword parameter (defaults preserve prior behavior exactly when omitted).
`_format_result` was split into `_format_naive`/`_format_ragnarok`/`_render_citations`;
callers of the old private helper (only the test suite) were updated accordingly.

**Validation:** Full test suite rewritten for the new functions (empty-input, error-path,
citation-rendering, hop-link-rendering cases all covered), 306 passed at time of commit.
Manual smoke test: `python -c "import app; ..."` confirmed the Blocks app builds, and
`render_comparison` was exercised directly for both the empty-question and populated-question
cases (see [`final-validation.md`](final-validation.md) for the exact commands and output).
Real-model, browser-based click-through was not performed — descoped along with all other
real-model verification per the user's earlier decision.

## 10. Close test-coverage gaps — `bbbe2c2`

**Problem:** `src/config.py` had no dedicated test file. The logging and repo-context helpers
added in Tasks 8–9 didn't yet have direct test coverage beyond what was added alongside them.

**Decision:** New `tests/test_config.py` with invariant-focused assertions (not literal-value
mirrors of the source): `SOURCE_EXTENSIONS` stays in sync with `LANGUAGE_BY_EXTENSION`,
every `WHOLE_FILE_LANGUAGES` entry is reachable from some extension, every pinned model
revision matches a 40-character hex git SHA pattern, and the resource-limit constants stay
positive/correctly ordered relative to each other. Added a test for
`app.py::_demo_repo_context` (both the built- and not-yet-built-manifest cases) and for the
`logger.exception` call in `_ingest`.

**Reasoning:** Testing literal constant values (`assert AGENT_MAX_HOPS == 3`) would just
mirror the source and break on every legitimate tuning change; testing the *invariants* that
matter (extension/language consistency, valid SHA format, positive limits) catches the kind
of typo or drift that's actually worth catching.

**Alternatives considered:** None significant — the gap and its fix were both narrowly
scoped and uncontroversial.

**Compatibility impact:** None; test-only addition.

**Validation:** `pytest -q` — 316 passed (up from 295 at the start of this session, all net-new
or relocated tests, no regressions).

## 11. CI / quality gates — `9c91d27`

**Problem:** CI ran only `ruff check .` (lint) and `pytest -q`; it never ran the mypy config
added in Task 3, never checked formatting with `ruff format --check .` (13 files had
pre-existing formatting drift that lint alone never caught), and never used the
`pytest-cov`/`coverage` packages that were already pinned as dependencies.

**Decision:** Add a "Format check" step (`ruff format --check .`), a "Type check" step
(`mypy`), and switch the test step to `pytest -q --cov --cov-report=term-missing` with no
hard coverage-failure threshold. Added `[tool.coverage.run] source = ["src", "scripts", "app"]`
to `pyproject.toml` so the same command works identically locally. Did **not** add a
`.env.example`: `HF_TOKEN`/`HF_DATASET_REPO` are the only env-like values in the project,
and both are passed as explicit CLI args / GitHub Actions secrets to
`scripts/push_index_to_hub.py`, never read from a local `.env` file
(`python-dotenv` in `requirements.txt` is a transitive dependency of `uvicorn[standard]`,
confirmed via `pip show`, not used directly by any first-party code).

**Reasoning:** A quality gate that isn't enforced in CI isn't really a gate — the
mypy config from Task 3 and the formatter both needed to run on every push, not just
locally. Coverage reporting (without a hard gate) gives visibility without risking a
brittle CI failure over a coverage percentage nobody has agreed on yet.

**Alternatives considered:** A hard coverage threshold (e.g. `--cov-fail-under=90`) —
rejected; current coverage is 96% but setting an enforced bar wasn't part of the approved
scope and risks blocking unrelated future PRs over incidental coverage dips.

**Compatibility impact:** None on runtime; CI takes marginally longer per run (two extra
steps).

**Validation:** `.github/workflows/ci.yml` parsed successfully with `yaml.safe_load`; every
step (`ruff check .`, `ruff format --check .`, `mypy`, `pytest -q --cov --cov-report=term-missing`)
run manually and passed — see [`final-validation.md`](final-validation.md).

## 12. Final documentation pass — this commit

**Problem:** The required deliverables (`docs/code-audit.md`, `docs/refactoring-decisions.md`,
`docs/final-validation.md`) didn't exist yet, and README/architecture/roadmap needed a final
pass to confirm they still accurately describe the post-refactor repository.

**Decision:** Write all three docs from the actual findings and commits of this session
(no invented numbers, no fabricated test/benchmark results). Re-verified README/architecture/roadmap
against the final code state rather than assuming earlier per-task doc edits were sufficient.

**Reasoning:** Documentation that isn't traceable to real commits and real command output is
worse than no documentation.

**Alternatives considered:** None — this is a reporting task, not a design decision.

**Compatibility impact:** None; documentation-only.

**Validation:** See [`final-validation.md`](final-validation.md) for the exact commands and
their real output.
