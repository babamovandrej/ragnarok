# Phase 6 — Naive-vs-RAGnarok comparison mode

**Status:** Code complete; manual `python app.py` UI check pending
**Milestone:** one question, typed once, visibly produces two different,
correctly-labeled answers side by side.

## Tasks

1. **Naive baseline pipeline** — `baseline/naive_rag.py`'s `answer_naive`:
   single dense top-k, a plain prompt (no BM25, rerank, hops, or citation
   instructions), same `LLMBackend`.
   *Done:* unit test with call-count assertions proves no hybrid/rerank/
   agent calls occur. ✅ `tests/baseline/test_naive_rag.py` — patches
   `query_sparse`, `fuse_rrf`, `Reranker.rerank`, and `build_graph`, and
   asserts none are called.

2. **Parallel run harness** — `baseline/naive_rag.py`'s `compare(query,
   ...) -> (naive_result, ragnarok_result)`, both tagged
   `PipelineResult` (`baseline/schemas.py`: `pipeline`, `answer`,
   `citations`, `hop_path`). `answer_ragnarok` runs the full Phase 5 agent
   graph, so the "ragnarok" side already covers hybrid+rerank+hops.
   *Done:* unit test runs one query, returns two tagged structured
   answers. ✅ `tests/baseline/test_naive_rag.py`.

3. **Side-by-side UI** — `ui/compare_view.py`'s `build_compare_view`, a
   two-column Gradio Blocks layout wired to an injected `compare_fn`
   (dependency-injected the same way as Phase 5's graph, so the UI is
   testable without any real model call). `app.py` now builds this view
   directly, bound to the demo repo's cache dir.
   *Done:* typing a question in the local app populates both columns with
   visibly different answers/citations.
   *Not yet manually run.* `python app.py` will launch the real UI, but
   clicking "Ask" requires a real built starlette index plus real
   embedding/rerank/generation inference — not exercised in this session
   (CPU cost). `build_compare_view`/`render_comparison`/`_format_result`
   are unit tested directly with fake pipeline results.
   ✅ `tests/ui/test_compare_view.py`, `tests/test_app.py`.

4. **Verified-vs-unverified badge** — visually distinguish RAGnarok's
   verified citations/hop path from naive's unverified single-shot answer.
   *Done:* UI shows a "verified" badge only on the RAGnarok column.
   ✅ `tests/ui/test_compare_view.py::TestFormatResult` — a naive result
   never renders the badge text; a ragnarok result renders "Verified
   citations" (or "No citations found") and, when present, a hop path.

## Verification for this phase

Run `python app.py`, ask a real multi-hop starlette question in the local
UI, and confirm the naive column gives a shallower/less-grounded answer
than the RAGnarok column, with the badge only on the latter. All other
verification (unit tests for the naive pipeline, comparison harness, and
UI rendering functions, `ruff check .`, `pytest -q`) is already green with
zero regressions to the Phase 1-5 suites.
