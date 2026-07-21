# Phase 3 — Hybrid retrieval + reranking

**Status:** Code complete; manual comparison run (task 5) pending
**Milestone:** hybrid retrieval + reranking demonstrably beats dense-only
retrieval on a manual sample of real starlette queries.

## Tasks

1. **Dense retriever** — `retrieval/dense.py`.
   *Done:* unit test returns top-k by cosine similarity on the starlette
   index. ✅ `tests/retrieval/test_dense.py`. The Chroma collection is now
   created with `metadata={"hnsw:space": "cosine"}` (see `DECISIONS.md`)
   so `query_dense`'s `1 - distance` recovers true cosine similarity.

2. **Sparse retriever** — `retrieval/sparse.py` (`bm25s`).
   *Done:* unit test with an exact real function/symbol name from
   starlette (e.g. `AuthenticationMiddleware`) returns that chunk in
   top-3, proving BM25 catches what dense embeddings miss on identifiers.
   ✅ `tests/retrieval/test_sparse.py`.

3. **Hybrid fusion (RRF)** — `retrieval/hybrid.py`, dedupe by `chunk_id`.
   *Done:* constructed-case unit test shows the fused ranking differs from
   either single-retriever ranking. ✅ `tests/retrieval/test_hybrid.py`.

4. **Cross-encoder rerank** — `retrieval/rerank.py`
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`, pinned revision).
   *Done:* unit test on a 2-candidate pair (one obviously more relevant)
   asserts correct reorder. ✅ `tests/retrieval/test_rerank.py` — the
   `Reranker.rerank` sorting/wiring logic is covered with the real
   `CrossEncoder` mocked out (no model download/inference); a
   `@pytest.mark.integration` test using the real model on a 2-candidate
   pair is also present but not executed in this environment (see below).

5. **Manual dense-only vs hybrid+rerank comparison** — run 8-10
   hand-picked starlette queries through both paths, log a results table
   in this doc.
   *Not yet run.* `scripts/compare_retrieval.py` is implemented and ready
   (takes a built index directory + one or more queries, prints/JSON-dumps
   the dense-only and hybrid+reranked chunk ranking for each) — see
   `tests/scripts/test_compare_retrieval.py` for its wiring/CLI tests with
   `Embedder`/`Reranker` mocked out. Actually running it against
   `encode/starlette` requires: `python scripts/build_index.py
   https://github.com/encode/starlette` (clones + embeds, real model),
   then `python scripts/compare_retrieval.py .ragnarok_cache/encode/
   starlette "<query>" ...`. This was intentionally not run in this
   session to avoid the embedding + reranking model load/inference cost;
   run it locally, fill in the table below, and flip the status above to
   `Done` once hybrid+rerank wins on a majority of the sampled queries.

## Manual comparison results (fill in during execution)

| # | Query | Dense-only top-3 hit? | Hybrid+rerank top-3 hit? | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| ... | | | | |

## Verification for this phase

Run `scripts/compare_retrieval.py` against a built starlette index,
populate the table above, and confirm hybrid+rerank wins on a majority of
the sampled queries before moving to Phase 4. All other verification
(unit tests for each retrieval stage, `ruff check .`, `pytest -q`) is
already green with zero regressions to the Phase 2 suite.
