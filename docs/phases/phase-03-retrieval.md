# Phase 3 — Hybrid retrieval + reranking

**Status:** Not started
**Milestone:** hybrid retrieval + reranking demonstrably beats dense-only
retrieval on a manual sample of real starlette queries.

## Tasks

1. **Dense retriever** — `retrieval/dense.py`.
   *Done:* unit test returns top-k by cosine similarity on the starlette
   index.

2. **Sparse retriever** — `retrieval/sparse.py` (`bm25s`).
   *Done:* unit test with an exact real function/symbol name from
   starlette (e.g. `AuthenticationMiddleware`) returns that chunk in
   top-3, proving BM25 catches what dense embeddings miss on identifiers.

3. **Hybrid fusion (RRF)** — `retrieval/hybrid.py`, dedupe by `chunk_id`.
   *Done:* constructed-case unit test shows the fused ranking differs from
   either single-retriever ranking.

4. **Cross-encoder rerank** — `retrieval/rerank.py`
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`, pinned revision).
   *Done:* unit test on a 2-candidate pair (one obviously more relevant)
   asserts correct reorder.

5. **Manual dense-only vs hybrid+rerank comparison** — run 8-10
   hand-picked starlette queries through both paths, log a results table
   in this doc.
   *Done:* the table below is filled in, showing hybrid+rerank places the
   correct ground-truth chunk in top-3 more often than dense-only for a
   majority of sampled queries.

## Manual comparison results (fill in during execution)

| # | Query | Dense-only top-3 hit? | Hybrid+rerank top-3 hit? | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| ... | | | | |

## Verification for this phase

Run the manual comparison script/notebook against starlette, populate the
table above, and confirm hybrid+rerank wins on a majority of the sampled
queries before moving to Phase 4.
