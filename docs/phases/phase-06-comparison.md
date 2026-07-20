# Phase 6 — Naive-vs-RAGnarok comparison mode

**Status:** Not started
**Milestone:** one question, typed once, visibly produces two different,
correctly-labeled answers side by side.

## Tasks

1. **Naive baseline pipeline** — `baseline/naive_rag.py`: single dense
   top-k, no BM25/rerank/hops, same LLM.
   *Done:* unit test with call-count assertions proves no hybrid/rerank/
   agent calls occur.

2. **Parallel run harness** — `compare(query) -> (naive_result,
   ragnarok_result)`.
   *Done:* unit test runs one query, returns two tagged structured
   answers.

3. **Side-by-side UI** — `ui/compare_view.py`, two-column Gradio Blocks
   layout.
   *Done:* typing a question in the local app populates both columns with
   visibly different answers/citations.

4. **Verified-vs-unverified badge** — visually distinguish RAGnarok's
   verified citations/hop path from naive's unverified single-shot answer.
   *Done:* UI shows a "verified" badge only on the RAGnarok column.

## Verification for this phase

Run `python app.py`, ask a real multi-hop starlette question in the local
UI, and confirm the naive column gives a shallower/less-grounded answer
than the RAGnarok column, with the badge only on the latter.
