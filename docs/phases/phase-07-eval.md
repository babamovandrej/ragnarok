# Phase 7 — Eval suite

**Status:** Code complete; manual `scripts/run_eval.py` run pending
**Milestone:** RAGAS-style scores and a naive/hybrid+rerank/multi-hop
ablation table, computed with zero external API calls.

## Tasks

1. **Hand-labeled Q&A dataset** — `data/eval/qa_starlette.jsonl`, 35 items
   (20 single-hop, 10 multi-hop, 5 adversarial "not in this repo" items).
   *Done:* file loads as 30-50 well-formed records with required fields.
   ✅ `tests/eval/test_qa_dataset.py` — every `expected_files`/
   `grounding_notes` pair was checked directly against a shallow clone of
   `encode/starlette` at commit `5174d4c8358a6f06aa8056bafd14c2272dab8dd1`
   (read-only source inspection, no model involved) rather than recalled
   from memory. Rubric: `data/eval/README.md`.

2. **RAGAS runner, own-model judge** — `eval/ragas_runner.py`'s
   `RagnarokRagasLLM(BaseRagasLLM)` wraps `LLMBackend` as the judge (no
   external judge API); `run_ragas_eval` scores `Faithfulness`,
   `LLMContextPrecisionWithoutReference`, and `LLMContextRecall`.
   *Done:* `python scripts/run_eval.py --pipeline ragnarok` produces
   per-item + aggregate scores with zero external API calls.
   ✅ `tests/eval/test_ragas_runner.py` — the `RagnarokRagasLLM` adapter
   (prompt → backend.generate → `LLMResult`) is unit tested with a fake
   backend; a `@pytest.mark.integration` test exercises the real
   `run_ragas_eval`/`CPUBackend` path but is not executed in this
   environment (see below).

3. **Ablation harness** — `eval/ablation.py`'s `run_ablation`: the same
   35 items through naive / hybrid+rerank (`answer_ragnarok(...,
   max_hops=0)`) / full multi-hop (default hops), each scored by RAGAS.
   *Done:* `python scripts/run_eval.py --ablation` produces one table, 3
   rows × N metric columns. ✅ `tests/eval/test_ablation.py` — pipeline
   routing and row aggregation are tested with `answer_naive`/
   `answer_ragnarok`/`run_ragas_eval` mocked out.

4. **Adversarial IDK check** — `eval/metrics.py`'s `idk_rate`/
   `is_idk_response`, computed automatically over the 5 adversarial items
   whenever `scripts/run_eval.py` (non-ablation mode) runs.
   *Done:* automated check that ≥80% of adversarial items produce an
   explicit "I don't know." ✅ `tests/eval/test_metrics.py` for the metric
   itself; the real ≥80% check against generated answers is part of the
   deferred real run (task 2).

5. **Results committed** — eval output artifacts saved for README reuse
   (Phase 10).
   *Not yet run,* since it depends on task 2's real run. Once
   `scripts/run_eval.py` is actually run against a built starlette index,
   commit its output (or a summary of it) for Phase 10 to reuse verbatim.

## Open items (resolved during this phase)

- ~~Exact composition of the 30-50 Q&A items~~ — resolved: see
  `data/eval/README.md` for the full rubric and how each item was
  verified against the real repo.
- **What RAGAS score/delta counts as "RAGnarok beats naive RAG"** for the
  README's headline claim — still an editorial call, deferred to Phase 10
  since it requires real ablation numbers to judge against.

## Verification for this phase

`python scripts/run_eval.py --ablation` against a built starlette index
should produce a 3-row table where `hybrid_rerank` and `multi_hop` score
meaningfully higher than `naive` on faithfulness/context precision/recall.
*Not yet run* — requires a real built starlette index plus real
embedding/rerank/generation/RAGAS-judge inference; intentionally not run
in this session (CPU cost, see `DECISIONS.md`). All other verification
(unit tests for the dataset loader, metrics, RAGAS adapter, and ablation
routing, `ruff check .`, `pytest -q`) is already green with zero
regressions to the Phase 1-6 suites.
