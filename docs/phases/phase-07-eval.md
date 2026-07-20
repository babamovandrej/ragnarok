# Phase 7 — Eval suite

**Status:** Not started
**Milestone:** RAGAS-style scores and a naive/hybrid+rerank/multi-hop
ablation table, computed with zero external API calls.

## Tasks

1. **Hand-labeled Q&A dataset** — `data/eval/qa_starlette.jsonl`, 30-50
   items (single-hop, multi-hop, a few adversarial "not in this repo"
   items).
   *Done:* file loads as 30-50 well-formed records with required fields.

2. **RAGAS runner, own-model judge** — `eval/ragas_runner.py`, custom
   `BaseRagasLLM` subclass around our own generation backend; computes
   faithfulness, context precision/recall.
   *Done:* `python scripts/run_eval.py --pipeline ragnarok` produces
   per-item + aggregate scores with zero external API calls.

3. **Ablation harness** — `eval/ablation.py`: same Q&A set through naive /
   hybrid+rerank / full multi-hop.
   *Done:* `python scripts/run_eval.py --ablation` produces one table, 3
   rows × N metric columns.

4. **Adversarial IDK check** — subset of items about things not in the
   repo.
   *Done:* automated check that ≥80% of adversarial items produce an
   explicit "I don't know."

5. **Results committed** — eval output artifacts saved for README reuse
   (Phase 10).
   *Done:* rerunning `scripts/run_eval.py` reproduces qualitatively
   consistent scores.

## Open item to resolve before/during this phase

- Exact composition of the 30-50 Q&A items (split across
  single-hop/multi-hop/adversarial-IDK) and which specific starlette code
  paths to target — draft a labeling rubric in `data/eval/README.md`
  first, get sign-off, then label.
- What RAGAS score/delta counts as "RAGnarok beats naive RAG" for the
  README's headline claim — an editorial call, not an engineering one.

## Verification for this phase

`python scripts/run_eval.py --ablation` produces a 3-row table where
hybrid+rerank and multi-hop score meaningfully higher than naive on
faithfulness/context precision/recall for the hand-labeled set.
