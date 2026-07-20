# Phase 10 — README/pitch polish

**Status:** Not started
**Milestone:** the README stands alone as evidence — architecture,
numbers, and a demo GIF, no stale claims.

## Tasks

1. **Architecture diagram** — one-page pipeline diagram in `README.md`.
   *Done:* renders correctly in GitHub's README preview.

2. **Demo GIF/video** — walkthrough of pasting a repo + multi-hop question
   with hop path visible.
   *Done:* embeds/plays in README preview.

3. **Eval numbers + ablation table** — Phase 7 output reproduced in
   README with interpretation.
   *Done:* README numbers match the latest committed eval artifact (no
   stale copies).

4. **DECISIONS.md final pass** — every major architectural choice across
   phases 1-9 has a dated entry.
   *Done:* entry count consistent with major decisions made across all
   phases.

5. **License/attribution audit** — table of every third-party model
   (`Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Coder-1.5B-Instruct-GGUF`,
   `jina-embeddings-v2-base-code`, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
   with confirmed license.
   *Done:* table present in README, all links verified.

## Open item to resolve before this phase

- What RAGAS score/delta counts as "RAGnarok beats naive RAG" for the
  README's headline claim (editorial call, see phase-07-eval.md).

## Verification for this phase (project definition of done)

A visitor to the public HF Space can paste an arbitrary public repo URL
and get cited, grounded answers; the naive-vs-RAGnarok comparison and
hop-path visualization are visible in the UI; eval numbers and the
ablation table are in the README; the whole stack runs at no ongoing cost.
