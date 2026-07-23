# Phase 10 — README/pitch polish

**Status:** Partially done — diagram, license audit, and the DECISIONS.md
pass are done; eval numbers and the demo GIF are genuinely pending a real
run/deployment, not filled in with placeholders.
**Milestone:** the README stands alone as evidence — architecture,
numbers, and a demo GIF, no stale claims.

## Tasks

1. **Architecture diagram** — a Mermaid pipeline diagram in `README.md`
   (ingestion → dense+sparse retrieval → RRF fusion → rerank → multi-hop
   agent → generation (CPU/ZeroGPU+fallback) → citation verification,
   plus the naive baseline running in parallel into the comparison UI).
   *Done:* renders correctly in GitHub's README preview. ✅

2. **Demo GIF/video** — walkthrough of pasting a repo + multi-hop
   question with hop path visible.
   *Not done.* Requires a real running deployment (local or the live
   Space) to record from — genuinely can't be produced without one. See
   `DECISIONS.md`.

3. **Eval numbers + ablation table** — Phase 7 output reproduced in
   README with interpretation.
   *Not done.* `scripts/run_eval.py --ablation` has never been run for
   real (Phase 7's deferred real run); the README explicitly says so
   rather than showing placeholder or fabricated numbers. Once that run
   happens, its output goes directly into a new README "Eval numbers"
   section, sourced from the artifact, not retyped from memory.

4. **DECISIONS.md final pass** — every major architectural choice across
   phases 1-9 has a dated entry.
   *Done:* reviewed the full file; every phase 0-9 has at least one dated
   entry (phases 6, 7, 8, 9 each have 3-4, reflecting more judgment calls
   as the system grew), consistent dating (`2026-07-2X`), and consistent
   `[Phase N]` heading format throughout. ✅

5. **License/attribution audit** — table of every third-party model
   (`Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Coder-1.5B-Instruct-GGUF`,
   `jina-embeddings-v2-base-code`, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
   with confirmed license.
   *Done:* table in `README.md`; every license was fetched live from each
   model's real Hugging Face API metadata (`cardData.license`) in this
   session, not recalled from memory — all four are Apache 2.0. Links
   point at the real model pages. ✅

## Open item (still unresolved)

- **What RAGAS score/delta counts as "RAGnarok beats naive RAG"** for the
  README's headline claim — still an editorial call, and still blocked on
  Phase 7's real ablation numbers existing to judge against.

## Verification for this phase (project definition of done)

A visitor to the public HF Space can paste an arbitrary public repo URL
and get cited, grounded answers; the naive-vs-RAGnarok comparison and
hop-path visualization are visible in the UI; eval numbers and the
ablation table are in the README; the whole stack runs at no ongoing
cost. *Not fully met yet* — this is the project's overall definition of
done, not just this phase's, and depends on Phase 8's live deployment and
Phase 7's real eval run, both still pending (see `DECISIONS.md`).
