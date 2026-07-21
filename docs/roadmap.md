# RAGnarok Roadmap

Index of every phase's task breakdown. See [`architecture.md`](architecture.md)
for the current-state design snapshot and root [`DECISIONS.md`](../DECISIONS.md)
for the chronological log of judgment calls and why they were made.

Fixed demo repo for the running example and Phase 7 eval set: **`encode/starlette`**.

| Phase | Doc | Milestone | Status |
|---|---|---|---|
| 0 | (this planning session) | Repo structure, dependencies, phased task list agreed | Done |
| 1 | [phase-01-scaffold.md](phases/phase-01-scaffold.md) | Package imports, tests run, CI green, app stub serves | Done |
| 2 | [phase-02-ingestion.md](phases/phase-02-ingestion.md) | Chunker → embedder → indexer works offline on starlette | Done |
| 3 | [phase-03-retrieval.md](phases/phase-03-retrieval.md) | Hybrid retrieval + rerank beats dense-only on sampled queries | Code complete — manual comparison pending |
| 4 | [phase-04-generation.md](phases/phase-04-generation.md) | Single-hop generation + citation verification, CPU backend | Code complete — manual `scripts/ask.py` run pending |
| 5 | [phase-05-agent.md](phases/phase-05-agent.md) | LangGraph multi-hop agent with tracked hop path | Not started |
| 6 | [phase-06-comparison.md](phases/phase-06-comparison.md) | Naive-vs-RAGnarok side-by-side comparison UI | Not started |
| 7 | [phase-07-eval.md](phases/phase-07-eval.md) | Hand-labeled eval set, RAGAS scoring, ablation table | Not started |
| 8 | [phase-08-deployment.md](phases/phase-08-deployment.md) | Live public HF Space, ZeroGPU + CPU fallback, paste-a-repo UX | Not started |
| 9 | [phase-09-ci.md](phases/phase-09-ci.md) | Incremental reindexing via GitHub Actions | Not started |
| 10 | [phase-10-polish.md](phases/phase-10-polish.md) | README pitch, eval numbers, demo GIF, license audit | Not started |

## Deferred, non-blocking decisions

These were flagged during Phase 0 planning as genuinely open but not
blocking — revisit at the noted phase:

- **Ingestion ceiling defaults** (proposed: ~1,500 indexable chunks / ~15MB
  source / 60s clone timeout) — confirm exact numbers before Phase 8.
- **HF namespace/org** for the Space and Dataset repo — needed by Phase 8/9.
- **RAGAS "passing bar"** for the README's headline claim — editorial call,
  needed by Phase 10.
- **Exact composition of the 30-50 Q&A items** (split across
  single-hop/multi-hop/adversarial-IDK) — draft rubric in Phase 7, final
  sign-off once drafted.

## How to use this roadmap

Update the Status column as phases complete. When a task reveals that an
earlier decision needs to change, update the relevant phase doc, this
roadmap, and add a new dated entry to `DECISIONS.md` referencing the old
one — don't silently diverge or edit history.
