# RAGnarok Roadmap

Phase-by-phase milestone and status tracker. See
[`architecture.md`](architecture.md) for the current-state design snapshot.

Fixed demo repo for the running example and Phase 7 eval set: **`encode/starlette`**.

| Phase | Milestone | Status |
|---|---|---|
| 0 | Repo structure, dependencies, phased task list agreed | Done |
| 1 | Package imports, tests run, CI green, app stub serves | Done |
| 2 | Chunker → embedder → indexer works offline on starlette | Done |
| 3 | Hybrid retrieval + rerank beats dense-only on sampled queries | Code complete — manual comparison pending |
| 4 | Single-hop generation + citation verification, CPU backend | Code complete — manual `scripts/ask.py` run pending |
| 5 | LangGraph multi-hop agent with tracked hop path | Code complete — real multi-hop demonstration run pending |
| 6 | Naive-vs-RAGnarok side-by-side comparison UI | Code complete — manual `python app.py` UI check pending |
| 7 | Hand-labeled eval set, RAGAS scoring, ablation table | Code complete — manual `scripts/run_eval.py` run pending |
| 8 | Live public HF Space, ZeroGPU + CPU fallback, paste-a-repo UX | Code complete — no live Space deployed yet |
| 9 | Incremental reindexing via GitHub Actions | Code complete — workflow never dispatched |
| 10 | README pitch, eval numbers, demo GIF, license audit | Partially done — diagram and license audit done; eval numbers and demo GIF pending |

## Deferred, non-blocking decisions

These were flagged during Phase 0 planning as genuinely open but not
blocking — revisit at the noted phase:

- **Ingestion ceiling defaults** (proposed: ~1,500 indexable chunks / ~15MB
  source / 60s clone timeout) — confirm exact numbers before Phase 8.
- **HF namespace/org** for the Space and Dataset repo — needed by Phase 8/9.
- **RAGAS "passing bar"** for the README's headline claim — editorial call,
  needed by Phase 10; still open since real RAGAS numbers don't exist yet.
- ~~Exact composition of the 30-50 Q&A items~~ — resolved in Phase 7:
  `data/eval/qa_starlette.jsonl` (35 items: 20 single-hop, 10 multi-hop, 5
  adversarial), rubric in `data/eval/README.md`.

## How to use this roadmap

Update the Status column as phases complete. When a task reveals that an
earlier decision needs to change, update this roadmap directly rather than
silently diverging from it.
