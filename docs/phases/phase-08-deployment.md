# Phase 8 — Space deployment

**Status:** Not started
**Milestone:** a public HF Space is live; a visitor can paste an arbitrary
public repo URL and get cited, grounded answers end to end.

## Tasks

1. **ZeroGPU wiring** — `generation/zerogpu_backend.py`,
   `@spaces.GPU`-decorated `Qwen/Qwen2.5-Coder-7B-Instruct`.
   *Done:* deployed Space answers one query via the GPU path (visible in
   Spaces logs).

2. **CPU/GPU auto-fallback** — quota-exceeded exception → transparent CPU
   fallback, backend surfaced in UI.
   *Done:* forcing the exception path still returns an answer, tagged "CPU
   fallback."

3. **Live "paste any repo" ingestion** — URL box, size/file-count cap,
   progress indicator, then unlocks Q&A.
   *Done:* pasting a small real public repo through the browser completes
   ingestion and answers a question end-to-end.

4. **Memory-budget guardrails** — `utils/caching.py`: LRU eviction capped
   at N resident in-memory indexes, hard ingestion ceiling, starlette
   pre-warmed at startup.
   *Done:* sequentially ingesting 3 repos (cap=2) evicts the oldest from
   memory while disk cache re-load stays fast.

5. **Hop-path visualization in UI** — `ui/hop_viz.py`.
   *Done:* a known multi-hop demo question visibly shows ≥2 hops with
   clickable `file#Lstart-Lend` citations.

6. **Space repo + README front-matter** — HF Spaces YAML (`sdk: gradio`,
   `app_file: app.py`, license).
   *Done:* Space shows "Running" at its public URL.

7. **Model revision pinning audit** — every third-party
   `from_pretrained`/model load pinned to an exact revision.
   *Done:* no unpinned third-party model loads remain (grep-verified).

## Open items to resolve before this phase

- Ingestion ceiling defaults (proposed: ~1,500 indexable chunks / ~15MB
  source / 60s clone timeout) — confirm exact numbers.
- HF namespace/org for the Space and Dataset repo.

## Verification for this phase

Visit the public Space URL from a fresh browser session (not logged in as
the owner), paste a small public repo, ask a question, and confirm a
cited, grounded answer appears with the comparison view and hop-path
visualization both visible.
