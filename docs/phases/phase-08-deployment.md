# Phase 8 — Space deployment

**Status:** Code complete; no live Space deployed yet
**Milestone:** a public HF Space is live; a visitor can paste an arbitrary
public repo URL and get cited, grounded answers end to end.

## Tasks

1. **ZeroGPU wiring** — `generation/zerogpu_backend.py`'s `ZeroGPUBackend`,
   `@spaces.GPU`-decorated `.generate`, `Qwen/Qwen2.5-Coder-7B-Instruct`
   pinned to `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
   *Done:* deployed Space answers one query via the GPU path (visible in
   Spaces logs).
   *Not yet run* — requires an actual ZeroGPU Space; the class loads real
   `transformers`/`torch` and calls `.to("cuda")`, so it cannot even be
   instantiated on this (non-CUDA) machine. ✅
   `tests/generation/test_zerogpu_backend.py` covers the pinned-revision
   loading and generate-decode wiring with `AutoTokenizer`/
   `AutoModelForCausalLM` mocked out.

2. **CPU/GPU auto-fallback** — `generation/fallback_backend.py`'s
   `FallbackBackend(primary, fallback)`: catches the `gr.Error` ZeroGPU
   raises on quota-exceeded and transparently retries on the CPU backend;
   `last_backend_used` ("primary"/"fallback") lets the UI surface which
   one answered.
   *Done:* forcing the exception path still returns an answer, tagged
   "CPU fallback." ✅ `tests/generation/test_fallback_backend.py` — this
   logic needs no real model at all (it's pure control flow over two
   injected `LLMBackend`s), so it's tested for real, not deferred.

3. **Live "paste any repo" ingestion** — `ui/ingest_view.py`'s
   `build_ingest_view`: a URL textbox, "Ingest" button, and status
   message, wired in `app.py` to `scripts.build_index.ingest_repository`
   (the same real ingestion pipeline `scripts/build_index.py` uses).
   *Simplified from the original scope:* a single status message on
   completion, not a live streaming progress indicator — a Gradio
   `gr.Progress`-based streaming version is a follow-up, not implemented
   here (see `DECISIONS.md`).
   *Done:* pasting a small real public repo through the browser completes
   ingestion and answers a question end-to-end.
   *Not yet run* — real ingestion requires cloning + real embedding
   inference. ✅ `tests/ui/test_ingest_view.py` covers the view's
   construction; `tests/test_app.py::TestIngest` covers `_ingest`'s error
   handling with `ingest_repository` mocked out.

4. **Memory-budget guardrails** — `utils/caching.py`'s `LRUCache`
   (capacity `MAX_RESIDENT_REPOS = 2`), used by `app.py`'s `_load_models`
   to cache loaded `(Embedder, Reranker, CPUBackend)` per repo instead of
   reloading them on every question (closing the inefficiency flagged in
   the Phase 6 `DECISIONS.md` entry). The hard ingestion ceiling
   (`MAX_INDEXABLE_CHUNKS`, `MAX_SOURCE_BYTES`) was already enforced in
   `chunker.py` since Phase 2 — confirmed still in place, not re-done.
   *Done:* sequentially ingesting 3 repos (cap=2) evicts the oldest from
   memory while disk cache re-load stays fast. ✅
   `tests/utils/test_caching.py` covers `LRUCache` eviction/recency
   directly (pure data structure, no model needed);
   `tests/test_app.py::TestLoadModels` covers `_load_models`'s
   cache-reuse-per-key behavior with `Embedder`/`Reranker`/`CPUBackend`
   mocked out.

5. **Hop-path visualization in UI** — `ui/hop_viz.py`'s `render_hop_path`:
   numbered markdown with clickable `https://github.com/{repo_slug}/blob/
   {repo_sha}/{file_path}#Lstart-Lend` links per hop.
   *Done:* a known multi-hop demo question visibly shows ≥2 hops with
   clickable `file#Lstart-Lend` citations. ✅ `tests/ui/test_hop_viz.py`
   covers rendering for zero/one/multiple hops. *Not yet wired* into
   `compare_view`'s rendered columns (which currently list hop symbol
   names as plain text) — needs `repo_slug`/`repo_sha` threaded through
   `PipelineResult`, a small follow-up.

6. **Space repo + README front-matter** — HF Spaces YAML front matter
   added to `README.md` (`sdk: gradio`, `sdk_version: 5.50.0`,
   `app_file: app.py`, `license: mit`).
   *Done:* Space shows "Running" at its public URL.
   *Not yet done* — no Space has actually been created/pushed to Hugging
   Face; this requires an HF account/namespace decision (see the open
   item below) and a real `git push`/`huggingface_hub` upload, both real
   external actions not taken in this session.

7. **Model revision pinning audit** — grepped every
   `from_pretrained`/`SentenceTransformer(`/`CrossEncoder(`/
   `hf_hub_download(` call in `src/`.
   *Done:* no unpinned third-party model loads remain (grep-verified). ✅
   All five call sites (`Embedder`, `Reranker`, `CPUBackend`,
   `ZeroGPUBackend`'s tokenizer and model) pass an explicit `revision=`.

## Open items (still unresolved — need the user's decision)

- **HF namespace/org** for the Space and Dataset repo — needed before any
  real deployment; not something this session can decide unilaterally.
- **Ingestion ceiling defaults** — the proposed numbers
  (`MAX_INDEXABLE_CHUNKS=1500`, `MAX_SOURCE_BYTES=15MB`,
  `CLONE_TIMEOUT_SECONDS=60`) have been in place since Phase 2 and were
  not revisited here; confirm they're still the right numbers before a
  real public launch.

## Verification for this phase

Visit the public Space URL from a fresh browser session (not logged in as
the owner), paste a small public repo, ask a question, and confirm a
cited, grounded answer appears with the comparison view visible.
*Not performed* — no live Space exists yet (see `DECISIONS.md`'s Phase 8
entry: deploying a real public Space is a real, external, credentialed
action, deliberately not taken without explicit user confirmation,
independent of the "don't run the model" constraint). All other
verification (unit tests for every new module, `ruff check .`,
`pytest -q`) is already green with zero regressions to the Phase 1-7
suites.
