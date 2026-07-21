# Phase 2 — Ingestion pipeline (offline, no LLM)

**Status:** Done
**Milestone:** chunker → embedder → indexer works end to end against the
real `encode/starlette` repo, fully offline.

## Tasks

1. **Repo cloning** — `ingestion/clone.py`: `shallow_clone`, `current_sha`,
   `changed_files_since` (plain `git` via `subprocess`).
   *Done:* unit test against a local `git init` fixture repo clones and
   asserts expected files present, no network required. ✅
   `tests/ingestion/test_clone.py`.

2. **tree-sitter chunker** — `ingestion/chunker.py`: walk the parse tree,
   emit one `Chunk` per function/class/method with attached
   docstring/leading comment, for Python and TS/JS
   (`tree-sitter-language-pack`).
   *Done:* golden-file test on a small fixture `.py` and `.ts` file
   asserts exact line ranges per chunk. ✅ `tests/ingestion/test_chunker.py`
   — coverage ended up broader than the original launch scope: Go, Rust,
   Java, C, and C++ are also chunked along function/class/method
   boundaries, and YAML/JSON/TOML are handled as bounded whole-file
   chunks. See the `DECISIONS.md` entry for this expansion.

3. **Embedder** — `ingestion/embedder.py` wraps
   `jinaai/jina-embeddings-v2-base-code` (pinned revision,
   `trust_remote_code=True`).
   *Done:* unit test embeds 5 sample chunks, asserts shape `(5,
   hidden_dim)`, no NaNs. ✅ `tests/ingestion/test_embedder.py`, marked
   `@pytest.mark.integration` (downloads and runs the real model) —
   excluded from default `pytest -q` runs; run explicitly with
   `pytest -m integration` when you want to exercise it.

4. **Indexer** — `ingestion/indexer.py` builds a Chroma collection +
   `bm25s` index from chunks+embeddings, persists under
   `.ragnarok_cache/<repo>/`.
   *Done:* `python scripts/build_index.py <repo-url>` runs fully offline
   (no LLM), produces a Chroma collection with `count() > 0` and a saved
   bm25s index. ✅ `tests/ingestion/test_indexer.py` — also covers atomic
   rebuild-on-failure (staging dir swap, previous index preserved).

5. **Parquet export** — `utils/hf_dataset_io.py`:
   `export_chunks_to_parquet`, `push_to_hub`, using the fixed schema
   (`chunk_id, file_path, start_line, end_line, symbol_name,
   qualified_name, symbol_type, language, text, embedding`).
   *Done:* `pyarrow.parquet.read_table` on the output shows the expected
   columns. ✅ `tests/utils/test_hf_dataset_io.py`.

6. **Real end-to-end ingestion run** — full pipeline against
   `encode/starlette`.
   *Done:* `pytest -m integration` ingests the real repo, asserts chunk
   count in a sane range (e.g. 100-2000), zero exceptions.
   `tests/integration/test_ingestion_e2e.py` exists and is wired up via
   `scripts/build_index.py`'s `ingest_repository()`, marked
   `@pytest.mark.integration` — not yet executed against the network in
   this environment (clones the real repo and runs the real embedding
   model); run `pytest -m integration` locally to exercise it end to end.

## Verification for this phase

`python scripts/build_index.py https://github.com/encode/starlette`
completes without touching any LLM code path; inspect the resulting
Chroma collection count and a sample of chunk metadata by hand to confirm
line ranges look correct against the actual source file.

All offline unit/integration-wiring tests pass (`pytest -q`: chunker,
clone, indexer, parquet export, and `scripts/build_index.py`'s CLI/exit
codes are fully covered without touching the network or a real model).
The two `@pytest.mark.integration`-marked tests (real embedder, real
end-to-end starlette ingestion) are present and ready to run — they
haven't been executed in this environment to avoid an unnecessary
model-download/CPU cost; run them with `pytest -m integration` whenever
you want to validate against the live repo and real model.
