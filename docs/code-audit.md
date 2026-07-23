# Code Audit

Findings from a repository-wide review conducted 2026-07-23, before any changes were made.
Three parallel research passes covered: (1) top-level layout, packaging, and tooling config,
(2) `src/` module architecture, typing, and error handling, (3) the UI, agent, eval suite,
and test suite. Each finding below is tagged with the task (see
[`refactoring-decisions.md`](refactoring-decisions.md)) that addressed it, or **not addressed**
where it was explicitly descoped.

## Calibration note

This codebase was **already disciplined** going in, which shaped the scope of this audit:

- 100% of `src/` functions had parameter and return type annotations before any typing work
  began; no bare `Any`; containers were already parameterized.
- `ruff check .` (rules `E,F,I,UP`) reported zero issues at the start — no unused imports,
  no unused variables, no wildcard imports.
- No `TODO`/`FIXME`/stub/`NotImplementedError` placeholders anywhere in `src/`, `scripts/`,
  or `app.py`.
- No comment banners, no top-of-file essays; existing docstrings were already minimal, and
  the few multi-line ones encoded real invariants worth keeping as-is.
- No circular imports.

So this audit does not report a long list of syntax errors, dead code, or naming violations,
because there weren't any. It reports the real architectural, typing-enforcement, duplication,
and UX gaps found by reading every module.

## Correctness and reliability issues

| Finding | File(s) | Task |
|---|---|---|
| `pyproject.toml` declared `dependencies = []`; `pip install -e .` alone installed nothing, silently depending on also running `pip install -r requirements.txt`. | `pyproject.toml` | 2 |
| `scripts/build_index.py` caught `NotImplementedError` in its CLI exception handler; nothing in the codebase ever raises it — dead branch. | `scripts/build_index.py:530` (pre-fix) | 4 |
| `app.py`'s `_ingest` caught `Exception` from `ingest_repository` and returned a status string to the Gradio UI without logging anything — a failed ingestion in the deployed Space left no server-side trace. | `app.py:42` (pre-fix) | 8 |
| Uncommitted working-tree deletions of `DECISIONS.md` and all 11 `docs/phases/*.md` files, while `README.md`/`docs/architecture.md`/`docs/roadmap.md` still linked to them — an inconsistent repo state if left as-is. | `DECISIONS.md`, `docs/phases/`, `README.md`, `docs/architecture.md`, `docs/roadmap.md` | 1 |
| `docs/architecture.md`'s repository-structure diagram referenced `src/ui/layout.py` and `src/utils/logging.py`, neither of which exist, and omitted `scripts/compare_retrieval.py`. | `docs/architecture.md` | 1 |

## Typing issues

| Finding | File(s) | Task |
|---|---|---|
| No mypy config anywhere (`pyproject.toml` had no `[tool.mypy]`, no `mypy.ini`); mypy wasn't a pinned dependency; CI never ran it. The excellent existing annotation discipline was a convention, not an enforced gate. | repo-wide | 3, 11 |
| `chunk_from_chroma_record`'s parameter type (`Mapping[str, str \| int]`) was narrower than what Chroma's API actually returns (`Mapping[str, str \| int \| float \| bool]`), so every call site technically passed an incompatible type. | `src/retrieval/schemas.py` | 3 |
| `include=["documents", "metadatas", ...]` calls passed plain strings where chromadb's stub wants `IncludeEnum` members, and `query`/`get` results are typed `Optional` even though every call site always requests those fields. | `src/retrieval/dense.py`, `src/retrieval/sparse.py`, `src/agent/tools.py`, `scripts/build_index.py`, `scripts/push_index_to_hub.py` | 3 |
| `HfApi().dataset_info(...).sha` and llama-cpp's chat-completion message `content` are typed `Optional` but were used without narrowing. | `src/utils/hf_dataset_io.py`, `src/generation/cpu_backend.py` | 3 |
| `RagnarokRagasLLM.backend: LLMBackend \| None` was called (`.generate(...)`) without a None-check. | `src/eval/ragas_runner.py` | 3 |
| LangGraph node functions in `build_graph` returned bare `dict` instead of a type reflecting `AgentState`'s real fields; `follow_symbol_node` passed `state.pending_symbol` (`str \| None`) where `find_symbol` requires `str`, relying on an invariant enforced only by the graph's own routing logic. | `src/agent/graph.py` | 3 |
| Two missing return-type annotations: `app.py`'s `_demo_compare`, `scripts/run_eval.py`'s `run_pipeline_eval`. `app.py`'s `LRUCache` instance wasn't parameterized (`LRUCache` instead of `LRUCache[_ModelBundle]`). | `app.py`, `scripts/run_eval.py` | 3 |

## Architectural issues

| Finding | File(s) | Task |
|---|---|---|
| `src/baseline/naive_rag.py` contained not just the naive baseline (`answer_naive`) but the full production orchestration (`answer_ragnarok`, `compare`), and imported from `src.agent` to do it — a "baseline" package depending on the "agent" package is backwards layering, and the name no longer matched most of the module's contents. | `src/baseline/naive_rag.py` | 6 |
| The dense→sparse→RRF→rerank pipeline was hand-rolled three times: the canonical `src/agent/tools.py::retrieve_hybrid`, inline in `scripts/ask.py::answer_question`, inline in `scripts/compare_retrieval.py::compare_query`. | `scripts/ask.py`, `scripts/compare_retrieval.py` | 5 |
| `scripts/build_index.py` held real business logic (`_validate_cache_dir`, `_ensure_disjoint`, `_merge_incremental_chunks`) that belonged with the rest of the ingestion pipeline in `src/ingestion/`, not embedded in a CLI script. | `scripts/build_index.py` | 7 |
| `src/generation/zerogpu_backend.py` defined its own `ZEROGPU_MODEL_ID`/`ZEROGPU_MODEL_REVISION` module constants, inconsistent with every other backend, which pull model IDs/revisions from `src/config.py`. | `src/generation/zerogpu_backend.py` | 4 |
| The agent's hop-decision LLM call hardcoded `max_tokens=32` inline instead of a named `src/config.py` constant. | `src/agent/graph.py` | 3 (folded in), 4 |

## Error-handling issues

| Finding | File(s) | Task |
|---|---|---|
| `app.py::_ingest` swallowed the ingestion exception without logging (see Correctness table above). | `app.py` | 8 |
| Three call sites deliberately catch broad `Exception`/`BaseException` (per-file chunking guard, index-build staging-dir cleanup, cache-swap rollback) — all justified, but none had a comment explaining why, so a future reviewer could mistake them for oversights. | `src/ingestion/chunker.py`, `src/ingestion/indexer.py` | 8 |

## UI/UX issues

| Finding | File(s) | Task |
|---|---|---|
| `src/ui/hop_viz.py::render_hop_path` (clickable GitHub file/line links per hop) was fully implemented and unit-tested but never called from `compare_view.py`; the RAGnarok column only showed a comma-joined list of symbol names. | `src/ui/compare_view.py` | 9 |
| Citations were shown as a single aggregate "Verified citations" / "No citations found" line, never per-citation file/line spans or their individual verified status. | `src/ui/compare_view.py` | 9 |
| No empty-question guard: submitting a blank question round-tripped through the full retrieval+generation pipeline. | `src/ui/compare_view.py`, `src/ui/ingest_view.py` | 9 |
| No error handling around the compare/ingest callbacks: a backend failure surfaced as Gradio's generic, unstyled error toast instead of an inline, readable message. | `src/ui/compare_view.py` | 9 |
| No loading feedback specific to the operation in progress; CPU inference (compare) and clone+chunk+embed+index (ingest) can both take a long time with zero indication of what's happening. | `src/ui/compare_view.py`, `src/ui/ingest_view.py` | 9 |
| No theming — pure default Gradio look, no `gr.themes.*`, no `css=`. | `app.py` | 9 |

## Testing gaps

| Finding | File(s) | Task |
|---|---|---|
| `src/config.py` had no dedicated test file; a renamed or mis-valued constant had nothing to catch it. | — | 10 |
| No test coverage for the logging/repo-context helpers added during this session until they were added alongside the code. | `app.py` | 8, 9, 10 |

## Security or reliability issues

| Finding | File(s) | Task |
|---|---|---|
| `requirements.txt` pinned `langchain`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`, `python-dotenv`, which looked unused from source-level imports alone (only `langchain-core` is imported directly, in `src/eval/ragas_runner.py`). **Verified via `pip show` that all are real transitive dependencies of `ragas`/`langchain`/`uvicorn[standard]`** — none were pruned; see [`refactoring-decisions.md`](refactoring-decisions.md) Task 2. | `requirements.txt` | 2 (investigated, no change) |
| CI never ran `mypy` or a formatter check, only `ruff check .` and `pytest -q`; formatting drift in 13 files had gone undetected because nothing gated it. | `.github/workflows/ci.yml` | 3, 11 |

## Explicitly out of scope

- **Real-model verification** (a live RAGAS eval run, a live `app.py` smoke test with the real GGUF/embedding/rerank models, dispatching the HF Space deploy or the incremental-reindex GitHub Action) — descoped per user decision; see [`final-validation.md`](final-validation.md).
- **Connecting the Ingest tab's result to the Compare tab's repo selection** (Compare still always targets the fixed demo repo) — flagged during UI work (Task 9) as a larger behavioral change needing separate confirmation, not implemented here.
