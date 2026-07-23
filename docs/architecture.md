# RAGnarok Architecture (living snapshot)

This is the current-state design doc — it describes how the system is
*meant* to fit together as of the latest phase completed. It is not a
changelog. For task-by-task execution status, see [`roadmap.md`](roadmap.md).

## What it does

Paste a public GitHub URL, ask a natural-language question about the code,
get a plain-language answer grounded in the actual repo, with every claim
cited to an exact `file#Lstart-Lend` span, and able to trace logic across
multiple files (multi-hop) rather than returning a single best-match
chunk. Every query also runs through a naive single-vector baseline so the
value of hybrid retrieval + reranking + multi-hop is shown side by side,
live, on the same query — not just asserted in prose.

Fixed demo/eval repo: **`encode/starlette`**.

## Pipeline

1. **Ingestion** (`src/ingestion/`) — shallow-clone the repo via
   `subprocess` git calls, parse source with `tree-sitter`
   (`tree-sitter-language-pack`, launch scope: Python + JavaScript/TypeScript),
   chunk along function/class boundaries so docstrings/signatures stay
   attached to bodies, embed each chunk with `jina-embeddings-v2-base-code`
   (pinned revision, `trust_remote_code=True`), and index into Chroma
   (dense) + `bm25s` (sparse). Chunks/embeddings/metadata are also
   exportable to a fixed parquet schema (`chunk_id, file_path, start_line,
   end_line, symbol_name, qualified_name, symbol_type, language, text,
   embedding`) for
   pushing to a free HF Dataset repo.

2. **Retrieval** (`src/retrieval/`) — `dense.py` queries the Chroma
   collection (created with `hnsw:space: cosine`) by embedding, `sparse.py`
   queries the persisted `bm25s` index and hydrates full chunk data back
   from Chroma by `chunk_id`, `hybrid.py` fuses both ranked lists via
   Reciprocal Rank Fusion (deduped by `chunk_id`), and `rerank.py`'s
   `Reranker` reorders the fused candidates with
   `cross-encoder/ms-marco-MiniLM-L-6-v2` (pinned revision) before
   anything reaches the LLM. Every stage returns `list[RetrievedChunk]`
   (a `Chunk` plus a relevance score, higher-is-better throughout).

3. **Multi-hop agent** (`src/agent/`) — `state.py`'s pydantic `AgentState`
   (question, retrieved context, `hop_path`, `hops_remaining`, answer) and
   `HopRecord` (file, symbol, line range, reason — JSON round-trippable for
   UI use). `tools.py` exposes `retrieve_hybrid` (the full Phase 3
   pipeline) and `find_symbol_definition` (exact-name Chroma metadata
   lookup, no embedding needed). `graph.py`'s `build_graph` wires a
   LangGraph `StateGraph(AgentState)`: `retrieve → decide_more_hops →
   follow_symbol → compose_answer`, looping `follow_symbol` back to
   `decide_more_hops` until either it declines another hop or
   `hops_remaining` (capped at `AGENT_MAX_HOPS`) hits zero. The hop-decision
   function and symbol resolver are injected dependencies, so the graph's
   control flow is testable without any real model call.

4. **Generation** (`src/generation/`) — `llm.py`'s `LLMBackend` is a
   `typing.Protocol` (`generate(prompt, max_tokens) -> str`) that every
   backend satisfies structurally. `cpu_backend.py`'s `CPUBackend` pulls
   `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` (Q4_K_M) at a pinned revision
   via `huggingface_hub.hf_hub_download` (not `Llama.from_pretrained`,
   which has no `revision` parameter) and runs it
   through `llama-cpp-python`, installed from the prebuilt CPU wheel
   index. `zerogpu_backend.py`'s `ZeroGPUBackend` runs the pinned
   `Qwen/Qwen2.5-Coder-7B-Instruct` on an allocated ZeroGPU worker (its
   `generate` method is `@spaces.GPU`-decorated). `fallback_backend.py`'s
   `FallbackBackend` wraps a primary (ZeroGPU) and fallback (CPU) backend,
   catching the `gr.Error` ZeroGPU raises on quota-exceeded and
   transparently retrying on the CPU backend; `last_backend_used` lets the
   UI surface which one answered. `prompts.py`'s `render_prompt` renders
   numbered context chunks with mandatory `file#Lstart-Lend` citation
   instructions and an explicit "say I don't know" instruction.

5. **Citation verification** (`src/generation/citations.py`) —
   `verify_citations(answer, chunks)` parses every `file#Lstart-Lend` span
   out of the model's output and marks each one verified only if it falls
   within some real chunk's file and line range; unverifiable citations
   are flagged (`Citation.verified = False`), never shown as fact.

6. **Comparison mode** (`src/baseline/`, `src/ui/compare_view.py`) —
   `baseline/naive_rag.py`'s `answer_naive` runs a single dense top-k
   retrieval with a plain prompt (no BM25, rerank, hops, or citation
   instructions), `answer_ragnarok` runs the full Phase 5 agent graph
   (`max_hops` overridable — `0` gives pure hybrid+rerank, used by the
   Phase 7 ablation), and `compare()` runs both (naive first) tagged as a
   `PipelineResult` (`pipeline`, `answer`, `citations`, `hop_path`,
   `context_texts`). `ui/compare_view.py`'s `build_compare_view` wires a
   two-column Gradio Blocks UI over an injected `compare_fn`;
   `render_comparison`/`_format_result` render each column, showing a
   "Verified citations"/hop-path badge only for the `ragnarok` column,
   never for `naive`. `ui/hop_viz.py`'s `render_hop_path` renders a hop
   path as markdown with clickable GitHub `blob` links (needs
   `repo_slug`/`repo_sha` from the index manifest) — implemented and unit
   tested standalone; not yet wired into `compare_view`'s rendering.
   `ui/ingest_view.py`'s `build_ingest_view` is a second Gradio Blocks tab
   (paste a repo URL → `scripts.build_index.ingest_repository` → status
   message), mounted in `app.py` alongside the compare view. `app.py`
   caches loaded `(Embedder, Reranker, CPUBackend)` per repo in an
   `LRUCache` (`utils/caching.py`, capacity `MAX_RESIDENT_REPOS`) instead
   of reloading them on every question.

7. **Evaluation** (`src/eval/`, `data/eval/qa_starlette.jsonl`) —
   `qa_dataset.py`'s `QAItem`/`load_qa_dataset` validate the 35-item
   hand-labeled starlette set (20 single-hop, 10 multi-hop, 5 adversarial
   "not in this repo" items — see `data/eval/README.md` for the labeling
   rubric). `ragas_runner.py`'s `RagnarokRagasLLM(BaseRagasLLM)` wraps our
   own `LLMBackend` as the RAGAS judge (no external judge API);
   `run_ragas_eval` scores faithfulness, context precision, and context
   recall. `ablation.py`'s `run_ablation` runs the same set through naive
   / hybrid+rerank (0 hops) / full multi-hop and returns one `AblationRow`
   per arm. `metrics.py`'s `idk_rate` checks what fraction of adversarial
   items get an explicit "I don't know." `scripts/run_eval.py` wires it
   all into a CLI (`--ablation` for the 3-row table).

8. **Incremental indexing** (`.github/workflows/reindex.yml`,
   `scripts/build_index.py --incremental`, `scripts/push_index_to_hub.py`) —
   `build_index.py`'s `_merge_incremental_chunks` uses
   `ingestion/clone.py`'s `changed_files_since` to re-chunk/re-embed only
   files changed since the cached manifest's `repo_sha`, reusing every
   other chunk's existing embedding straight from Chroma; `--incremental`
   requires `--repo-path` and an existing compatible index to diff
   against. `push_index_to_hub.py`'s `export_and_push` reads the built
   Chroma collection back out, writes it to our parquet schema, and pushes
   it to the HF Dataset repo. `utils/hf_dataset_io.py`'s
   `refresh_if_stale` (used by both the CI job and, eventually, the Space
   at startup) compares a local manifest's `repo_sha` against
   `latest_commit_sha` on the Dataset repo and, if stale,
   `pull_parquet`+`rehydrate_index_from_parquet` rebuild the local
   Chroma+bm25 index from parquet — never from Chroma's own on-disk
   format, per the Phase 0 decision.

## Repository structure

```
ragnarok/
├── LICENSE
├── .gitignore
├── README.md
├── pyproject.toml                 # hatchling, requires-python ">=3.11,<3.13"
├── requirements.txt
├── app.py                         # HF Space entrypoint — thin Gradio Blocks launcher
│
├── src/
│   ├── config.py                  # model IDs + pinned revisions, cache paths, caps
│   ├── ingestion/                 # clone.py, chunker.py, embedder.py, indexer.py, schemas.py
│   ├── retrieval/                 # dense.py, sparse.py, hybrid.py, rerank.py
│   ├── agent/                     # state.py, tools.py, graph.py
│   ├── generation/                # llm.py, zerogpu_backend.py, cpu_backend.py, prompts.py, citations.py
│   ├── baseline/                  # naive_rag.py
│   ├── eval/                      # qa_dataset.py, ragas_runner.py, ablation.py, metrics.py
│   ├── ui/                        # compare_view.py, hop_viz.py, ingest_view.py
│   └── utils/                     # caching.py, hf_dataset_io.py
│
├── tests/                         # mirrors package layout + fixtures/ + integration/
├── data/eval/                     # qa_starlette.jsonl + labeling rubric README
├── docs/
│   ├── roadmap.md
│   └── architecture.md            # this file
├── scripts/                       # build_index.py, ask.py, compare_retrieval.py, run_eval.py, push_index_to_hub.py
├── .github/workflows/             # ci.yml, reindex.yml
└── .ragnarok_cache/                # GITIGNORED: cloned repos + local Chroma/bm25s indexes
```

`data/eval/` (committed, hand-authored ground truth) is kept separate from
`.ragnarok_cache/` (gitignored, regenerable) on purpose.

## Key technology choices

| Concern | Choice | Why (short) |
|---|---|---|
| Language | Python 3.11 | Broadest prebuilt-wheel coverage across the whole stack |
| Vector store | Chroma | Needs metadata + `where`-filtering anyway; FAISS is a bare index blob |
| Sparse retrieval | `bm25s` | 100-500x faster than `rank-bm25`; matters for live per-repo indexing |
| Parsing | `tree-sitter` + `tree-sitter-language-pack` | MIT, vetted permissive grammars, zero manual grammar builds |
| Launch language scope | Python + JS/TS | Keeps chunk-boundary logic tractable before widening |
| UI | Gradio Blocks | `@spaces.GPU`/ZeroGPU is exclusively Gradio-compatible |
| Judge model (eval) | Our own hosted model via custom `BaseRagasLLM` | Avoids a paid external judge API |
| Repo cloning | plain `git` via `subprocess` | GitPython shells out to `git` anyway for our narrow needs |

## Flagged operational risks and mitigations

1. **ZeroGPU free quota (~3.5 min/day, shared across every visitor)** →
   `FallbackBackend` transparently retries on the CPU backend on a
   `gr.Error` (quota-exceeded); `last_backend_used` lets the UI show which
   backend answered; comparison mode uses the CPU backend for both
   columns.
2. **`llama-cpp-python` won't build from source in Spaces' free build
   environment in time** → install from the prebuilt CPU wheel index.
3. **Free CPU Spaces (2 vCPU/16GB RAM) shared across everything** → hard
   ingestion ceiling (`MAX_INDEXABLE_CHUNKS`/`MAX_SOURCE_BYTES`, enforced
   in `chunker.py` since Phase 2), `utils/caching.py`'s `LRUCache`
   (capacity `MAX_RESIDENT_REPOS`) evicting the oldest resident
   models/index once a new repo is ingested, demo repo pre-warmed at
   startup (`app.py`'s `_load_models`/`_demo_compare`).
4. **`jina-embeddings-v2-base-code` needs `trust_remote_code=True`** → pin
   an exact commit revision for every third-party model load — audited
   Phase 8: all five model loads (`Embedder`, `Reranker`, `CPUBackend`,
   `ZeroGPUBackend` ×2) pass an explicit `revision=`, grep-verified.
