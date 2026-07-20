# RAGnarok Architecture (living snapshot)

This is the current-state design doc — it describes how the system is
*meant* to fit together as of the latest phase completed. It is not a
changelog; for the reasoning behind each choice and its history, see
[`DECISIONS.md`](../DECISIONS.md). For task-by-task execution status, see
[`roadmap.md`](roadmap.md).

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
   end_line, symbol_name, symbol_type, language, text, embedding`) for
   pushing to a free HF Dataset repo.

2. **Retrieval** (`src/retrieval/`) — dense (Chroma) and sparse
   (`bm25s`) candidates are fused via Reciprocal Rank Fusion, deduped by
   `chunk_id`, then reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`
   (pinned revision) before anything reaches the LLM.

3. **Multi-hop agent** (`src/agent/`) — a LangGraph graph
   (`retrieve → decide_more_hops → follow_symbol (max 3 hops) →
   compose_answer`) can issue a follow-up retrieval for a symbol's
   definition in another file. The ordered hop path (file, symbol, line
   range, reason) is recorded as a serializable `HopRecord` list for UI
   visualization.

4. **Generation** (`src/generation/`) — backend-agnostic
   `generate(prompt, max_tokens)` interface with two implementations:
   `zerogpu_backend.py` (`Qwen/Qwen2.5-Coder-7B-Instruct` via
   `@spaces.GPU`) and `cpu_backend.py`
   (`Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` via `llama-cpp-python`,
   installed from the prebuilt CPU wheel index). ZeroGPU quota-exceeded
   errors are caught and silently fall back to the CPU backend; the UI
   surfaces which backend answered. Prompts mandate citing every claim and
   saying "I don't know" when context doesn't support an answer.

5. **Citation verification** (`src/generation/citations.py`) — every
   cited `file#Lstart-Lend` span is parsed out of the model's output and
   checked against the real chunk index before display; unverifiable
   citations are dropped or flagged, never shown as fact.

6. **Comparison mode** (`src/baseline/`, `src/ui/compare_view.py`) —
   a naive baseline (single dense top-k, no BM25/rerank/hops, same LLM)
   runs in parallel with the full RAGnarok pipeline on every query; the
   Gradio UI shows both answers side by side with a verified-citation
   badge on the RAGnarok column only.

7. **Evaluation** (`src/eval/`) — a hand-labeled 30-50 item Q&A set
   against starlette (single-hop, multi-hop, adversarial "not in this
   repo" items) is scored with RAGAS-style metrics (faithfulness, context
   precision/recall) using our own hosted model as the judge (a
   `BaseRagasLLM` subclass wrapping `generation/llm.py` — no external judge
   API), plus an ablation table comparing naive / hybrid+rerank / full
   multi-hop.

8. **Incremental indexing** (`.github/workflows/reindex.yml`,
   `scripts/build_index.py --incremental`) — a scheduled/dispatched CI job
   re-chunks and re-embeds only files changed since the last-indexed SHA,
   merges into the parquet index, and pushes to the HF Dataset repo; the
   Space compares its cached SHA against the Dataset repo's latest on
   startup and pulls fresh parquet if stale.

## Repository structure

```
ragnarok/
├── LICENSE
├── .gitignore
├── README.md                      # Phase 10
├── DECISIONS.md
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
│   ├── ui/                        # layout.py, compare_view.py, hop_viz.py
│   └── utils/                     # caching.py, hf_dataset_io.py, logging.py
│
├── tests/                         # mirrors package layout + fixtures/ + integration/
├── data/eval/                     # qa_starlette.jsonl + labeling rubric README
├── docs/
│   ├── roadmap.md
│   ├── architecture.md            # this file
│   └── phases/phase-01-scaffold.md … phase-10-polish.md
├── scripts/                       # build_index.py, ask.py, run_eval.py, push_index_to_hub.py
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

Full reasoning, alternatives considered, and trade-offs for each of these
live in [`DECISIONS.md`](../DECISIONS.md).

## Flagged operational risks and mitigations

1. **ZeroGPU free quota (~3.5 min/day, shared across every visitor)** →
   transparent fallback to CPU backend on quota-exceeded; UI shows which
   backend answered; comparison mode defaults to CPU for both columns.
2. **`llama-cpp-python` won't build from source in Spaces' free build
   environment in time** → install from the prebuilt CPU wheel index.
3. **Free CPU Spaces (2 vCPU/16GB RAM) shared across everything** → hard
   ingestion ceiling, LRU eviction on resident in-memory indexes, demo
   repo pre-warmed at startup.
4. **`jina-embeddings-v2-base-code` needs `trust_remote_code=True`** → pin
   an exact commit revision for every third-party model load.
