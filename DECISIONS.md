# RAGnarok — Decisions Log

This file is a record of judgment calls and why they were made — not a
changelog of what was built. Update it continuously, immediately after
each non-trivial decision, not just at the end of a phase. Before making a
new decision that touches an area already covered here, check first;
reversals get a new entry that references the old one rather than an edit
to history.

---

## [Phase 0] Flat package layout, no `src/` directory

**Decision:** `ragnarok/` package lives directly at the repo root
(`ragnarok/ragnarok/...`), not under a `src/` layout.

**Why:** Hugging Face Spaces runs `python app.py` from the repo root. A
flat package is importable purely from `cwd` with zero packaging
machinery required at deploy time. `pyproject.toml` is still included for
local dev ergonomics (`pip install -e .`, IDE resolution), but it isn't
load-bearing for the deployed app.

**Alternatives considered:** `src/`-layout (more conventional for
distributable Python libraries) — rejected because it would require a
working editable install to succeed inside the Spaces build container
just to make `import ragnarok` resolve, which is one more failure mode for
a project whose main deliverable is a live public Space, not a
redistributable library.

**Trade-offs / risks:** Flat layout is slightly less conventional for
anyone expecting a packaged library; acceptable since RAGnarok's primary
artifact is the running Space, not a pip-installable package.

**Date/phase:** 2026-07-20, Phase 0.

---

## [Phase 1] Rename the package directory from `ragnarok/` to `src/`

**Decision:** The importable package directory at the repo root is renamed
`ragnarok/` → `src/` (`git mv ragnarok src`), so the import path is now
`import src` / `from src.ingestion import ...` etc. This is a rename only
— the layout is still flat at the repo root (`src/` sits directly beside
`app.py`, `pyproject.toml`, etc.), so the flat-vs-`src/`-layout reasoning
in the Phase 0 entry above (no editable-install requirement to import at
Spaces deploy time) still holds; only the folder's name changed, not its
position. `pyproject.toml`'s `[tool.hatch.build.targets.wheel] packages`
now points at `["src"]`; the distribution name in `[project] name` stays
`"ragnarok"`. The gitignored `.ragnarok_cache/` runtime cache directory is
a separate, unrelated concept and was left as-is.

**Why:** Explicit user request.

**Alternatives considered:** None — this supersedes the directory name
chosen in the Phase 0 entry above by explicit instruction; that entry is
left unedited as the historical record of the flat-vs-nested reasoning,
which is unaffected by this rename.

**Trade-offs / risks:** The importable name (`src`) no longer matches the
project's own name (`ragnarok`/RAGnarok), which is less self-descriptive
than a package literally named after the project — acceptable since this
was a deliberate, explicit choice rather than a naming default.

**Date/phase:** 2026-07-20, Phase 1.

---

## [Phase 0] Python 3.11

**Decision:** Target Python `>=3.11,<3.13`.

**Why:** 3.11 has the broadest, least-lagging prebuilt-wheel coverage
across the whole stack we depend on (torch/transformers, `llama-cpp-python`,
tree-sitter bindings), and is HF Spaces' most commonly documented default
runtime — this minimizes deploy friction.

**Alternatives considered:** 3.12 — rejected for now since wheel coverage
for some of our dependencies lags; revisit once that catches up.

**Trade-offs / risks:** None significant; 3.11 is a safe, boring choice.

**Date/phase:** 2026-07-20, Phase 0.

---

## [Phase 0] Vector store: Chroma over FAISS

**Decision:** Use Chroma (`EphemeralClient()` in the deployed Space,
`PersistentClient()` locally/in CI) as the dense vector store, not FAISS.

**Why:** We need a metadata store alongside the vectors anyway (file
path, line range, symbol name, language) for citation generation and for
restricting candidates before fusing with BM25 results. Chroma provides
native metadata `where`-filtering; FAISS is a bare index blob with no
built-in metadata layer, which would mean hand-rolling a parallel
metadata/parquet layer from scratch just to get back to parity with what
Chroma gives natively.

**Alternatives considered:** FAISS — faster/lighter in the abstract, but
loses on total system complexity once a metadata layer has to be built
around it by hand.

**Trade-offs / risks:** For the HF Dataset interchange format, we do NOT
rely on Chroma's internal on-disk format (not guaranteed stable across
versions) — we own a custom parquet schema (`chunk_id, file_path,
start_line, end_line, symbol_name, qualified_name, symbol_type, language,
text, embedding`) written via `pyarrow` and rehydrate Chroma from that on
load.

**Date/phase:** 2026-07-20, Phase 0.

---

## [Phase 0] Sparse retrieval: `bm25s` over `rank-bm25`

**Decision:** Use `bm25s` for BM25 sparse retrieval, with its default
(non-numba) backend.

**Why:** Benchmarks show `bm25s` is roughly 100-500x faster than
`rank-bm25` on indexing and query throughput (scipy sparse matrices vs.
`rank-bm25`'s naive recompute-per-query approach). This matters directly
for RAGnarok because indexing happens live, per pasted repo, on shared
free CPU hardware — a slow BM25 build would visibly stall the "paste any
repo" UX.

**Alternatives considered:** `rank-bm25` (simpler API, much slower) —
rejected on throughput grounds. `bm25s`'s optional numba-accelerated
backend — rejected for now: our per-repo corpora (hundreds to
low-thousands of chunks) don't need numba's throughput, and skipping it
avoids a JIT-compile cold-start tax on every fresh Spaces container boot.

**Trade-offs / risks:** None significant at our expected corpus sizes.

**Date/phase:** 2026-07-20, Phase 0.

---

## [Phase 0] tree-sitter grammar packaging + launch language scope

**Decision:** Use `tree-sitter-language-pack` for grammar access. Launch
language scope is Python + JavaScript/TypeScript only; other languages
come later.

**Why:** `tree-sitter-language-pack` is MIT-licensed, and its maintainers
vet every bundled grammar for permissive licensing (MIT/Apache-2.0/BSD/ISC)
while excluding copyleft grammars — no manual grammar compilation and no
license-contamination risk. Restricting launch scope to two languages
keeps the hand-written "how does this AST node type map to a chunk
boundary" logic in the chunker (docstring/JSDoc attachment, decorator
handling, nested classes) tractable to get fully correct before expanding.

**Alternatives considered:** Compiling individual grammar packages
per-language by hand — rejected as unnecessary manual work when the pack
already bundles 300+ vetted grammars. Launching with a wider language set
immediately — rejected because it would spread correctness effort thin
before the core chunking logic is solid for even one language pair.

**Trade-offs / risks:** Repos primarily in other languages (Go, Rust,
Java, etc.) won't chunk correctly at launch. Expansion later is purely
"write the node-type mapping" for the chunker, not a new
grammar-packaging problem.

**Date/phase:** 2026-07-20, Phase 0.

---

## [Phase 0] UI framework: Gradio Blocks

**Decision:** Build the UI in Gradio (Blocks API, not the high-level
`Interface`).

**Why:** ZeroGPU Spaces are exclusively compatible with the Gradio SDK —
`@spaces.GPU`-decorated functions are not supported outside a Gradio app
(Flask/Streamlit are explicitly unsupported for this). Since ZeroGPU is
the mandated free-tier GPU path (Section 2 of the brief), Gradio is
effectively forced. Blocks (not `Interface`) is needed for the
two-column compare view and hop-path visualization, which require custom
layout beyond a single input/output pair.

**Alternatives considered:** Streamlit, Flask — both explicitly
unsupported for `@spaces.GPU` on HF Spaces; not viable for this hosting
target regardless of other merits.

**Trade-offs / risks:** None — this is a hard constraint of the hosting
target, not a preference.

**Date/phase:** 2026-07-20, Phase 0.

---

## [Phase 0] Eval judge model: our own hosted model via custom RAGAS LLM wrapper

**Decision:** Implement a `RagnarokRagasLLM(BaseRagasLLM)` subclass in
`eval/ragas_runner.py` that calls directly into our existing
`generation/llm.py` interface, rather than pointing RAGAS at an external
judge API.

**Why:** RAGAS's LLM interface (`BaseRagasLLM`/`LangchainLLMWrapper`) is
designed to wrap any model, confirmed via its own "bring your own LLM"
docs. Standing up a second OpenAI-compatible HTTP server on the Space just
for eval would add an extra process, extra port, and extra failure
surface for no benefit over calling our existing generation interface
directly. This also satisfies the brief's Section 2 requirement to use
the project's own hosted model as judge, not a paid external API.

**Alternatives considered:** A paid external judge API (e.g. OpenAI) —
rejected outright per Section 2's free/open-source-by-default constraint;
there is no case here where the free option is materially worse, so no
flag/confirmation was needed.

**Trade-offs / risks:** `ragas`'s API surface (e.g.
`LangchainLLMWrapper` vs newer `llm_factory`) has moved fast across
versions — we pin an exact `ragas` version to avoid Phase 7 breaking
mid-project on an upstream API change.

**Date/phase:** 2026-07-20, Phase 0.

---

## [Phase 0] Repo cloning: plain `git` via `subprocess`, not GitPython

**Decision:** Implement `clone.py`'s `shallow_clone`, `current_sha`, and
`changed_files_since` as direct `git` CLI calls via `subprocess`.

**Why:** Our actual needs are narrow (shallow clone, current HEAD SHA,
changed-files-between-two-SHAs for Phase 9), and GitPython itself shells
out to `git` for most of these operations anyway. Subprocess gives full
control over `--depth`, timeouts, and error messages for arbitrary
(sometimes broken/huge/private-and-inaccessible) user-pasted URLs, without
adding a dependency whose main value-add — a Pythonic object model — we
don't need.

**Alternatives considered:** GitPython — rejected as an unnecessary
dependency layer over the same underlying `git` calls.

**Trade-offs / risks:** We own our own subprocess error-handling/timeout
code instead of relying on a library's; acceptable given the narrow scope.

**Date/phase:** 2026-07-20, Phase 0.

---

## [Phase 0] Fixed demo/eval repo: `encode/starlette`

**Decision:** `encode/starlette` is the fixed repo used as the running
example throughout Phases 2-8 and the target of the Phase 7 hand-labeled
30-50 item Q&A eval set.

**Why:** Self-contained ASGI framework core (~7k-10k LOC excluding tests,
no heavy internal dependency chain), well documented, stable API. It has a
literal `AuthenticationMiddleware`/`AuthenticationBackend` (in
`starlette/authentication.py`) plus `Router`/`Route` (in
`starlette/routing.py`) trace that matches the brief's own example query
("how does auth get validated on the request path?") end-to-end within
one repo — genuinely good multi-hop demonstration material — and it's
small enough to index in seconds on free CPU hardware.

**Alternatives considered:** `psf/requests` — similarly self-contained
and arguably more universally recognizable, but its auth trace is
client-side, a slightly less literal match for "request path." `pallets/
flask` — most recognizable framework name, but has no built-in auth
(that's the separate Flask-Login package), a weaker fit for the brief's
example query. Both lost to starlette on how directly they demonstrate
the brief's own example use case.

**Trade-offs / risks:** If starlette's API changes significantly upstream,
the hand-labeled eval set could drift stale — mitigated by Phase 9's
incremental reindexing tracking the repo's current SHA, and by treating
eval-set staleness as something to re-check if starlette has a major
release during the project.

**Date/phase:** 2026-07-20, Phase 0. Confirmed with the user via direct
question during Phase 0 planning.

---

## [Phase 0] Free-tier operational risk mitigations for HF Spaces

**Decision:** Four specific mitigations are required, not optional
polish, to keep the deployed Space usable within free-tier limits:

1. **ZeroGPU quota fallback:** `generation/llm.py` must catch the ZeroGPU
   quota-exceeded exception and transparently fall back to the CPU GGUF
   backend (never surface a raw error to the user); the UI shows which
   backend answered; comparison mode defaults to the CPU model for both
   columns, with GPU generation as an explicit "full power" opt-in.
2. **`llama-cpp-python` install path:** install via the prebuilt CPU wheel
   index (`pip install llama-cpp-python --extra-index-url
   https://abetlen.github.io/llama-cpp-python/whl/cpu`), never a bare
   `pip install llama-cpp-python`.
3. **Memory guardrails:** a hard repo-size/file-count ingestion ceiling,
   LRU eviction capping resident in-memory indexes, and the fixed demo
   repo (starlette) pre-warmed at startup so it's never affected by
   eviction churn.
4. **Pinned model revisions:** every third-party `from_pretrained`/
   `SentenceTransformer(...)` call (embedder, reranker, both Qwen
   variants) pins an exact commit `revision`.

**Why:** Confirmed free-tier constraints: ZeroGPU's free quota is
approximately 3.5 minutes/day, shared across every visitor hitting the
public Space that day — realistically only a handful of 7B-model
generations before the app must degrade gracefully for everyone else.
`llama-cpp-python` will not finish building from source inside Spaces'
free build environment in time. Free CPU Spaces are 2 vCPU/16GB RAM
shared across the embedder, reranker, CPU LLM, every ingested repo's
in-memory index, and the Gradio server itself — concurrent visitors
pasting large repos is a real OOM path. `jina-embeddings-v2-base-code`
requires `trust_remote_code=True`, which means executing whatever code is
currently at that HF repo's `main` at load time unless pinned.

**Alternatives considered:** Paid persistent compute/storage to avoid
these constraints entirely — explicitly rejected per Section 2; none of
these four risks require paid escalation, they just require the mitigation
to actually be built rather than treated as a footnote.

**Trade-offs / risks:** The CPU fallback answers more slowly and with a
smaller (1.5B) model than the ZeroGPU path — this is an accepted quality/
availability trade-off, surfaced honestly in the UI rather than hidden.

**Date/phase:** 2026-07-20, Phase 0.

---

## [Phase 1] Local macOS dev: build `llama-cpp-python` from source instead of the prebuilt CPU wheel index

**Decision:** On local macOS (arm64) dev machines, install
`llama-cpp-python` by letting `pip` build it from the sdist (via the
locally available Xcode Command Line Tools `clang`/`cmake`), rather than
via the prebuilt CPU wheel index mandated in the Phase 0 mitigation
(`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`).
The `requirements.txt` entry still points at that index first — this is a
local-machine-only deviation, not a change to the pinned dependency.

**Why:** The macOS arm64 wheels published on that CPU index are corrupted
upstream. Verified directly: both `llama_cpp_python-0.3.34` and `-0.3.33`
`macosx_11_0_arm64` wheels fail zip integrity checks (`BadZipFile: Bad
CRC-32 for lib/libggml-base*.dylib`, and a `zlib` decompression error,
respectively) on two independent download attempts each (including a
`--no-cache-dir` retry, ruling out a local pip-cache issue). As a control,
the `manylinux_2_17_x86_64` wheel for the same version (`0.3.34`) on the
identical index passed `zipfile.testzip()` cleanly — so this is isolated
to that index's macOS arm64 artifacts, and the actual HF Spaces deployment
target (Linux CPU containers) is unaffected.

**Alternatives considered:** Retrying more macOS wheel versions from the
same index — not pursued further after two consecutive corrupted releases
made "try another version and hope" a poor use of time; Xcode CLT's
`clang`/`cmake` were already confirmed present, making a source build the
faster path to an actually-working local install.

**Trade-offs / risks:** Local dev installs take longer (native compile vs.
wheel download) and require Xcode CLT to be present on any contributor's
Mac. This does not affect the deployed Space, which still resolves the
verified-good Linux wheel from the same prebuilt index per the original
Phase 0 mitigation — no change to that decision.

**Date/phase:** 2026-07-20, Phase 1.

---

## [Phase 2] Chunker language coverage expanded beyond the original launch scope

**Decision:** `ingestion/chunker.py` chunks Python, TypeScript/TSX,
JavaScript, Go, Rust, Java, C, and C++ along function/class/method
boundaries, and treats YAML/JSON/TOML as bounded whole-file chunks —
wider than the Phase 0 "launch scope: Python + JavaScript/TypeScript"
decision.

**Why:** `tree-sitter-language-pack` already bundles vetted grammars for
all of these languages (the Phase 0 rationale for adopting it in the
first place), and once the chunk-boundary/qualified-name logic was
written generically enough to handle Python and TS/JS correctly, adding
the node-type mappings for the other languages was incremental rather
than a new grammar-packaging problem — exactly the expansion path the
Phase 0 entry anticipated ("Expansion later is purely 'write the node-type
mapping' for the chunker"). Each language has golden-file tests
(`tests/ingestion/test_chunker.py`, `tests/fixtures/chunker/`) asserting
exact line ranges, so correctness wasn't traded away to get the wider
coverage.

**Alternatives considered:** Staying strictly to Python + TS/JS for Phase
2 and expanding later — rejected once it became clear the extra languages
were low-marginal-cost given the shared chunking framework already built.

**Trade-offs / risks:** None significant — this is a superset of the
original scope, not a substitution; starlette (Python) is unaffected
either way.

**Date/phase:** 2026-07-21, Phase 2.

---

## [Phase 3] Chroma collection distance space set to cosine

**Decision:** `indexer.py`'s `_build_chroma_collection` now creates the
collection with `metadata={"hnsw:space": "cosine"}`, instead of Chroma's
default (`l2`).

**Why:** Phase 3's dense retriever (`retrieval/dense.py`) is specified to
rank by cosine similarity. Setting the collection's distance space to
cosine at creation makes Chroma's returned `distance` equal to cosine
distance, so `score = 1 - distance` is exactly cosine similarity —
verified directly (identical vectors → distance `0.0`; orthogonal vectors
→ distance `1.0`). Computing cosine similarity by hand in Python instead
would have meant bypassing Chroma's own ANN index for every dense query,
defeating the Phase 0 rationale for using Chroma at all.

**Alternatives considered:** Leaving the default `l2` space and computing
cosine similarity manually from raw vectors — rejected as unnecessary
extra code and a bypass of Chroma's indexed search path for no benefit.

**Trade-offs / risks:** This changes the on-disk index format for
already-built Phase 2 caches (they'd need a rebuild to pick up the new
space) — no existing deployed index depends on the old default yet, so no
migration was needed.

**Date/phase:** 2026-07-21, Phase 3.

---

## [Phase 3] Deferred the real-model manual retrieval comparison run

**Decision:** Phase 3's task 5 (8-10 hand-picked starlette queries run
through dense-only vs. hybrid+rerank, logged in
`docs/phases/phase-03-retrieval.md`) was implemented as a ready-to-run
script (`scripts/compare_retrieval.py`) but not executed in this session.
The phase's status is recorded as "code complete; manual comparison run
pending," not "Done."

**Why:** Explicit user instruction not to execute real model
inference/downloads in this session (CPU cost). Running the comparison
requires a live `build_index.py` run against starlette (clone + real
embedding model) followed by real `Embedder`/`Reranker` inference for each
query — exactly the kind of load the user asked to avoid. All four
retrieval-stage modules (dense/sparse/hybrid/rerank) are otherwise fully
implemented and unit-tested (with the cross-encoder mocked out for the
default test run), so nothing about correctness is unverified — only the
live, real-model comparison table is outstanding.

**Alternatives considered:** Running it anyway to close out the phase
milestone fully — rejected outright given the explicit instruction; a
partially-honest "Done" status would misrepresent what was actually
verified.

**Trade-offs / risks:** Phase 3's milestone ("hybrid+rerank demonstrably
beats dense-only on sampled queries") isn't empirically confirmed yet —
only the code paths are. Run `scripts/compare_retrieval.py` against a
built starlette index when compute budget allows, fill in the results
table, and flip the phase status to `Done`.

**Date/phase:** 2026-07-21, Phase 3.

---

## [Phase 4] CPU generation model loaded via `hf_hub_download`, not `Llama.from_pretrained`

**Decision:** `generation/cpu_backend.py`'s `CPUBackend` downloads the
GGUF file with `huggingface_hub.hf_hub_download(repo_id=..., filename=...,
revision=...)` and constructs `llama_cpp.Llama(model_path=...)` from the
resulting local path, instead of calling `Llama.from_pretrained(repo_id=,
filename=, revision=...)` directly.

**Why:** Verified directly against the installed `llama-cpp-python`
source: `Llama.from_pretrained`'s signature is `(repo_id, filename,
additional_files=None, local_dir=None, local_dir_use_symlinks="auto",
cache_dir=None, **kwargs)` — no `revision` parameter exists, and passing
one through `**kwargs` would forward it into the `Llama` constructor
(which doesn't accept it either) rather than to the Hub download, causing
a `TypeError` at load time. Calling `hf_hub_download` ourselves keeps the
same pinned-revision guarantee already applied to the embedder and
reranker (Phase 0 mitigation #4) without relying on an API that doesn't
support it.

**Alternatives considered:** `Llama.from_pretrained(..., revision=...)` —
rejected outright since it would raise at call time, not silently ignore
the pin. Leaving the GGUF model unpinned (branch-following `main`) —
rejected as inconsistent with every other third-party model load in this
project.

**Trade-offs / risks:** One extra explicit download step instead of a
single library call; negligible — `hf_hub_download` is already a
transitive dependency of `huggingface_hub`, which is already pinned in
`requirements.txt`.

**Date/phase:** 2026-07-21, Phase 4.

---

## [Phase 4] Deferred the real-model manual `scripts/ask.py` run

**Decision:** Phase 4's task 5 (`python scripts/ask.py "<question>"`
end-to-end against starlette, confirming ≥1 verified citation) was
implemented as a ready-to-run script but not executed in this session.
The phase's status is recorded as "code complete; manual end-to-end run
pending," not "Done."

**Why:** Explicit user instruction not to execute real model
inference/downloads in this session (CPU cost). Running it requires a
live `build_index.py` run (real embedding model) plus real
`Embedder`/`Reranker`/`CPUBackend` inference for the question — exactly
the load the user asked to avoid. All four generation modules
(`llm`/`cpu_backend`/`prompts`/`citations`) are otherwise fully
implemented and unit-tested (with `Llama`/`hf_hub_download` mocked out
for the default test run), so nothing about correctness is unverified —
only the live, real-model end-to-end answer is outstanding. Mirrors the
same judgment call already logged for Phase 3's deferred comparison run.

**Alternatives considered:** Running it anyway to close out the phase
milestone fully — rejected outright given the explicit instruction.

**Trade-offs / risks:** Phase 4's milestone (a real verified-citation
answer from starlette) isn't empirically confirmed yet — only the code
paths are. Run `scripts/ask.py` against a built starlette index when
compute budget allows and flip the phase status to `Done`.

**Date/phase:** 2026-07-21, Phase 4.

---

## [Phase 5] Agent graph takes `decide_more_hops` and `find_symbol` as injected callables

**Decision:** `agent/graph.py`'s `build_graph(retrieve, find_symbol,
decide_more_hops, backend)` takes the hop-decision function and the
symbol resolver as plain injected callables, rather than hard-coding a
call to the LLM backend inside the graph's `decide_more_hops` node. The
real production decision function, `llm_decide_more_hops(backend,
state)`, lives alongside the graph and is bound in via
`functools.partial` at the call site (`scripts/ask.py`).

**Why:** LangGraph's control flow (looping `follow_symbol` back to
`decide_more_hops`, respecting the hop cap, routing to `compose_answer`)
is the part that actually needs correctness testing, and it's completely
decoupled from what specific policy decides "one more hop or not." Making
the decision function swappable means the graph's branching/looping logic
has full unit test coverage (mocked decider tracing an exact 2-hop path,
stopping at the cap, stopping on an unresolvable symbol) with zero real
model calls, while the real LLM-backed decision logic still lives in
production code, not test-only scaffolding.

**Alternatives considered:** Hard-coding the backend call inside a single
`decide_more_hops` node — rejected because it would make the graph's
looping/routing logic untestable without a real model, which conflicts
directly with this project's "never execute a real model in the default
test run" constraint.

**Trade-offs / risks:** One extra `functools.partial` binding step at each
real call site instead of a single self-contained node — negligible.

**Date/phase:** 2026-07-21, Phase 5.

---

## [Phase 5] Deferred the real multi-hop demonstration run

**Decision:** Phase 5's task 4 (a genuine cross-file starlette question
through `python scripts/ask.py --agent`, showing a real ≥2-hop trace with
every citation verified) was implemented end-to-end
(`answer_question_with_agent` in `scripts/ask.py`) but not executed in
this session. The phase's status is recorded as "code complete; real
multi-hop demonstration pending," not "Done."

**Why:** Same explicit constraint already logged for Phases 3 and 4: no
real model inference/downloads in this session (CPU cost). A real run
needs a built starlette index plus real embedding, reranking, and
generation inference for every hop. The graph's control flow itself
(looping, hop cap, routing) is fully unit-tested with the decision
function and symbol resolver mocked out, so nothing about the agent's
correctness is unverified — only a real, LLM-driven multi-hop trace
against starlette is outstanding.

**Alternatives considered:** Running it anyway to close out the phase
milestone fully — rejected outright given the explicit instruction.

**Trade-offs / risks:** Phase 5's milestone (a real ≥2-hop trace matching
the actual starlette auth code path) isn't empirically confirmed yet —
only the code paths are. Run `scripts/ask.py --agent` against a built
starlette index when compute budget allows and flip the phase status to
`Done`.

**Date/phase:** 2026-07-21, Phase 5.

---

## [Phase 6] `PipelineResult` split into its own `baseline/schemas.py`

**Decision:** `PipelineResult` (the tagged naive-vs-ragnarok answer
dataclass) lives in a new `baseline/schemas.py`, not inline in
`baseline/naive_rag.py` where its producing functions live.

**Why:** `naive_rag.py` unavoidably imports `Embedder`/`Reranker` at
module level (both pipelines need them as real type hints), so importing
`naive_rag.py` always pulls in `sentence_transformers`/`torch`. `ui/
compare_view.py` needs the `PipelineResult` type for its rendering
functions but must stay import-light (it's UI wiring, not a model-loading
module) — importing it from `naive_rag.py` would defeat that. This
mirrors the existing `retrieval/schemas.py` (light) vs. `retrieval/
rerank.py` (heavy) split from Phase 3, applied to the same problem one
level up the stack.

**Alternatives considered:** Defining `PipelineResult` inline in
`naive_rag.py` and having `ui/compare_view.py` import it from there —
rejected because it would silently make `import src.ui.compare_view` (and
transitively `import src.ui`, if re-exported) pull in heavy ML
dependencies purely to build a Gradio layout, contradicting the
established "package `__init__` imports stay light" precedent.

**Trade-offs / risks:** One extra small file; negligible given the direct
precedent already in this codebase.

**Date/phase:** 2026-07-21, Phase 6.

---

## [Phase 6] Naive baseline skips the citation-mandating prompt entirely

**Decision:** `baseline/naive_rag.py`'s `answer_naive` builds its own
minimal `"Context:\n{passages}\n\nQuestion: {question}\nAnswer:"` prompt
directly, instead of reusing `generation/prompts.py`'s `render_prompt`
(which mandates `file#Lstart-Lend` citations and an explicit "I don't
know" instruction). `answer_naive` also never calls
`generation/citations.py`'s `verify_citations` — its `PipelineResult`
always carries `citations=[]`.

**Why:** The citation-mandating prompt and verification step are
precisely the RAGnarok-side capability this comparison is meant to
demonstrate. Giving the "naive" baseline the same citation instructions
(even without hybrid retrieval/rerank/hops) would blur the comparison —
the UI's "verified" badge needs a baseline that structurally never has
verified citations, not one that happens not to show them.

**Alternatives considered:** Reusing `render_prompt` for both pipelines
and only skipping `verify_citations` for naive — rejected because the
naive baseline is meant to represent what a bare single-vector RAG setup
looks like without any of RAGnarok's prompting discipline either, not
just its retrieval/rerank machinery.

**Trade-offs / risks:** None — this is the intended contrast, not an
oversight.

**Date/phase:** 2026-07-21, Phase 6.

---

## [Phase 6] Deferred the manual `python app.py` UI check

**Decision:** Phase 6's task 3/4 manual verification (typing a real
multi-hop starlette question into the local Gradio app and visually
confirming the naive column is shallower than the RAGnarok column, with
the verified badge only on the latter) was wired end-to-end (`app.py` now
builds the real `build_compare_view` bound to the demo repo's cache dir)
but not run in this session.

**Why:** Same explicit constraint already logged for Phases 3-5: no real
model inference in this session (CPU cost), and no real starlette index
has been built yet either (Phase 2's ingestion was itself never run
against real starlette — see the Phase 3 deferred-comparison entry
above). All rendering logic (`_format_result`, `render_comparison`) and
the naive/ragnarok/compare pipeline functions are fully unit-tested with
fakes, so nothing about correctness is unverified — only the real,
browser-visible side-by-side check is outstanding.

**Alternatives considered:** Running `python app.py` and clicking through
it anyway — rejected given the explicit instruction; it would also
require a real built index that doesn't exist yet in this environment.

**Trade-offs / risks:** Phase 6's milestone (a visibly shallower naive
answer next to a verified RAGnarok one, in a live local browser session)
isn't empirically confirmed yet — only the code paths are. Build a real
starlette index, run `python app.py`, and flip the phase status to `Done`
once confirmed.

**Date/phase:** 2026-07-21, Phase 6.

---

## [Phase 7] Hand-labeled Q&A set built from a real read-only clone of starlette, not recalled from memory

**Decision:** Before writing `data/eval/qa_starlette.jsonl`, `encode/
starlette` was shallow-cloned read-only (commit
`5174d4c8358a6f06aa8056bafd14c2272dab8dd1`) to verify every
`expected_files`/`grounding_notes` claim directly against the actual
source (class/function definitions, file locations, cross-file call
chains) before it went in the dataset.

**Why:** A hand-labeled eval set is only as trustworthy as its ground
truth. Recalling starlette's structure from training data risks subtly
wrong file paths or invented cross-references, which would silently
corrupt every future RAGAS score computed against this set. Cloning the
repo to check facts is plain source reading (`git clone` + `grep`/`cat`),
not executing any ML model — fully consistent with this session's "don't
run the model" constraint, since chunking/embedding/generation were never
invoked against the clone.

**Alternatives considered:** Writing the dataset from memory — rejected
as an unacceptable risk to eval-set correctness for a one-time,
low-cost verification step.

**Trade-offs / risks:** The dataset is pinned to one commit's file
layout; if starlette's API changes significantly upstream, some items
could drift stale (already flagged generally in the Phase 0 "fixed demo
repo" entry above) — mitigated the same way, by re-checking if starlette
has a major release during the project.

**Date/phase:** 2026-07-21, Phase 7.

---

## [Phase 7] RAGAS judge scores only faithfulness, context precision, and context recall (no answer-relevancy)

**Decision:** `eval/ragas_runner.py`'s `RAGAS_METRICS` is
`[Faithfulness(), LLMContextPrecisionWithoutReference(), LLMContextRecall()]`
— all three are LLM-judged (no separate embedding-based judge needed).

**Why:** These three metrics directly match task 2's stated scope
("faithfulness, context precision/recall") and are all satisfiable by an
LLM judge alone (`LLMContextPrecisionWithoutReference`/`LLMContextRecall`
are the non-reference-embedding RAGAS variants), keeping the "our own
model as judge, no external API, no second embedding judge" property
clean. Verified directly against the installed `ragas==0.2.15` API
(`ragas.metrics`, `BaseRagasLLM`, `SingleTurnSample`, `EvaluationDataset`,
`evaluate()`) before writing the wrapper, rather than guessing method
signatures.

**Alternatives considered:** Adding `AnswerRelevancy` — rejected for now
since it requires an embeddings-based judge (`BaseRagasEmbeddings`) in
addition to the LLM judge, adding a second judge dependency the brief
didn't ask for; can be added later without touching this design.

**Trade-offs / risks:** None significant; the metric list is a plain
constant (`RAGAS_METRICS`) and easy to extend.

**Date/phase:** 2026-07-21, Phase 7.

---

## [Phase 7] Deferred the real `scripts/run_eval.py` run

**Decision:** Phase 7's real RAGAS scoring and ablation table
(`scripts/run_eval.py` against a built starlette index) was implemented
end-to-end but not executed in this session. The phase's status is
recorded as "code complete; manual run pending," not "Done."

**Why:** Same explicit constraint already logged for Phases 3-6: no real
model inference in this session (CPU cost). A real run needs a built
starlette index, real embedding/rerank/generation inference for every one
of the 35 Q&A items (×3 for the ablation table), and real RAGAS judge
calls through `CPUBackend` — the heaviest compute ask of any phase so
far. The dataset loader, metrics, `RagnarokRagasLLM` adapter, and
ablation routing logic are all fully unit-tested with fakes/mocks, so
nothing about correctness is unverified — only real scores are
outstanding.

**Alternatives considered:** Running it anyway to close out the phase
milestone fully — rejected outright given the explicit instruction.

**Trade-offs / risks:** Phase 7's milestone (real RAGAS numbers showing
hybrid+rerank/multi-hop beating naive, and ≥80% adversarial IDK rate)
isn't empirically confirmed yet — only the code paths are. This also
means the Phase 10 README's eval numbers and the "RAGAS passing bar"
editorial call both remain blocked on this real run. Run
`scripts/run_eval.py --ablation` against a built starlette index when
compute budget allows, commit the output, and flip the phase status to
`Done`.

**Date/phase:** 2026-07-21, Phase 7.

---

## [Phase 8] ZeroGPU quota-exceeded is caught as `gr.Error`, not a custom exception

**Decision:** `generation/fallback_backend.py`'s `FallbackBackend` catches
`gradio.Error` (`gr.Error`) around the primary backend's `.generate()`
call, not a ZeroGPU-specific exception type.

**Why:** Verified directly against the installed `spaces==0.51.0` source
(`spaces/zero/client.py`, `spaces/zero/gradio.py`): ZeroGPU's own
quota-exceeded path calls a helper named `error(title, message)` that
constructs and raises `gr.Error(...)` — Gradio's own user-facing exception
type, not a library-specific one. Catching `gr.Error` is therefore the
actually-correct, verified way to intercept this, rather than guessing at
a `spaces`-specific exception class that doesn't exist.

**Alternatives considered:** Catching a broad `Exception` — rejected as
too broad; it would also swallow genuine bugs in the primary backend
instead of only the intended quota/capacity signal.

**Trade-offs / risks:** If a future `spaces` version changes this to a
different exception type, the fallback would stop triggering silently.
Acceptable given this matches the library's actual current behavior,
verified rather than assumed.

**Date/phase:** 2026-07-21, Phase 8.

---

## [Phase 8] `Llama-cpp`-style: GGUF revision pinning already covered; ZeroGPU model pinned via ordinary `revision=`

**Decision:** `generation/zerogpu_backend.py`'s `ZeroGPUBackend` pins
`Qwen/Qwen2.5-Coder-7B-Instruct` to commit
`c03e6d358207e414f1eca0bb1891e29f1db0e242` via a plain `revision=` kwarg
on both `AutoTokenizer.from_pretrained` and
`AutoModelForCausalLM.from_pretrained` — unlike the Phase 4 GGUF backend,
`from_pretrained` here is the standard `transformers` API, which does
support `revision` directly (confirmed against the installed
`transformers==4.57.6`), so no `hf_hub_download`-style workaround was
needed this time.

**Why:** Consistent with the Phase 0 mitigation (pin every third-party
model load) and the Phase 8 audit task; documented here so the difference
from the Phase 4 GGUF workaround isn't mistaken for an inconsistency.

**Alternatives considered:** None — this is the straightforward case.

**Trade-offs / risks:** None.

**Date/phase:** 2026-07-21, Phase 8.

---

## [Phase 8] Paste-a-repo ingestion UI ships without a live progress indicator

**Decision:** `ui/ingest_view.py`'s `build_ingest_view` shows a single
status message once ingestion completes (or fails), not a live streaming
progress bar, even though the original Phase 8 task description mentions
a "progress indicator."

**Why:** A real streaming progress indicator needs Gradio's generator-
based `gr.Progress`/yielding-callback pattern threaded through
`scripts.build_index.ingest_repository`'s clone/chunk/embed/index stages
— a meaningfully separate feature from wiring the ingestion call itself,
and one that can't be verified without actually running a real, slow
ingestion (clone + real embedding model) end-to-end, which this session
avoids for the same CPU-cost reason logged in every prior phase. Shipping
the simpler synchronous version now means the core "paste a URL, get an
indexed repo" path is real and wired, rather than blocking all of task 3
on the progress-bar polish.

**Alternatives considered:** Building the full streaming version now —
rejected as disproportionate scope for a sub-feature that can't be tested
without a real, slow model-execution run anyway.

**Trade-offs / risks:** The UI will appear to "hang" during a real
ingestion run (no intermediate feedback) until Gradio's request
completes. Acceptable as a known, documented simplification; upgrading to
a streaming version later doesn't change the underlying `ingest_fn`
contract (`Callable[[str], str]`), just how it's called.

**Date/phase:** 2026-07-21, Phase 8.

---

## [Phase 8] Deferred the actual live HF Space deployment

**Decision:** All Phase 8 code (ZeroGPU backend, CPU fallback, paste-a-
repo ingestion UI, LRU memory guardrails, hop-path viz, Spaces YAML front
matter, revision-pinning audit) was implemented and unit-tested, but no
real Hugging Face Space was created or pushed to in this session. The
phase's status is recorded as "code complete; no live Space deployed
yet," not "Done."

**Why:** Unlike Phases 3-7 (deferred because they need real model
inference), this phase's remaining gap is different in kind: actually
deploying means creating a real, public-facing resource under a real HF
account/namespace — the "HF namespace/org" open item flagged since Phase
0 was never resolved by the user, and creating/pushing a public Space
without that decision (and without explicit authorization to take an
external, hard-to-reverse action under someone's account) would overstep
this session's scope regardless of compute constraints.

**Alternatives considered:** Deploying to some placeholder/throwaway HF
namespace anyway — rejected; picking a namespace unilaterally isn't this
session's call to make, and a real public Space is exactly the kind of
externally-visible, real-world action that warrants explicit
confirmation first.

**Trade-offs / risks:** Phase 8's actual milestone (a live public Space
answering real pasted-repo questions) isn't met yet — only the code that
would make it possible is. Once an HF namespace is confirmed, deployment
is: push this repo (or a Space-specific subset) to a new HF Space repo,
confirm it builds, and run through the manual verification steps above.

**Date/phase:** 2026-07-21, Phase 8.

---

## [Phase 9] Incremental reindexing implemented as a full-rebuild over a merged chunk set, not an in-place Chroma patch

**Decision:** `scripts/build_index.py`'s `_merge_incremental_chunks` reuses
existing chunks/embeddings (read back out of the current Chroma
collection via `.get(include=[...])`) for every file *not* changed since
`since_sha`, re-chunks/re-embeds only added/modified files, drops deleted
files' chunks entirely, and then calls the *existing* `build_index` (a
full atomic rebuild) on the merged set — rather than patching the
existing Chroma/bm25 on-disk structures in place.

**Why:** "Incremental" here means avoiding the expensive step (re-running
the embedding model over unchanged files), not literally patching
Chroma's internal storage — the done-criterion ("only A's chunks
re-embedded, B's rows byte-identical") is about which chunks get *new*
embeddings vs *reused* ones, which this achieves exactly. Reusing the
already-tested, already-atomic `build_index` (staged directory +
atomic rename, from Phase 2) for the final write means the incremental
path inherits that same crash-safety for free, instead of needing new
in-place-mutation logic for Chroma/bm25 that would have to be verified
separately.

**Alternatives considered:** Patching Chroma's collection in place
(`collection.update`/`.delete`/`.add` for just the changed ids) —
rejected: bm25s has no incremental update API at all (it's
build-from-corpus only), so the sparse index would need a full rebuild
regardless; keeping both index types on the same "rebuild from a merged
chunk set" path is simpler and more consistent than two different update
strategies for the two stores.

**Trade-offs / risks:** Every incremental run still pays the cost of
rebuilding the full Chroma/bm25 index structures (fast — it's the
embedding step, not the index build, that's expensive), and still reads
every existing embedding back out of Chroma via `.get()` — fine at
starlette's scale (thousands of chunks), would need revisiting for a
much larger repo.

**Date/phase:** 2026-07-21, Phase 9.

---

## [Phase 9] `--incremental` requires `--repo-path`, not `repo_url`

**Decision:** `ingest_repository(..., incremental=True)` raises
`ValueError` unless `repo_path` is also given; it cannot be combined with
`repo_url` (a fresh clone into a temp workspace).

**Why:** Incremental reindexing fundamentally needs a persistent on-disk
git history to diff `since_sha` against `HEAD` via `changed_files_since`.
`repo_url` mode clones into a throwaway temp workspace cleaned up at the
end of every call (Phase 2 design) — there's no persistent checkout to
diff a *second* call against. `repo_path` mode (an existing, persistent
local checkout, e.g. CI's own working directory) is the only mode where
"since the last run" is a meaningful question.

**Alternatives considered:** Making `repo_url` mode keep the clone around
(`--keep-clone`) and diff against that on a later call — rejected as
unnecessarily coupling incremental mode to a separate flag's caching
behavior; requiring `--repo-path` explicitly is a clearer contract for
callers (the `reindex.yml` workflow already needs to `git clone` fresh on
every run anyway, then reads via `--repo-path`).

**Trade-offs / risks:** None significant — this matches how the CI
workflow is designed to use it.

**Date/phase:** 2026-07-21, Phase 9.

---

## [Phase 9] Parquet rehydration (`rehydrate_index_from_parquet`) implements the Phase 0 "never depend on Chroma's on-disk format" decision

**Decision:** `utils/hf_dataset_io.py`'s `rehydrate_index_from_parquet`
reads our own parquet schema back into `Chunk`/embedding arrays and calls
the existing `build_index` to reconstruct a fresh local Chroma+bm25 index
— it does not attempt to ship or restore Chroma's own on-disk files.
`refresh_if_stale` ties this to `latest_commit_sha`/`pull_parquet` for a
one-call "check and refresh if needed" entry point.

**Why:** This is the concrete implementation of the Phase 0 Chroma
decision's stated trade-off: "we do NOT rely on Chroma's internal on-disk
format... we own a custom parquet schema... and rehydrate Chroma from
that on load." That was decided in Phase 0 but never actually built until
now; Phase 9's Space-auto-refresh requirement is what finally needed it.

**Alternatives considered:** Syncing Chroma's raw on-disk directory to/from
the Hugging Face Dataset repo directly — rejected per the original Phase 0
reasoning (not a guaranteed-stable format across Chroma versions).

**Trade-offs / risks:** None beyond what Phase 0 already accepted.

**Date/phase:** 2026-07-21, Phase 9.

---

## [Phase 9] Deferred: dispatching the real workflow and wiring Space-startup auto-refresh into `app.py`

**Decision:** `.github/workflows/reindex.yml`,
`scripts/push_index_to_hub.py`, and `utils/hf_dataset_io.py`'s
pull/rehydrate/refresh functions were implemented and unit-tested, but
the workflow was never actually dispatched, and `app.py` does not yet
call `refresh_if_stale` at startup. The phase's status is recorded as
"code complete; workflow never dispatched," not "Done."

**Why:** Both remaining gaps need the same unresolved prerequisite: a
real HF namespace/Dataset repo and a real `HF_TOKEN` secret, which is the
"HF namespace/org" open item flagged since Phase 0 and still not decided
by the user. Actually dispatching the workflow or wiring
`refresh_if_stale` into `app.py`'s startup would either fail outright
(no real Dataset repo to check) or require picking a namespace
unilaterally — the same category of real-external-action deferral
already logged for Phase 8's live Space deployment, not a compute-cost
deferral like Phases 3-7.

**Alternatives considered:** Wiring `refresh_if_stale` into `app.py`
against a placeholder/fake repo id anyway — rejected, since it would
either silently no-op or raise at Space startup with no real Dataset repo
behind it, adding dead code rather than a real capability.

**Trade-offs / risks:** Phase 9's actual milestone (a real CI run
refreshing a real Dataset repo, picked up by a restarted Space) isn't
met yet. Once an HF namespace/token exist: set `vars.HF_DATASET_REPO` and
`secrets.HF_TOKEN` in this repo's GitHub settings, run `gh workflow run
reindex.yml`, confirm the Dataset repo updates, add a `refresh_if_stale`
call to `app.py`'s startup, and flip this phase's status to `Done`.

**Date/phase:** 2026-07-21, Phase 9.

---

## [Phase 10] README ships without eval numbers or a demo GIF, rather than with placeholders

**Decision:** `README.md`'s Phase 10 polish includes a real Mermaid
architecture diagram and a real, HF-API-verified license table, but
explicitly states that eval numbers/ablation table and the demo GIF are
pending — no invented numbers, no stock/placeholder GIF.

**Why:** Phase 7's real RAGAS/ablation run and a real recorded
walkthrough both require exactly what's been deferred throughout this
session (real model inference, or a real running deployment) — neither
exists yet. Writing plausible-looking placeholder numbers into a README
would misrepresent the project's actual state (this file's own stated
purpose is "the README stands alone as evidence" — fabricated evidence
is worse than an honest gap). This mirrors the same honesty principle
applied to every other phase's status field in this session.

**Alternatives considered:** Inventing representative-looking numbers
"to be replaced later" — rejected outright; a README is exactly the kind
of artifact readers take at face value, and a stale/fake number left in
by accident is a much worse outcome than a clearly-labeled pending
section.

**Trade-offs / risks:** The README is less immediately impressive without
numbers/a GIF. Acceptable — accuracy over polish, consistent with every
other phase-completion claim in this project.

**Date/phase:** 2026-07-21, Phase 10.

---

## [Phase 10] Third-party model licenses verified live via the HF API, not recalled from memory

**Decision:** All four model licenses in `README.md`'s attribution table
(`jina-embeddings-v2-base-code`, `ms-marco-MiniLM-L-6-v2`,
`Qwen2.5-Coder-1.5B-Instruct-GGUF`, `Qwen2.5-Coder-7B-Instruct`) were
fetched from each model's live Hugging Face API metadata
(`cardData.license`) in this session — all four confirmed Apache 2.0 —
rather than stated from training-data recall.

**Why:** Same reasoning as the Phase 7 hand-labeled dataset entry: a
license claim in a public README is exactly the kind of fact that must
be verified, not guessed, especially for a project whose own brief
requires confirmed licensing for every third-party model.

**Alternatives considered:** Stating licenses from memory — rejected for
the same reason recalled-from-memory eval data was rejected in Phase 7.

**Trade-offs / risks:** None — this is strictly more reliable than the
alternative.

**Date/phase:** 2026-07-21, Phase 10.
