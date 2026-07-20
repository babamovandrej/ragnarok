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
start_line, end_line, symbol_name, symbol_type, language, text,
embedding`) written via `pyarrow` and rehydrate Chroma from that on load.

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
