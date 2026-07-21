# Phase 4 — Single-hop generation with citation verification

**Status:** Code complete; manual end-to-end run (task 5) pending
**Milestone:** end-to-end single-hop question answering on the CPU
fallback model, with every citation checked against the real index.

## Tasks

1. **LLM backend interface** — `generation/llm.py` `LLMBackend`, a
   `typing.Protocol` for `generate(prompt, max_tokens) -> str`.
   *Done:* unit test proves the contract structurally (a mock backend
   satisfies it, an unrelated class doesn't). ✅ `tests/generation/test_llm.py`.

2. **CPU backend** — `generation/cpu_backend.py`'s `CPUBackend`,
   `llama-cpp-python` + `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` (Q4_K_M),
   installed via `pip install llama-cpp-python --extra-index-url
   https://abetlen.github.io/llama-cpp-python/whl/cpu`. The GGUF file is
   pulled with `huggingface_hub.hf_hub_download(..., revision=...)` rather
   than `Llama.from_pretrained` (which has no `revision` parameter — see
   `DECISIONS.md`), keeping the pinned-revision guarantee for this model
   too.
   *Done:* unit tests assert the pinned repo/filename/revision and context
   window are used, and that `generate` wraps `create_chat_completion`
   correctly, with `Llama`/`hf_hub_download` mocked out (no download/
   inference). ✅ `tests/generation/test_cpu_backend.py` — a
   `@pytest.mark.integration` test using the real model on a fixed prompt
   is also present but not executed in this environment (see below).

3. **Prompt templates** — `generation/prompts.py`'s `render_prompt`:
   mandatory `file#Lstart-Lend` citation format, explicit "answer only
   from context" + "say I don't know" instructions.
   *Done:* unit test renders a prompt from sample chunks and asserts
   required strings present. ✅ `tests/generation/test_prompts.py`.

4. **Citation parser + verifier** — `generation/citations.py`'s
   `verify_citations`: parse cited spans out of model output, cross-check
   each against the real chunk index, flag unverifiable ones.
   *Done:* unit test with one real + one fabricated citation shows the
   verifier keeps the real one verified and flags the fake one.
   ✅ `tests/generation/test_citations.py`.

5. **Single-hop pipeline wire-up** — `scripts/ask.py`: retrieve (dense +
   sparse + RRF fusion + rerank) → generate (CPU) → verify citations.
   *Done:* `python scripts/ask.py "<question>"` against starlette produces
   an answer with ≥1 verified citation end-to-end.
   *Not yet run.* `scripts/ask.py` is implemented and ready (takes a built
   index directory + a question, prints the generated answer and each
   citation's verification status) — see `tests/scripts/test_ask.py` for
   its wiring/CLI tests with `Embedder`/`Reranker`/`CPUBackend` mocked
   out. Actually running it against `encode/starlette` requires:
   `python scripts/build_index.py https://github.com/encode/starlette`
   (clones + embeds, real model), then `python scripts/ask.py
   .ragnarok_cache/encode/starlette "how does starlette validate
   authentication on the request path?"`. This was intentionally not run
   in this session to avoid the embedding + reranking + generation model
   load/inference cost; run it locally and flip the status above to
   `Done` once it produces a verified-citation answer.

## Verification for this phase

Run `python scripts/ask.py "how does starlette validate authentication on
the request path?"` against a built starlette index and confirm the
answer cites real `file#Lstart-Lend` spans in
`starlette/authentication.py` and/or `starlette/routing.py`, all verified
against the index, before moving to Phase 5. All other verification (unit
tests for each generation module, `ruff check .`, `pytest -q`) is already
green with zero regressions to the Phase 2/3 suites.
