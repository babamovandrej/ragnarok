# Phase 4 — Single-hop generation with citation verification

**Status:** Not started
**Milestone:** end-to-end single-hop question answering on the CPU
fallback model, with every citation checked against the real index.

## Tasks

1. **LLM backend interface** — `generation/llm.py` abstract
   `generate(prompt, max_tokens) -> str`.
   *Done:* unit test against a mock backend proves the contract.

2. **CPU backend** — `generation/cpu_backend.py`, `llama-cpp-python` +
   `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` (Q4_K_M), installed via
   `pip install llama-cpp-python --extra-index-url
   https://abetlen.github.io/llama-cpp-python/whl/cpu`.
   *Done:* install completes without a from-source compile (verify no long
   build step), smoke test generates a completion for a fixed prompt.

3. **Prompt templates** — `generation/prompts.py`: mandatory
   `file#Lstart-Lend` citation format, explicit "answer only from context"
   + "say I don't know" instructions.
   *Done:* unit test renders a prompt from sample chunks and asserts
   required strings present.

4. **Citation parser + verifier** — `generation/citations.py`: parse cited
   spans out of model output, cross-check each against the real chunk
   index, drop/flag unverifiable ones.
   *Done:* unit test with one real + one fabricated citation shows the
   verifier keeps the real one and flags/drops the fake one.

5. **Single-hop pipeline wire-up** — retrieve → generate (CPU) → verify
   citations.
   *Done:* `python scripts/ask.py "<question>"` against starlette produces
   an answer with ≥1 verified citation end-to-end.

## Verification for this phase

`python scripts/ask.py "how does starlette validate authentication on the
request path?"` returns a plain-language answer citing real
`file#Lstart-Lend` spans in `starlette/authentication.py` and/or
`starlette/routing.py`, all verified against the index.
