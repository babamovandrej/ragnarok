# Phase 5 — Multi-hop LangGraph agent + hop-path tracking

**Status:** Code complete; real multi-hop demonstration (task 4) pending
**Milestone:** a real cross-file starlette question triggers ≥2 tracked
hops with verified citations at each step.

## Tasks

1. **Agent state schema** — `agent/state.py`: `AgentState`, `HopRecord`
   (pydantic).
   *Done:* serialization/validation unit tests. ✅ `tests/agent/test_state.py`
   — `HopRecord` round-trips through `model_dump_json`/`model_validate_json`;
   `AgentState` rejects a missing `question` and defaults `hops_remaining`
   to `AGENT_MAX_HOPS`.

2. **Retrieval-as-tool** — `agent/tools.py`: `retrieve_hybrid` (the full
   Phase 3 dense+sparse+RRF+rerank pipeline) and `find_symbol_definition`
   (exact `symbol_name` Chroma metadata lookup — no embedding call needed).
   *Done:* unit test resolves a known function name to its exact defining
   chunk. ✅ `tests/agent/test_tools.py`.

3. **LangGraph graph** — `agent/graph.py`'s `build_graph`: `retrieve →
   decide_more_hops → follow_symbol (loop, max `AGENT_MAX_HOPS` hops) →
   compose_answer`. `decide_more_hops` and `find_symbol` are injected
   callables, so the graph's control flow is fully testable without any
   real model call; `llm_decide_more_hops(backend, state)` is the real
   production decision function (asks the backend for a symbol name or
   `DONE`).
   *Done:* unit test with a mocked decision function traces exactly 2 hops
   on a constructed 2-file scenario, recorded in order in `hop_path`.
   ✅ `tests/agent/test_graph.py` — also covers stopping when the decider
   says `DONE` immediately, stopping at the hop cap even if the decider
   keeps requesting more, and stopping when a requested symbol can't be
   resolved.

4. **Real multi-hop demonstration** — run against a genuine cross-file
   question on starlette (e.g. "how does auth get validated on the
   request path?" — should trace `Router` match → middleware stack →
   `AuthenticationMiddleware` → `backend.authenticate()` → `request.user`).
   *Done:* `python scripts/ask.py --agent "<question>"` produces a
   `hop_path` with ≥2 distinct files/symbols, in order, every hop's
   citation verified.
   *Not yet run.* `scripts/ask.py --agent` is implemented and wired
   end-to-end (`answer_question_with_agent` builds the graph with real
   `Embedder`/`Reranker`/`CPUBackend` and prints the hop path before the
   answer) — see `tests/scripts/test_ask.py` for its wiring tests with all
   three mocked out. Actually running it requires a real built starlette
   index and real embedding/rerank/generation inference; intentionally not
   run in this session (CPU cost) — run it locally and flip the status
   above to `Done` once a real ≥2-hop trace is observed.

5. **Hop-path serialization for UI** — JSON-ready ordered hop list (file,
   symbol, line range, reason).
   *Done:* `HopRecord` round-trips through `model_dump_json`/
   `model_validate_json`. ✅ `tests/agent/test_state.py`.

## Verification for this phase

Run the brief's own example query through `scripts/ask.py --agent` against
starlette and manually confirm the printed hop path matches the actual
code path (Router → middleware → AuthenticationBackend → endpoint). All
other verification (unit tests for state/tools/graph, `ruff check .`,
`pytest -q`) is already green with zero regressions to the Phase 1-4
suites.
