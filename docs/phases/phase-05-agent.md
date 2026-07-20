# Phase 5 — Multi-hop LangGraph agent + hop-path tracking

**Status:** Not started
**Milestone:** a real cross-file starlette question triggers ≥2 tracked
hops with verified citations at each step.

## Tasks

1. **Agent state schema** — `agent/state.py`: `AgentState`, `HopRecord`
   (pydantic).
   *Done:* serialization/validation unit tests.

2. **Retrieval-as-tool** — `agent/tools.py`: hybrid retrieval tool + "find
   symbol definition by name" targeted-lookup tool.
   *Done:* unit test resolves a known starlette function name (e.g.
   `AuthenticationMiddleware.__call__`) to its exact defining chunk.

3. **LangGraph graph** — `agent/graph.py`: `retrieve → decide_more_hops →
   follow_symbol (loop, max 3 hops) → compose_answer`.
   *Done:* unit test with a mocked decision function traces exactly 2 hops
   on a constructed 2-file scenario, recorded in order in `hop_path`.

4. **Real multi-hop demonstration** — run against a genuine cross-file
   question on starlette (e.g. "how does auth get validated on the
   request path?" — should trace `Router` match → middleware stack →
   `AuthenticationMiddleware` → `backend.authenticate()` → `request.user`).
   *Done:* `python scripts/ask.py --agent "<question>"` produces a
   `hop_path` with ≥2 distinct files/symbols, in order, every hop's
   citation verified.

5. **Hop-path serialization for UI** — JSON-ready ordered hop list (file,
   symbol, line range, reason).
   *Done:* `HopRecord` round-trips through `model_dump_json`/
   `model_validate_json`.

## Verification for this phase

Run the brief's own example query through `scripts/ask.py --agent` against
starlette and manually confirm the printed hop path matches the actual
code path (Router → middleware → AuthenticationBackend → endpoint).
