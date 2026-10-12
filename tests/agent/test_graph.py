from src.agent.graph import NO_CANDIDATES_REASON, HopDecision, build_graph, decide_next_hop
from src.agent.state import AgentState
from src.config import AGENT_MAX_HOPS
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import RetrievedChunk


def _fake_chunk(name: str, file_path: str, text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path=file_path,
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        repo_sha="sha1",
        text=text if text is not None else f"def {name}(): ...",
    )


class _StubDecider:
    """Returns each queued symbol name (wrapped in a HopDecision) in order, then None/DONE."""

    def __init__(self, symbols: list[str | None]) -> None:
        self._symbols = iter(symbols)

    def __call__(self, state: AgentState) -> HopDecision:
        symbol = next(self._symbols, None)
        reason = "test reason" if symbol is not None else "test says DONE"
        return HopDecision(symbol=symbol, reason=reason)


class _FakeBackend:
    context_window = 100_000

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_tokens: int) -> str:
        return "the answer"


def _find_symbol(chunks_by_symbol: dict[str, RetrievedChunk]):
    def find_symbol(symbol_name: str) -> RetrievedChunk | None:
        return chunks_by_symbol.get(symbol_name)

    return find_symbol


class TestBuildGraph:
    def test_traces_exactly_two_hops_in_order(self):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)
        middleware = RetrievedChunk(
            chunk=_fake_chunk("AuthenticationMiddleware", "starlette/authentication.py"), score=1.0
        )
        backend_call = RetrievedChunk(
            chunk=_fake_chunk("authenticate", "starlette/authentication.py"), score=1.0
        )

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol(
                {"AuthenticationMiddleware": middleware, "authenticate": backend_call}
            ),
            decide_more_hops=_StubDecider(["AuthenticationMiddleware", "authenticate", None]),
            backend=_FakeBackend(),
        )

        result = graph.invoke(AgentState(question="how is auth validated?"))

        assert [hop.symbol_name for hop in result["hop_path"]] == [
            "AuthenticationMiddleware",
            "authenticate",
        ]
        assert [c.chunk.symbol_name for c in result["context"]] == ["Router"]
        assert [c.chunk.symbol_name for c in result["hop_context"]] == [
            "AuthenticationMiddleware",
            "authenticate",
        ]
        assert result["answer"] == "the answer"
        assert result["hops_remaining"] == AGENT_MAX_HOPS - 2

    def test_stops_immediately_when_no_hop_is_needed(self):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({}),
            decide_more_hops=_StubDecider([None]),
            backend=_FakeBackend(),
        )

        result = graph.invoke(AgentState(question="what is Router?"))

        assert result.get("hop_path", []) == []
        assert result["answer"] == "the answer"

    def test_stops_after_max_hops_even_if_the_decider_keeps_requesting_more(self):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)
        symbols = [f"Symbol{i}" for i in range(AGENT_MAX_HOPS + 5)]
        definitions = {
            name: RetrievedChunk(chunk=_fake_chunk(name, f"{name}.py"), score=1.0)
            for name in symbols
        }

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol(definitions),
            decide_more_hops=_StubDecider(list(symbols)),
            backend=_FakeBackend(),
        )

        result = graph.invoke(AgentState(question="trace it"))

        assert len(result["hop_path"]) == AGENT_MAX_HOPS
        assert result["hops_remaining"] == 0

    def test_stops_when_the_requested_definition_is_already_in_context(self):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)
        decider = _StubDecider(["Router", "Router", "Router"])

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({"Router": initial}),
            decide_more_hops=decider,
            backend=_FakeBackend(),
        )

        result = graph.invoke(AgentState(question="what is Router?"))

        assert result.get("hop_path", []) == []
        assert result.get("hop_context", []) == []

    def test_stops_when_a_requested_symbol_cannot_be_resolved(self):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({}),
            decide_more_hops=_StubDecider(["MissingSymbol", None]),
            backend=_FakeBackend(),
        )

        result = graph.invoke(AgentState(question="trace it"))

        assert result.get("hop_path", []) == []
        assert [c.chunk.symbol_name for c in result["context"]] == ["Router"]


class TestHopDecisionObservability:
    def test_hop_record_reason_reflects_the_deciders_actual_reason(self):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)
        found = RetrievedChunk(
            chunk=_fake_chunk("Middleware", "starlette/middleware.py"), score=1.0
        )

        def decide(state: AgentState) -> HopDecision:
            if state.hop_path:
                return HopDecision(symbol=None, reason="done")
            return HopDecision(symbol="Middleware", reason="need the middleware chain")

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({"Middleware": found}),
            decide_more_hops=decide,
            backend=_FakeBackend(),
        )

        result = graph.invoke(AgentState(question="trace it"))

        assert result["hop_path"][0].reason == "need the middleware chain"

    def test_logs_distinguish_an_intentional_done_from_an_unresolved_symbol_guess(self, caplog):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)

        done_graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({}),
            decide_more_hops=_StubDecider([None]),
            backend=_FakeBackend(),
        )
        unresolved_graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({}),
            decide_more_hops=_StubDecider(["MissingSymbol"] * AGENT_MAX_HOPS),
            backend=_FakeBackend(),
        )

        with caplog.at_level("INFO"):
            caplog.clear()
            done_graph.invoke(AgentState(question="what is Router?"))
            done_log = caplog.text

            caplog.clear()
            unresolved_graph.invoke(AgentState(question="trace it"))
            unresolved_log = caplog.text

        assert "No further hop" in done_log
        assert "Could not resolve symbol" not in done_log
        assert "Could not resolve symbol" in unresolved_log
        assert "No further hop" not in unresolved_log


class _RecordingBackend:
    """Records every prompt passed to generate; a small context_window forces trimming."""

    def __init__(self, context_window: int) -> None:
        self.context_window = context_window
        self.prompts: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.prompts.append(prompt)
        return "the answer"


class TestComposeAnswerContextBudget:
    def test_trims_lowest_ranked_chunks_to_fit_the_backends_context_window(self):
        big_text = "word " * 200
        ranked_chunks = [
            RetrievedChunk(
                chunk=_fake_chunk(f"chunk{i}", f"file{i}.py", big_text), score=float(10 - i)
            )
            for i in range(10)
        ]
        backend = _RecordingBackend(context_window=1000)

        graph = build_graph(
            retrieve=lambda question: ranked_chunks,
            find_symbol=_find_symbol({}),
            decide_more_hops=_StubDecider([None]),
            backend=backend,
        )

        graph.invoke(AgentState(question="q", hops_remaining=0))

        prompt = backend.prompts[0]
        assert "file0.py" in prompt
        assert "file9.py" not in prompt

    def test_keeps_hopped_definitions_over_low_ranked_retrieval_results(self):
        big_text = "word " * 200
        ranked_chunks = [
            RetrievedChunk(
                chunk=_fake_chunk(f"chunk{i}", f"file{i}.py", big_text), score=float(10 - i)
            )
            for i in range(10)
        ]
        hopped = RetrievedChunk(chunk=_fake_chunk("Target", "target.py", big_text), score=1.0)
        backend = _RecordingBackend(context_window=1100)

        graph = build_graph(
            retrieve=lambda question: ranked_chunks,
            find_symbol=_find_symbol({"Target": hopped}),
            decide_more_hops=_StubDecider(["Target", None]),
            backend=backend,
        )

        graph.invoke(AgentState(question="q"))

        prompt = backend.prompts[-1]
        assert "file0.py" in prompt
        assert "target.py" in prompt
        assert "file1.py" not in prompt


class TestComposeAnswerRelevanceGuardrail:
    def test_short_circuits_when_all_context_scores_are_below_threshold(self):
        irrelevant = RetrievedChunk(chunk=_fake_chunk("Unrelated", "other/file.py"), score=-5.0)
        backend = _RecordingBackend(context_window=100_000)

        graph = build_graph(
            retrieve=lambda question: [irrelevant],
            find_symbol=_find_symbol({}),
            decide_more_hops=_StubDecider([None]),
            backend=backend,
        )

        result = graph.invoke(AgentState(question="what color is the sky on mars?"))

        assert "doesn't appear to be covered" in result["answer"]
        assert backend.prompts == []

    def test_a_hop_does_not_bypass_the_threshold(self):
        irrelevant = RetrievedChunk(chunk=_fake_chunk("Unrelated", "other/file.py"), score=-5.0)
        looked_up = RetrievedChunk(chunk=_fake_chunk("Mars", "mars.py"), score=1.0)
        backend = _RecordingBackend(context_window=100_000)

        graph = build_graph(
            retrieve=lambda question: [irrelevant],
            find_symbol=_find_symbol({"Mars": looked_up}),
            decide_more_hops=_StubDecider(["Mars", None]),
            backend=backend,
        )

        result = graph.invoke(AgentState(question="what color is the sky on mars?"))

        assert "doesn't appear to be covered" in result["answer"]
        assert result.get("hop_path", []) == []
        assert backend.prompts == []

    def test_short_circuits_when_there_is_no_context_at_all(self):
        backend = _RecordingBackend(context_window=100_000)

        graph = build_graph(
            retrieve=lambda question: [],
            find_symbol=_find_symbol({}),
            decide_more_hops=_StubDecider([None]),
            backend=backend,
        )

        result = graph.invoke(AgentState(question="what color is the sky on mars?"))

        assert "doesn't appear to be covered" in result["answer"]
        assert backend.prompts == []

    def test_generates_normally_when_context_meets_the_threshold(self):
        relevant = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=5.0)
        backend = _RecordingBackend(context_window=100_000)

        graph = build_graph(
            retrieve=lambda question: [relevant],
            find_symbol=_find_symbol({}),
            decide_more_hops=_StubDecider([None]),
            backend=backend,
        )

        result = graph.invoke(AgentState(question="what is Router?"))

        assert result["answer"] == "the answer"
        assert len(backend.prompts) == 1


class TestRetrieveNodeErrorHandling:
    def test_retrieval_failure_degrades_to_empty_context_instead_of_raising(self, caplog):
        def failing_retrieve(question: str) -> list[RetrievedChunk]:
            raise RuntimeError("chroma exploded")

        graph = build_graph(
            retrieve=failing_retrieve,
            find_symbol=_find_symbol({}),
            decide_more_hops=_StubDecider([None]),
            backend=_FakeBackend(),
        )

        with caplog.at_level("ERROR"):
            result = graph.invoke(AgentState(question="how is auth handled?"))

        assert result["context"] == []
        assert "doesn't appear to be covered" in result["answer"]
        assert "Retrieval failed" in caplog.text


class TestFollowSymbolNodeErrorHandling:
    def test_symbol_lookup_failure_stops_hopping_instead_of_raising(self, caplog):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)

        def failing_find_symbol(symbol_name: str) -> RetrievedChunk | None:
            raise RuntimeError("chroma exploded")

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=failing_find_symbol,
            decide_more_hops=_StubDecider(["Middleware", None]),
            backend=_FakeBackend(),
        )

        with caplog.at_level("ERROR"):
            result = graph.invoke(AgentState(question="trace it"))

        assert result.get("hop_path", []) == []
        assert [c.chunk.symbol_name for c in result["context"]] == ["Router"]
        assert "Symbol lookup failed" in caplog.text


class TestDecideNextHop:
    def test_picks_the_most_relevant_candidate_and_reports_its_score(self):
        best = RetrievedChunk(chunk=_fake_chunk("RedirectResponse", "responses.py"), score=3.24)
        other = RetrievedChunk(chunk=_fake_chunk("HTTPException", "exceptions.py"), score=1.5)
        state = AgentState(question="q", context=[])

        decision = decide_next_hop(lambda retrieved: [best, other], state)

        assert decision.symbol == "RedirectResponse"
        assert "3.2" in decision.reason

    def test_stops_when_nothing_relevant_is_left_to_fetch(self):
        state = AgentState(question="q", context=[])

        decision = decide_next_hop(lambda retrieved: [], state)

        assert decision == HopDecision(symbol=None, reason=NO_CANDIDATES_REASON)

    def test_candidates_are_drawn_from_retrieved_and_hopped_chunks(self):
        retrieved = RetrievedChunk(chunk=_fake_chunk("Router", "routing.py"), score=1.0)
        hopped = RetrievedChunk(chunk=_fake_chunk("Mount", "routing.py"), score=1.0)
        state = AgentState(question="q", context=[retrieved], hop_context=[hopped])
        seen: list[list[str]] = []

        def list_candidates(chunks):
            seen.append([c.chunk.symbol_name for c in chunks])
            return []

        decide_next_hop(list_candidates, state)

        assert seen == [["Router", "Mount"]]


class TestDecideNodeErrorHandling:
    def test_a_failing_hop_decision_degrades_to_answering_without_hops(self, caplog):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)

        def failing_decider(state: AgentState) -> HopDecision:
            raise RuntimeError("index unavailable")

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({}),
            decide_more_hops=failing_decider,
            backend=_FakeBackend(),
        )

        with caplog.at_level("ERROR"):
            result = graph.invoke(AgentState(question="q"))

        assert result["answer"] == "the answer"
        assert result.get("hop_path", []) == []
        assert "Hop decision failed" in caplog.text


class _CountingDecider(_StubDecider):
    def __init__(self, symbols: list[str | None]) -> None:
        super().__init__(symbols)
        self.calls = 0

    def __call__(self, state: AgentState) -> HopDecision:
        self.calls += 1
        return super().__call__(state)


class TestHopStopping:
    def test_an_unresolvable_symbol_ends_hopping_without_asking_again(self):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)
        decider = _CountingDecider(["Missing", "Missing", "Missing"])

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({}),
            decide_more_hops=decider,
            backend=_FakeBackend(),
        )

        result = graph.invoke(AgentState(question="q"))

        assert decider.calls == 1
        assert result["answer"] == "the answer"

    def test_a_symbol_already_in_context_ends_hopping_without_asking_again(self):
        initial = RetrievedChunk(chunk=_fake_chunk("Router", "starlette/routing.py"), score=1.0)
        decider = _CountingDecider(["Router", "Router", "Router"])

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({"Router": initial}),
            decide_more_hops=decider,
            backend=_FakeBackend(),
        )

        graph.invoke(AgentState(question="q"))

        assert decider.calls == 1
