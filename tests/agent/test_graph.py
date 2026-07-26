from src.agent.graph import HopDecision, build_graph, llm_decide_more_hops
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
        assert [c.chunk.symbol_name for c in result["context"]] == [
            "Router",
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
        found = RetrievedChunk(
            chunk=_fake_chunk("Middleware", "starlette/middleware.py"), score=1.0
        )

        graph = build_graph(
            retrieve=lambda question: [initial],
            find_symbol=_find_symbol({"Middleware": found}),
            decide_more_hops=_StubDecider(["Middleware"] * (AGENT_MAX_HOPS + 5)),
            backend=_FakeBackend(),
        )

        result = graph.invoke(AgentState(question="trace it"))

        assert len(result["hop_path"]) == AGENT_MAX_HOPS
        assert result["hops_remaining"] == 0

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
            # Never yields None/DONE: every attempt is an unresolved guess, so the loop only
            # ends by exhausting hops_remaining, never by the decider intentionally stopping.
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

        assert "chose DONE" in done_log
        assert "Could not resolve symbol" not in done_log
        assert "Could not resolve symbol" in unresolved_log
        assert "chose DONE" not in unresolved_log


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


class _CannedBackend:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.calls.append((prompt, max_tokens))
        return self.reply


class TestLLMDecideMoreHops:
    def test_a_done_reply_means_stop_hopping(self):
        state = AgentState(question="q", context=[])

        assert llm_decide_more_hops(_CannedBackend("DONE"), state).symbol is None

    def test_a_bare_symbol_reply_is_returned_as_the_next_hop(self):
        """A reply that ignores the requested SYMBOL:/REASON: format still resolves (fallback)."""
        state = AgentState(question="q", context=[])

        decision = llm_decide_more_hops(_CannedBackend("AuthenticationMiddleware"), state)

        assert decision.symbol == "AuthenticationMiddleware"

    def test_a_well_formed_reply_returns_both_symbol_and_reason(self):
        state = AgentState(question="q", context=[])
        reply = "SYMBOL: AuthenticationMiddleware\nREASON: it authenticates the request"

        decision = llm_decide_more_hops(_CannedBackend(reply), state)

        assert decision.symbol == "AuthenticationMiddleware"
        assert decision.reason == "it authenticates the request"

    def test_a_well_formed_done_reply_carries_its_reason(self):
        state = AgentState(question="q", context=[])
        reply = "SYMBOL: DONE\nREASON: context already answers the question"

        decision = llm_decide_more_hops(_CannedBackend(reply), state)

        assert decision.symbol is None
        assert decision.reason == "context already answers the question"

    def test_prompt_includes_the_question_and_context_so_far(self):
        chunk = _fake_chunk("Router", "starlette/routing.py")
        state = AgentState(
            question="how is auth validated?",
            context=[RetrievedChunk(chunk=chunk, score=1.0)],
        )
        backend = _CannedBackend("DONE")

        llm_decide_more_hops(backend, state)

        prompt = backend.calls[0][0]
        assert "how is auth validated?" in prompt
        assert "Router" in prompt
        assert "starlette/routing.py" in prompt
