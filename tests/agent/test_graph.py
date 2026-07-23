from src.agent.graph import build_graph, llm_decide_more_hops
from src.agent.state import AgentState
from src.config import AGENT_MAX_HOPS
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import RetrievedChunk


def _fake_chunk(name: str, file_path: str) -> Chunk:
    return Chunk(
        chunk_id=name,
        file_path=file_path,
        start_line=1,
        end_line=2,
        symbol_name=name,
        qualified_name=name,
        symbol_type="function",
        language="python",
        text=f"def {name}(): ...",
    )


class _StubDecider:
    """Returns each queued symbol name in order, then None once exhausted."""

    def __init__(self, symbols: list[str | None]) -> None:
        self._symbols = iter(symbols)

    def __call__(self, state: AgentState) -> str | None:
        return next(self._symbols, None)


class _FakeBackend:
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

        assert llm_decide_more_hops(_CannedBackend("DONE"), state) is None

    def test_a_symbol_reply_is_returned_as_the_next_hop(self):
        state = AgentState(question="q", context=[])

        assert llm_decide_more_hops(_CannedBackend("AuthenticationMiddleware"), state) == (
            "AuthenticationMiddleware"
        )

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
