import logging
from collections.abc import Callable
from typing import NamedTuple, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.state import AgentState, HopRecord
from src.config import GENERATION_MAX_TOKENS_DEFAULT, MIN_CONTEXT_RELEVANCE_SCORE
from src.generation.llm import GenerationBackend
from src.generation.prompts import NOT_IN_REPO_ANSWER, fit_chunks_to_budget, render_prompt
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str], list[RetrievedChunk]]
CandidatesFn = Callable[[list[RetrievedChunk]], list[RetrievedChunk]]
FindSymbolFn = Callable[[str], RetrievedChunk | None]


class HopDecision(NamedTuple):
    """The next hop: the symbol whose definition to fetch (None means stop), and why."""

    symbol: str | None
    reason: str


DecideMoreHopsFn = Callable[[AgentState], HopDecision]


class _AgentStateUpdate(TypedDict, total=False):
    """Partial update applied to `AgentState` after a graph node runs."""

    context: list[RetrievedChunk]
    hop_context: list[RetrievedChunk]
    hop_path: list[HopRecord]
    pending_symbol: str | None
    pending_reason: str | None
    hops_remaining: int
    answer: str


NO_CANDIDATES_REASON = "the retrieved code refers to no other relevant definitions"


def _best_retrieval_score(state: AgentState) -> float:
    return max((c.score for c in state.context), default=float("-inf"))


def _prompt_context(state: AgentState) -> list[RetrievedChunk]:
    """Order context by priority for fit_chunks_to_budget, which trims from the tail.

    The top retrieved chunk comes first, then every chunk the agent deliberately hopped
    to, then the remaining retrieved chunks — so trimming drops low-ranked retrieval
    results before it drops the definitions the agent went looking for.
    """
    return [*state.context[:1], *state.hop_context, *state.context[1:]]


def decide_next_hop(list_candidates: CandidatesFn, state: AgentState) -> HopDecision:
    """Pick the next definition to fetch: the most relevant one the context refers to.

    list_candidates returns the definitions the context's code refers to that are relevant
    to the question, best first, each scored by the reranker. The choice is deliberately not
    left to the generation model: asked to choose, a 3B model took every hop it was offered
    and at times named none legibly, while the reranker's scores separate useful definitions
    from incidental ones.
    """
    candidates = list_candidates(_prompt_context(state))
    if not candidates:
        return HopDecision(symbol=None, reason=NO_CANDIDATES_REASON)
    best = candidates[0]
    return HopDecision(
        symbol=best.chunk.symbol_name,
        reason=f"The retrieved code refers to it, and it scored {best.score:.1f} for relevance",
    )


def build_graph(
    retrieve: RetrieveFn,
    find_symbol: FindSymbolFn,
    decide_more_hops: DecideMoreHopsFn,
    backend: GenerationBackend,
) -> CompiledStateGraph:
    """Wire retrieve -> decide_more_hops -> follow_symbol (looped) -> compose_answer."""

    def retrieve_node(state: AgentState) -> _AgentStateUpdate:
        try:
            context = retrieve(state.question)
        except Exception:
            logger.exception(
                "Retrieval failed for question %r; continuing with no context", state.question
            )
            context = []
        logger.info("Initial retrieval returned %d chunk(s)", len(context))
        return {"context": context}

    def decide_more_hops_node(state: AgentState) -> _AgentStateUpdate:
        if state.hops_remaining <= 0:
            logger.info("No hops remaining")
            return {"pending_symbol": None, "pending_reason": None}
        if _best_retrieval_score(state) < MIN_CONTEXT_RELEVANCE_SCORE:
            logger.info("Nothing relevant retrieved; skipping hops")
            return {"pending_symbol": None, "pending_reason": None}
        try:
            decision = decide_more_hops(state)
        except Exception:
            logger.exception("Hop decision failed; continuing without further hops")
            return {"pending_symbol": None, "pending_reason": None}
        if decision.symbol is None:
            logger.info("No further hop: %s", decision.reason)
        return {"pending_symbol": decision.symbol, "pending_reason": decision.reason}

    def route_after_decision(state: AgentState) -> str:
        return "follow_symbol" if state.pending_symbol else "compose_answer"

    def follow_symbol_node(state: AgentState) -> _AgentStateUpdate:
        symbol = state.pending_symbol
        assert symbol is not None, "follow_symbol_node reached without a pending symbol"
        stop: _AgentStateUpdate = {
            "pending_symbol": None,
            "pending_reason": None,
            "hops_remaining": 0,
        }
        try:
            found = find_symbol(symbol)
        except Exception:
            logger.exception("Symbol lookup failed for %r; stopping hops", symbol)
            return stop
        if found is None:
            logger.info("Could not resolve symbol %s; stopping hops", symbol)
            return stop
        known_ids = {c.chunk.chunk_id for c in [*state.context, *state.hop_context]}
        if found.chunk.chunk_id in known_ids:
            logger.info("Definition of %s is already in context; stopping hops", symbol)
            return stop
        hop = HopRecord(
            file_path=found.chunk.file_path,
            symbol_name=found.chunk.symbol_name,
            start_line=found.chunk.start_line,
            end_line=found.chunk.end_line,
            reason=state.pending_reason or f"Followed reference to {symbol}",
        )
        return {
            "hop_context": [*state.hop_context, found],
            "hop_path": [*state.hop_path, hop],
            "pending_symbol": None,
            "pending_reason": None,
            "hops_remaining": state.hops_remaining - 1,
        }

    def compose_answer_node(state: AgentState) -> _AgentStateUpdate:
        best_score = _best_retrieval_score(state)
        if best_score < MIN_CONTEXT_RELEVANCE_SCORE:
            logger.info(
                "Best context score %.3f is below the relevance threshold %.3f; answering "
                "without calling the model",
                best_score,
                MIN_CONTEXT_RELEVANCE_SCORE,
            )
            return {"answer": NOT_IN_REPO_ANSWER}

        fitted_context = fit_chunks_to_budget(
            state.question, _prompt_context(state), backend, GENERATION_MAX_TOKENS_DEFAULT
        )
        prompt = render_prompt(state.question, fitted_context)
        answer = backend.generate(prompt, max_tokens=GENERATION_MAX_TOKENS_DEFAULT)
        return {"answer": answer}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decide_more_hops", decide_more_hops_node)
    graph.add_node("follow_symbol", follow_symbol_node)
    graph.add_node("compose_answer", compose_answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "decide_more_hops")
    graph.add_conditional_edges(
        "decide_more_hops",
        route_after_decision,
        {"follow_symbol": "follow_symbol", "compose_answer": "compose_answer"},
    )
    graph.add_edge("follow_symbol", "decide_more_hops")
    graph.add_edge("compose_answer", END)

    return graph.compile()
