import logging
from collections.abc import Callable
from typing import NamedTuple, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.state import AgentState, HopRecord
from src.config import (
    AGENT_HOP_DECISION_MAX_TOKENS,
    GENERATION_MAX_TOKENS_DEFAULT,
    MIN_CONTEXT_RELEVANCE_SCORE,
)
from src.generation.llm import GenerationBackend, LLMBackend
from src.generation.prompts import NOT_IN_REPO_ANSWER, fit_chunks_to_budget, render_prompt
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str], list[RetrievedChunk]]
FindSymbolFn = Callable[[str], RetrievedChunk | None]


class HopDecision(NamedTuple):
    """A parsed hop-decision reply: the symbol to follow (None means stop), and why."""

    symbol: str | None
    reason: str


DecideMoreHopsFn = Callable[[AgentState], HopDecision]


class _AgentStateUpdate(TypedDict, total=False):
    """Partial update applied to `AgentState` after a graph node runs."""

    context: list[RetrievedChunk]
    hop_path: list[HopRecord]
    pending_symbol: str | None
    pending_reason: str | None
    hops_remaining: int
    answer: str


_SYMBOL_LINE_PREFIX = "symbol:"
_REASON_LINE_PREFIX = "reason:"

DECIDE_MORE_HOPS_PROMPT = (
    "You are deciding whether one more symbol lookup would help answer the question, "
    "given the context already retrieved. Reply on exactly two lines:\n"
    "SYMBOL: <the symbol name to look up, or DONE if the context is already sufficient>\n"
    "REASON: <one short sentence explaining why>\n\n"
    "Question: {question}\n\nContext so far:\n{context}\n\nReply:"
)


def _parse_hop_decision(reply: str) -> HopDecision:
    """Parse a SYMBOL:/REASON: reply; tolerates a model that ignores the requested format."""
    symbol_text = ""
    reason_text = ""
    for line in reply.strip().splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith(_SYMBOL_LINE_PREFIX):
            symbol_text = stripped[len(_SYMBOL_LINE_PREFIX) :].strip()
        elif lowered.startswith(_REASON_LINE_PREFIX):
            reason_text = stripped[len(_REASON_LINE_PREFIX) :].strip()

    if not symbol_text and not reason_text:
        # The model ignored the two-line format; fall back to treating the whole reply as a
        # bare symbol guess, matching this function's pre-task-11 behavior.
        symbol_text = reply.strip()

    if not symbol_text or symbol_text.upper() == "DONE":
        return HopDecision(
            symbol=None, reason=reason_text or "model indicated no further hop is needed"
        )
    return HopDecision(symbol=symbol_text, reason=reason_text or "no reason given")


def llm_decide_more_hops(backend: LLMBackend, state: AgentState) -> HopDecision:
    """Ask backend whether another symbol lookup would help; symbol=None means stop hopping."""
    context = "\n".join(f"- {c.chunk.qualified_name} ({c.chunk.file_path})" for c in state.context)
    prompt = DECIDE_MORE_HOPS_PROMPT.format(question=state.question, context=context)
    reply = backend.generate(prompt, max_tokens=AGENT_HOP_DECISION_MAX_TOKENS)
    decision = _parse_hop_decision(reply)
    logger.debug("Hop decision prompt: %s", prompt)
    logger.debug("Hop decision raw reply: %r -> %r", reply, decision)
    return decision


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
            logger.info("No hops remaining; stopping without asking the model")
            return {"pending_symbol": None, "pending_reason": None}
        decision = decide_more_hops(state)
        if decision.symbol is None:
            logger.info("Hop decision: model chose DONE (%s)", decision.reason)
        return {"pending_symbol": decision.symbol, "pending_reason": decision.reason}

    def route_after_decision(state: AgentState) -> str:
        return "follow_symbol" if state.pending_symbol else "compose_answer"

    def follow_symbol_node(state: AgentState) -> _AgentStateUpdate:
        symbol = state.pending_symbol
        assert symbol is not None, "follow_symbol_node reached without a pending symbol"
        try:
            found = find_symbol(symbol)
        except Exception:
            logger.exception("Symbol lookup failed for %r; stopping hops", symbol)
            found = None
        if found is None:
            logger.info("Could not resolve symbol %s; stopping hops", symbol)
            return {
                "pending_symbol": None,
                "pending_reason": None,
                "hops_remaining": state.hops_remaining - 1,
            }
        hop = HopRecord(
            file_path=found.chunk.file_path,
            symbol_name=found.chunk.symbol_name,
            start_line=found.chunk.start_line,
            end_line=found.chunk.end_line,
            reason=state.pending_reason or f"Followed reference to {symbol}",
        )
        return {
            "context": [*state.context, found],
            "hop_path": [*state.hop_path, hop],
            "pending_symbol": None,
            "pending_reason": None,
            "hops_remaining": state.hops_remaining - 1,
        }

    def compose_answer_node(state: AgentState) -> _AgentStateUpdate:
        best_score = max((c.score for c in state.context), default=float("-inf"))
        if best_score < MIN_CONTEXT_RELEVANCE_SCORE:
            logger.info(
                "Best context score %.3f is below the relevance threshold %.3f; answering "
                "without calling the model",
                best_score,
                MIN_CONTEXT_RELEVANCE_SCORE,
            )
            return {"answer": NOT_IN_REPO_ANSWER}

        fitted_context = fit_chunks_to_budget(
            state.question, state.context, backend, GENERATION_MAX_TOKENS_DEFAULT
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
