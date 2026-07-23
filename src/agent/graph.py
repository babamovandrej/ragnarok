import logging
from collections.abc import Callable

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.state import AgentState, HopRecord
from src.config import GENERATION_MAX_TOKENS_DEFAULT
from src.generation.llm import LLMBackend
from src.generation.prompts import render_prompt
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str], list[RetrievedChunk]]
FindSymbolFn = Callable[[str], RetrievedChunk | None]
DecideMoreHopsFn = Callable[[AgentState], str | None]

DECIDE_MORE_HOPS_PROMPT = (
    "You are deciding whether one more symbol lookup would help answer the question, "
    "given the context already retrieved. If a lookup would help, reply with exactly "
    "that symbol's name and nothing else. If the context is already sufficient, reply "
    "with exactly DONE.\n\nQuestion: {question}\n\nContext so far:\n{context}\n\nReply:"
)


def llm_decide_more_hops(backend: LLMBackend, state: AgentState) -> str | None:
    """Ask backend whether another symbol lookup would help; None means stop hopping."""
    context = "\n".join(f"- {c.chunk.qualified_name} ({c.chunk.file_path})" for c in state.context)
    prompt = DECIDE_MORE_HOPS_PROMPT.format(question=state.question, context=context)
    reply = backend.generate(prompt, max_tokens=32).strip()
    return None if reply.upper() == "DONE" else reply


def build_graph(
    retrieve: RetrieveFn,
    find_symbol: FindSymbolFn,
    decide_more_hops: DecideMoreHopsFn,
    backend: LLMBackend,
) -> CompiledStateGraph:
    """Wire retrieve -> decide_more_hops -> follow_symbol (looped) -> compose_answer."""

    def retrieve_node(state: AgentState) -> dict:
        context = retrieve(state.question)
        logger.info("Initial retrieval returned %d chunk(s)", len(context))
        return {"context": context}

    def decide_more_hops_node(state: AgentState) -> dict:
        if state.hops_remaining <= 0:
            return {"pending_symbol": None}
        return {"pending_symbol": decide_more_hops(state)}

    def route_after_decision(state: AgentState) -> str:
        return "follow_symbol" if state.pending_symbol else "compose_answer"

    def follow_symbol_node(state: AgentState) -> dict:
        symbol = state.pending_symbol
        found = find_symbol(symbol)
        if found is None:
            logger.info("Could not resolve symbol %s; stopping hops", symbol)
            return {"pending_symbol": None, "hops_remaining": state.hops_remaining - 1}
        hop = HopRecord(
            file_path=found.chunk.file_path,
            symbol_name=found.chunk.symbol_name,
            start_line=found.chunk.start_line,
            end_line=found.chunk.end_line,
            reason=f"Followed reference to {symbol}",
        )
        return {
            "context": [*state.context, found],
            "hop_path": [*state.hop_path, hop],
            "pending_symbol": None,
            "hops_remaining": state.hops_remaining - 1,
        }

    def compose_answer_node(state: AgentState) -> dict:
        prompt = render_prompt(state.question, state.context)
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
