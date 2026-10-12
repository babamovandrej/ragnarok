import logging

from src.config import GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS
from src.generation.llm import GenerationBackend
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

PLACEHOLDER_LOCATION = "path/to/file.py#L10-L20"
BLOCK_SEPARATOR = "\n\n"


def citation_instructions(example_location: str) -> str:
    """Answering rules, with an example citation built from a real location in the context.

    Small models copy the example's path verbatim, so a made-up example path ends up in
    their citations; a location taken from the actual context makes the copy correct.
    """
    return (
        "Answer only using the numbered context below. Each context item is headed by its "
        f'exact location, for example "{example_location}". Answer in one to three full '
        "sentences, and after each claim cite the location string of the context item it "
        "came from, copied exactly. If the context does not contain the answer, say \"I don't "
        'know" instead of guessing.\n\n'
        "Example of a correctly cited answer:\n"
        f"The `handler` function validates the request before dispatching it ({example_location})."
    )


def citation_reminder(example_location: str) -> str:
    return (
        "Reminder: write full sentences and cite each claim with a location string copied "
        f"from the context items above, like ({example_location})."
    )


NOT_IN_REPO_ANSWER = "I don't know — this doesn't appear to be covered by the indexed repo."


def _location(retrieved: RetrievedChunk) -> str:
    chunk = retrieved.chunk
    return f"{chunk.file_path}#L{chunk.start_line}-L{chunk.end_line}"


def _chunk_block(index: int, retrieved: RetrievedChunk) -> str:
    """Render one numbered context item: its location header, optional imports, then its text.

    Imports are shown here only — never embedded or reranked (see Chunk.imports in
    ingestion/schemas.py) — so a function chunk retrieved in isolation still tells the model
    which module a bare name like `Request` resolves to.
    """
    header = f"[{index}] {_location(retrieved)}"
    if retrieved.chunk.imports:
        return f"{header}\nImports:\n{retrieved.chunk.imports}\n\n{retrieved.chunk.text}"
    return f"{header}\n{retrieved.chunk.text}"


def _render(question: str, context: str, example_location: str) -> str:
    return (
        f"{citation_instructions(example_location)}\n\nContext:\n{context}\n\n"
        f"Question: {question}\n{citation_reminder(example_location)}\nAnswer:"
    )


def render_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Render a single-hop prompt: numbered context chunks, citation rules, then the question."""
    context = BLOCK_SEPARATOR.join(
        _chunk_block(i, retrieved) for i, retrieved in enumerate(chunks, start=1)
    )
    example = _location(chunks[0]) if chunks else PLACEHOLDER_LOCATION
    logger.info("Rendered prompt from %d chunk(s)", len(chunks))
    return _render(question, context, example)


def select_blocks_that_fit(
    scaffold: str, blocks: list[str], backend: GenerationBackend, max_output_tokens: int
) -> list[int]:
    """Indices of the blocks that fit in backend's context window alongside scaffold.

    Reserves max_output_tokens for the completion and GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS
    for the chat template the backend wraps the prompt in, which count_tokens can't see.
    Blocks are considered in order, so callers pass them highest-priority first; a block too
    large for the remaining budget is skipped rather than ending the selection, so one
    oversized chunk can't leave the model with no context at all.
    """
    budget = (
        backend.context_window
        - max_output_tokens
        - GENERATION_CHAT_TEMPLATE_RESERVE_TOKENS
        - backend.count_tokens(scaffold)
    )
    kept: list[int] = []
    used = 0
    for index, block in enumerate(blocks):
        block_tokens = backend.count_tokens(block + BLOCK_SEPARATOR)
        if used + block_tokens <= budget:
            kept.append(index)
            used += block_tokens
    if len(kept) < len(blocks):
        logger.warning(
            "Trimmed context from %d to %d chunk(s) to fit the %d-token context window",
            len(blocks),
            len(kept),
            backend.context_window,
        )
    return kept


def fit_chunks_to_budget(
    question: str,
    chunks: list[RetrievedChunk],
    backend: GenerationBackend,
    max_output_tokens: int,
) -> list[RetrievedChunk]:
    """Drop the chunks that don't fit so the rendered prompt fits backend's context window.

    Chunks arrive highest-priority first. Dropping whole chunks here, rather than letting the
    backend truncate an overlong prompt, keeps the question and citation instructions at the
    end of the prompt intact.
    """
    blocks = [_chunk_block(i, retrieved) for i, retrieved in enumerate(chunks, start=1)]
    longest_location = max((_location(c) for c in chunks), key=len, default=PLACEHOLDER_LOCATION)
    scaffold = _render(question, "", longest_location)
    kept = select_blocks_that_fit(scaffold, blocks, backend, max_output_tokens)
    return [chunks[index] for index in kept]
