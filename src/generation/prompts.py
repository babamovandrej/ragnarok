import logging

from src.generation.llm import GenerationBackend
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

CITATION_INSTRUCTIONS = (
    "Answer only using the numbered context below. Each context item is headed by its exact "
    'location, for example "src/foo.py#L10-L20". Cite every claim by copying the exact '
    "location string from the context item it came from. If the context does not contain the "
    'answer, say "I don\'t know" instead of guessing.'
)

CITATION_REMINDER = (
    "Reminder: end your answer with the exact location string(s), copied from the context "
    "items above, that support it."
)

NOT_IN_REPO_ANSWER = "I don't know — this doesn't appear to be covered by the indexed repo."


def _chunk_block(index: int, retrieved: RetrievedChunk) -> str:
    """Render one numbered context item: its location header, optional imports, then its text.

    Imports are shown here only — never embedded or reranked (see Chunk.imports in
    ingestion/schemas.py) — so a function chunk retrieved in isolation still tells the model
    which module a bare name like `Request` resolves to.
    """
    header = (
        f"[{index}] {retrieved.chunk.file_path}#L{retrieved.chunk.start_line}-"
        f"L{retrieved.chunk.end_line}"
    )
    if retrieved.chunk.imports:
        return f"{header}\nImports:\n{retrieved.chunk.imports}\n\n{retrieved.chunk.text}"
    return f"{header}\n{retrieved.chunk.text}"


def render_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Render a single-hop prompt: numbered context chunks, citation rules, then the question."""
    context = "\n\n".join(_chunk_block(i, retrieved) for i, retrieved in enumerate(chunks, start=1))
    logger.info("Rendered prompt from %d chunk(s)", len(chunks))
    return (
        f"{CITATION_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}\n"
        f"{CITATION_REMINDER}\nAnswer:"
    )


def fit_chunks_to_budget(
    question: str,
    chunks: list[RetrievedChunk],
    backend: GenerationBackend,
    max_output_tokens: int,
) -> list[RetrievedChunk]:
    """Trim chunks (lowest-relevance last) so the rendered prompt fits backend's context window.

    Reserves max_output_tokens for the completion; whatever's left after the fixed prompt
    scaffolding (instructions + question, no context) is the budget for chunk content.
    Chunks arrive already ranked highest-relevance-first, so trimming drops from the tail —
    this avoids letting the backend's own context truncation silently drop the question or
    citation instructions from the *end* of an overlong prompt.
    """
    scaffold_tokens = backend.count_tokens(render_prompt(question, []))
    budget = backend.context_window - max_output_tokens - scaffold_tokens
    if budget <= 0:
        logger.warning(
            "No token budget left for context (%d-token window, %d reserved for output, %d "
            "for prompt scaffolding); answering with no retrieved context",
            backend.context_window,
            max_output_tokens,
            scaffold_tokens,
        )
        return []

    kept: list[RetrievedChunk] = []
    used = 0
    for index, retrieved in enumerate(chunks, start=1):
        block_tokens = backend.count_tokens(_chunk_block(index, retrieved))
        if used + block_tokens > budget:
            break
        kept.append(retrieved)
        used += block_tokens

    if len(kept) < len(chunks):
        logger.warning(
            "Trimmed context from %d to %d chunk(s) to fit the %d-token context window",
            len(chunks),
            len(kept),
            backend.context_window,
        )
    return kept
