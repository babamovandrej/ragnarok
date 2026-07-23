import logging

from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

CITATION_INSTRUCTIONS = (
    "Answer only using the numbered context below. Cite every claim with the exact span it "
    'came from, in the form "file#Lstart-Lend" (e.g. src/foo.py#L10-L20). If the context does '
    'not contain the answer, say "I don\'t know" instead of guessing.'
)


def render_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Render a single-hop prompt: numbered context chunks, citation rules, then the question."""
    context = "\n\n".join(
        f"[{i}] {retrieved.chunk.file_path}#L{retrieved.chunk.start_line}-"
        f"L{retrieved.chunk.end_line}\n{retrieved.chunk.text}"
        for i, retrieved in enumerate(chunks, start=1)
    )
    logger.info("Rendered prompt from %d chunk(s)", len(chunks))
    return f"{CITATION_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
