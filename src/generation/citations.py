import logging
import re
from dataclasses import dataclass

from src.ingestion.schemas import Chunk

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"([\w./-]+)#L(\d+)-L(\d+)")


@dataclass(frozen=True)
class Citation:
    """A file#Lstart-Lend span parsed from model output, with its verification result."""

    file_path: str
    start_line: int
    end_line: int
    verified: bool


def verify_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    """Parse every file#Lstart-Lend span in answer and mark it verified against real chunks."""
    citations = [
        Citation(
            file_path=file_path,
            start_line=int(start),
            end_line=int(end),
            verified=any(
                chunk.file_path == file_path
                and chunk.start_line <= int(start)
                and int(end) <= chunk.end_line
                for chunk in chunks
            ),
        )
        for file_path, start, end in CITATION_PATTERN.findall(answer)
    ]
    logger.info(
        "Parsed %d citation(s), %d verified",
        len(citations),
        sum(citation.verified for citation in citations),
    )
    return citations
