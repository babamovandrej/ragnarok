from pathlib import Path
from typing import Literal

from pydantic import BaseModel

QACategory = Literal["single_hop", "multi_hop", "adversarial"]


class QAItem(BaseModel):
    """One hand-labeled eval question: what's expected, and which files should ground it."""

    id: str
    question: str
    category: QACategory
    expected_files: list[str]
    grounding_notes: str
    expect_idk: bool = False


def load_qa_dataset(path: Path) -> list[QAItem]:
    """Load and validate the hand-labeled Q&A dataset; one JSON object per line."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [QAItem.model_validate_json(line) for line in lines]
