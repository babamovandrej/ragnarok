from src.retrieval.schemas import RetrievedChunk


def ranked_files(retrieved: list[RetrievedChunk]) -> list[str]:
    """File paths in retrieval order, keeping only each file's first (best) appearance."""
    return list(dict.fromkeys(r.chunk.file_path for r in retrieved))


def hit_at_k(files: list[str], expected: list[str], k: int) -> float:
    """1.0 if any expected file appears among the top k distinct files, else 0.0."""
    return float(any(f in expected for f in files[:k]))


def reciprocal_rank(files: list[str], expected: list[str]) -> float:
    """1/rank of the first expected file among files, or 0.0 if none was retrieved."""
    for rank, file_path in enumerate(files, start=1):
        if file_path in expected:
            return 1.0 / rank
    return 0.0


def file_recall(files: list[str], expected: list[str]) -> float:
    """Fraction of expected files retrieved at all — what multi-hop questions need."""
    if not expected:
        return 0.0
    return sum(f in files for f in expected) / len(expected)
