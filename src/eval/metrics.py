IDK_PHRASES = (
    "i don't know",
    "i do not know",
    "cannot find",
    "can't find",
    "not found in",
    "no information",
    "not contain",
    "not in the",
)


def is_idk_response(answer: str) -> bool:
    """Check whether answer is an explicit 'I don't know'-style refusal rather than a guess."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in IDK_PHRASES)


def idk_rate(answers: list[str]) -> float:
    """Fraction of answers that are explicit 'I don't know' responses; 0.0 for an empty list."""
    if not answers:
        return 0.0
    return sum(is_idk_response(answer) for answer in answers) / len(answers)
