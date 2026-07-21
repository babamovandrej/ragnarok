from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Structural interface every generation backend (CPU, ZeroGPU) must satisfy."""

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate a completion for prompt, capped at max_tokens output tokens."""
        ...
