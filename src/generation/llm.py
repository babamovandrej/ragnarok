from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Structural interface every generation backend (CPU, ZeroGPU) must satisfy."""

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate a completion for prompt, capped at max_tokens output tokens."""
        ...


@runtime_checkable
class GenerationBackend(LLMBackend, Protocol):
    """An LLMBackend that can also report its context window and count tokens in text.

    Needed wherever a prompt must be sized to fit a specific backend's context window
    before generation (see fit_chunks_to_budget in prompts.py) — plain LLMBackend doesn't
    carry enough information for that. llm_decide_more_hops and answer_naive don't need
    this: their prompts are small/fixed and never risk overflowing a context window.
    """

    @property
    def context_window(self) -> int:
        """This backend's context window, in tokens."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return how many tokens text would consume in this backend's own tokenizer."""
        ...
