from typing import Protocol, runtime_checkable


@runtime_checkable
class GenerationBackend(Protocol):
    """Structural interface every generation backend must satisfy.

    Besides generating, a backend reports its context window and counts tokens, so a prompt
    can be sized to fit before generation (see select_blocks_that_fit in prompts.py).
    """

    @property
    def context_window(self) -> int:
        """This backend's context window, in tokens."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return how many tokens text would consume in this backend's own tokenizer."""
        ...

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate a completion for prompt, capped at max_tokens output tokens."""
        ...
