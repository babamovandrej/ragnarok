import logging

import gradio as gr

from src.generation.llm import GenerationBackend

logger = logging.getLogger(__name__)


class FallbackBackend:
    """Tries primary (ZeroGPU) first; falls back to CPU on a gr.Error (ZeroGPU quota exceeded)."""

    def __init__(self, primary: GenerationBackend, fallback: GenerationBackend) -> None:
        self._primary = primary
        self._fallback = fallback
        self.last_backend_used: str | None = None

    @property
    def context_window(self) -> int:
        """The smaller of the two backends' windows, since either may end up answering."""
        return min(self._primary.context_window, self._fallback.context_window)

    def count_tokens(self, text: str) -> int:
        """Count using whichever backend has the smaller (more binding) context window."""
        binding = (
            self._fallback
            if self._fallback.context_window <= self._primary.context_window
            else self._primary
        )
        return binding.count_tokens(text)

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate via primary; on a gr.Error, transparently retry via fallback."""
        try:
            answer = self._primary.generate(prompt, max_tokens)
            self.last_backend_used = "primary"
            return answer
        except gr.Error as exc:
            logger.warning("Primary backend unavailable (%s); falling back to CPU", exc)
            answer = self._fallback.generate(prompt, max_tokens)
            self.last_backend_used = "fallback"
            return answer
