import logging

import gradio as gr

from src.generation.llm import LLMBackend

logger = logging.getLogger(__name__)


class FallbackBackend:
    """Tries primary (ZeroGPU) first; falls back to CPU on a gr.Error (ZeroGPU quota exceeded)."""

    def __init__(self, primary: LLMBackend, fallback: LLMBackend) -> None:
        self._primary = primary
        self._fallback = fallback
        self.last_backend_used: str | None = None

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
