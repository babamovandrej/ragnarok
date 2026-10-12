import logging
import time
from typing import cast

from huggingface_hub import hf_hub_download
from llama_cpp import CreateChatCompletionResponse, Llama

from src.config import (
    GENERATION_CONTEXT_WINDOW,
    GENERATION_MAX_TOKENS_DEFAULT,
    GENERATION_MODEL_FILENAME,
    GENERATION_MODEL_REPO,
    GENERATION_MODEL_REVISION,
    GENERATION_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class CPUBackend:
    """Downloads the pinned GGUF chat model once and generates completions on the CPU."""

    def __init__(self) -> None:
        logger.info(
            "Loading CPU generation model %s/%s@%s",
            GENERATION_MODEL_REPO,
            GENERATION_MODEL_FILENAME,
            GENERATION_MODEL_REVISION,
        )
        started = time.monotonic()
        model_path = hf_hub_download(
            repo_id=GENERATION_MODEL_REPO,
            filename=GENERATION_MODEL_FILENAME,
            revision=GENERATION_MODEL_REVISION,
        )
        self._model: Llama = Llama(
            model_path=model_path, n_ctx=GENERATION_CONTEXT_WINDOW, verbose=False
        )
        logger.info("Loaded CPU generation model in %.1fs", time.monotonic() - started)

    @property
    def context_window(self) -> int:
        return GENERATION_CONTEXT_WINDOW

    def count_tokens(self, text: str) -> int:
        """Count tokens using this backend's own loaded tokenizer (llama.cpp's BPE vocab)."""
        return len(self._model.tokenize(text.encode("utf-8")))

    def generate(self, prompt: str, max_tokens: int = GENERATION_MAX_TOKENS_DEFAULT) -> str:
        """Generate a chat completion for prompt, capped at max_tokens output tokens.

        Decoding is greedy (temperature 0), which removes sampling randomness. Wording can
        still vary slightly between runs through floating-point effects.
        """
        response = cast(
            CreateChatCompletionResponse,
            self._model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=GENERATION_TEMPERATURE,
            ),
        )
        content = response["choices"][0]["message"]["content"]
        assert content is not None
        return content
