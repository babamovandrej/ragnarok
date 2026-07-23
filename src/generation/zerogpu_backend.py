import logging
import time

import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import GENERATION_MAX_TOKENS_DEFAULT, ZEROGPU_MODEL_ID, ZEROGPU_MODEL_REVISION

logger = logging.getLogger(__name__)


class ZeroGPUBackend:
    """Loads the pinned 7B chat model once and generates on a ZeroGPU-allocated worker."""

    def __init__(self) -> None:
        logger.info(
            "Loading ZeroGPU generation model %s@%s", ZEROGPU_MODEL_ID, ZEROGPU_MODEL_REVISION
        )
        started = time.monotonic()
        self._tokenizer = AutoTokenizer.from_pretrained(
            ZEROGPU_MODEL_ID, revision=ZEROGPU_MODEL_REVISION
        )
        model = AutoModelForCausalLM.from_pretrained(
            ZEROGPU_MODEL_ID, revision=ZEROGPU_MODEL_REVISION, torch_dtype=torch.bfloat16
        )
        self._model = model.to("cuda")  # type: ignore[arg-type]  # transformers' AutoModel stub mistypes .to()
        logger.info("Loaded ZeroGPU generation model in %.1fs", time.monotonic() - started)

    @spaces.GPU
    def generate(self, prompt: str, max_tokens: int = GENERATION_MAX_TOKENS_DEFAULT) -> str:
        """Generate a completion for prompt; runs inside a ZeroGPU-allocated GPU worker."""
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self._model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)
