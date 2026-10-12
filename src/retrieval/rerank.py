import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    DEFAULT_DEVICE,
    RERANK_BATCH_SIZE,
    RERANK_INSTRUCTION,
    RERANK_MAX_LENGTH,
    RERANK_MODEL_ID,
    RERANK_MODEL_REVISION,
)
from src.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_pair(query: str, document: str, instruction: str = RERANK_INSTRUCTION) -> str:
    """Render one (query, document) pair in the model card's input format."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


class Reranker:
    """Loads the pinned Qwen3 reranker once and reorders candidates by relevance to a query.

    Qwen3-Reranker is a causal LM, not a sequence-classification cross-encoder: loading it
    via sentence-transformers' CrossEncoder (at the pinned version) attaches a randomly
    initialized classification head, yielding meaningless scores and failing outright on
    any batch > 1. It's loaded directly instead, and each candidate's score is the log-odds
    of "yes" vs. "no" as the next token — positive means the model judges it relevant.
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = device or DEFAULT_DEVICE or _default_device()
        logger.info(
            "Loading rerank model %s@%s (device=%s)",
            RERANK_MODEL_ID,
            RERANK_MODEL_REVISION,
            self._device,
        )
        started = time.monotonic()
        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            RERANK_MODEL_ID, revision=RERANK_MODEL_REVISION, padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            RERANK_MODEL_ID, revision=RERANK_MODEL_REVISION
        )
        self._model = model.to(self._device).eval()  # type: ignore[arg-type]
        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        self._prefix_ids = self._tokenizer.encode(_PREFIX, add_special_tokens=False)
        self._suffix_ids = self._tokenizer.encode(_SUFFIX, add_special_tokens=False)
        logger.info("Loaded rerank model in %.1fs", time.monotonic() - started)

    @torch.no_grad()
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return the log-odds that each document is relevant to query."""
        body_budget = RERANK_MAX_LENGTH - len(self._prefix_ids) - len(self._suffix_ids)
        scores: list[float] = []
        for start in range(0, len(documents), RERANK_BATCH_SIZE):
            batch = documents[start : start + RERANK_BATCH_SIZE]
            encoded = self._tokenizer(
                [format_pair(query, document) for document in batch],
                add_special_tokens=False,
                truncation="longest_first",
                max_length=body_budget,
                return_attention_mask=False,
            )
            input_ids = [self._prefix_ids + ids + self._suffix_ids for ids in encoded["input_ids"]]
            padded = self._tokenizer.pad(
                {"input_ids": input_ids}, padding=True, return_tensors="pt"
            ).to(self._device)
            last_logits = self._model(**padded).logits[:, -1, :].float()
            log_odds = last_logits[:, self._yes_id] - last_logits[:, self._no_id]
            scores.extend(log_odds.tolist())
        return scores

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Reorder candidates by the reranker's relevance score for each (query, text) pair."""
        if not candidates:
            return []
        scores = self.score(query, [candidate.chunk.search_text for candidate in candidates])
        reranked = [
            RetrievedChunk(chunk=candidate.chunk, score=score)
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda retrieved: retrieved.score, reverse=True)
        return reranked
