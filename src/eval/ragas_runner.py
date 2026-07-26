import logging
from dataclasses import dataclass

from langchain_core.callbacks import Callbacks
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas import evaluate
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
    SingleTurnSample,
    SingleTurnSampleOrMultiTurnSample,
)
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, LLMContextRecall

from src.config import GENERATION_MAX_TOKENS_DEFAULT
from src.eval.qa_dataset import QAItem
from src.generation.llm import LLMBackend

logger = logging.getLogger(__name__)

RAGAS_METRICS = [Faithfulness(), LLMContextPrecisionWithoutReference(), LLMContextRecall()]


@dataclass
class RagnarokRagasLLM(BaseRagasLLM):  # type: ignore[misc]  # ragas ships no py.typed marker
    """Wraps our own LLMBackend as a RAGAS judge model, avoiding any external judge API."""

    backend: LLMBackend | None = None

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        assert self.backend is not None, "RagnarokRagasLLM requires a backend"
        text = self.backend.generate(prompt.to_string(), max_tokens=GENERATION_MAX_TOKENS_DEFAULT)
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        return self.generate_text(
            prompt, n=n, temperature=temperature or 1e-8, stop=stop, callbacks=callbacks
        )


def run_ragas_eval(
    qa_items: list[QAItem],
    answers: list[str],
    contexts: list[list[str]],
    backend: LLMBackend,
) -> EvaluationResult:
    """Score each (question, generated answer, retrieved context) triple with RAGAS metrics."""
    samples: list[SingleTurnSampleOrMultiTurnSample] = [
        SingleTurnSample(user_input=item.question, response=answer, retrieved_contexts=context)
        for item, answer, context in zip(qa_items, answers, contexts)
    ]
    logger.info("Scoring %d item(s) with RAGAS", len(samples))
    dataset = EvaluationDataset(samples=samples)
    return evaluate(dataset, metrics=RAGAS_METRICS, llm=RagnarokRagasLLM(backend=backend))
