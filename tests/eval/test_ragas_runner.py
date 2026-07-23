import asyncio

import pytest
from langchain_core.prompt_values import StringPromptValue

from src.eval.qa_dataset import QAItem
from src.eval.ragas_runner import RagnarokRagasLLM, run_ragas_eval


class _FakeBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.calls.append((prompt, max_tokens))
        return self.response


class TestRagnarokRagasLLM:
    def test_generate_text_wraps_the_backends_response_in_an_llm_result(self):
        judge = RagnarokRagasLLM(backend=_FakeBackend("the judged answer"))

        result = judge.generate_text(StringPromptValue(text="judge this"))

        assert result.generations[0][0].text == "the judged answer"

    def test_generate_text_passes_the_prompt_as_a_plain_string(self):
        backend = _FakeBackend("ok")
        judge = RagnarokRagasLLM(backend=backend)

        judge.generate_text(StringPromptValue(text="judge this"))

        assert backend.calls[0][0] == "judge this"

    def test_agenerate_text_delegates_to_generate_text(self):
        judge = RagnarokRagasLLM(backend=_FakeBackend("async answer"))

        result = asyncio.run(judge.agenerate_text(StringPromptValue(text="judge this")))

        assert result.generations[0][0].text == "async answer"


@pytest.mark.integration
def test_run_ragas_eval_scores_a_real_item_with_the_real_cpu_backend():
    from src.generation.cpu_backend import CPUBackend

    item = QAItem(
        id="x",
        question="Where is AuthenticationMiddleware defined?",
        category="single_hop",
        expected_files=["starlette/middleware/authentication.py"],
        grounding_notes="n",
    )

    result = run_ragas_eval(
        [item],
        ["It's defined in starlette/middleware/authentication.py#L16-L52."],
        [["class AuthenticationMiddleware: ..."]],
        CPUBackend(),
    )

    assert result.to_pandas() is not None
