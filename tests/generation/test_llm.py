from src.generation.llm import LLMBackend


class _MockBackend:
    def __init__(self, response: str = "mock answer") -> None:
        self.response = response
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.calls.append((prompt, max_tokens))
        return self.response


class _NotABackend:
    pass


class TestLLMBackendProtocol:
    def test_a_class_implementing_generate_satisfies_the_protocol(self):
        assert isinstance(_MockBackend(), LLMBackend)

    def test_a_class_without_generate_does_not_satisfy_the_protocol(self):
        assert isinstance(_NotABackend(), LLMBackend) is False

    def test_generate_receives_the_prompt_and_max_tokens_and_returns_a_string(self):
        backend = _MockBackend("the answer")

        result = backend.generate("my prompt", max_tokens=64)

        assert result == "the answer"
        assert backend.calls == [("my prompt", 64)]
