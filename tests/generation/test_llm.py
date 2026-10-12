from src.generation.llm import GenerationBackend


class _MockBackend:
    context_window = 4096

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_tokens: int) -> str:
        return "mock answer"


class _GenerateOnly:
    def generate(self, prompt: str, max_tokens: int) -> str:
        return "mock answer"


class TestGenerationBackendProtocol:
    def test_a_class_with_generate_context_window_and_count_tokens_satisfies_it(self):
        assert isinstance(_MockBackend(), GenerationBackend)

    def test_a_class_that_can_only_generate_does_not_satisfy_it(self):
        assert isinstance(_GenerateOnly(), GenerationBackend) is False
