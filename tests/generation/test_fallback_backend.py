import gradio as gr
import pytest

from src.generation.fallback_backend import FallbackBackend


class _FakeBackend:
    def __init__(self, response: str | None = None, raises: Exception | None = None) -> None:
        self.response = response
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.calls.append((prompt, max_tokens))
        if self.raises is not None:
            raise self.raises
        return self.response


class TestFallbackBackend:
    def test_uses_the_primary_backend_when_it_succeeds(self):
        primary = _FakeBackend(response="primary answer")
        fallback = _FakeBackend(response="fallback answer")
        backend = FallbackBackend(primary, fallback)

        result = backend.generate("prompt", max_tokens=32)

        assert result == "primary answer"
        assert backend.last_backend_used == "primary"
        assert fallback.calls == []

    def test_falls_back_to_cpu_on_a_gradio_error(self):
        primary = _FakeBackend(raises=gr.Error("ZeroGPU quota exceeded"))
        fallback = _FakeBackend(response="fallback answer")
        backend = FallbackBackend(primary, fallback)

        result = backend.generate("prompt", max_tokens=32)

        assert result == "fallback answer"
        assert backend.last_backend_used == "fallback"
        assert fallback.calls == [("prompt", 32)]

    def test_does_not_swallow_non_gradio_errors(self):
        primary = _FakeBackend(raises=RuntimeError("unrelated failure"))
        fallback = _FakeBackend(response="fallback answer")
        backend = FallbackBackend(primary, fallback)

        with pytest.raises(RuntimeError):
            backend.generate("prompt", max_tokens=32)
