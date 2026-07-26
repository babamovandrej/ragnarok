import gradio as gr
import pytest

from src.generation.fallback_backend import FallbackBackend


class _FakeBackend:
    def __init__(
        self,
        response: str | None = None,
        raises: Exception | None = None,
        context_window: int = 4096,
    ) -> None:
        self.response = response
        self.raises = raises
        self.context_window = context_window
        self.calls: list[tuple[str, int]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

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


class TestFallbackBackendContextWindow:
    def test_uses_the_smaller_of_the_two_context_windows(self):
        primary = _FakeBackend(context_window=32768)
        fallback = _FakeBackend(context_window=4096)
        backend = FallbackBackend(primary, fallback)

        assert backend.context_window == 4096

    def test_uses_the_smaller_window_regardless_of_which_side_its_on(self):
        primary = _FakeBackend(context_window=4096)
        fallback = _FakeBackend(context_window=32768)
        backend = FallbackBackend(primary, fallback)

        assert backend.context_window == 4096


class _TaggedBackend(_FakeBackend):
    """A fake whose count_tokens returns a fixed sentinel, to identify which one was called."""

    def __init__(self, tag: int, context_window: int) -> None:
        super().__init__(context_window=context_window)
        self._tag = tag

    def count_tokens(self, text: str) -> int:
        return self._tag


class TestFallbackBackendCountTokens:
    def test_counts_using_primary_when_it_has_the_smaller_context_window(self):
        primary = _TaggedBackend(tag=1, context_window=4096)
        fallback = _TaggedBackend(tag=2, context_window=32768)
        backend = FallbackBackend(primary, fallback)

        assert backend.count_tokens("text") == 1

    def test_counts_using_fallback_when_it_has_the_smaller_context_window(self):
        primary = _TaggedBackend(tag=1, context_window=32768)
        fallback = _TaggedBackend(tag=2, context_window=4096)
        backend = FallbackBackend(primary, fallback)

        assert backend.count_tokens("text") == 2
