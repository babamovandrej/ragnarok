from unittest.mock import patch

import pytest

from src.config import (
    GENERATION_CONTEXT_WINDOW,
    GENERATION_MODEL_FILENAME,
    GENERATION_MODEL_REPO,
    GENERATION_MODEL_REVISION,
)
from src.generation.cpu_backend import CPUBackend


@pytest.fixture
def mock_download():
    with patch("src.generation.cpu_backend.hf_hub_download") as mock_download:
        mock_download.return_value = "/fake/model.gguf"
        yield mock_download


@pytest.fixture
def mock_llama():
    with patch("src.generation.cpu_backend.Llama") as mock_llama:
        yield mock_llama


class TestCPUBackend:
    def test_downloads_the_pinned_model_repo_filename_and_revision(
        self, mock_download, mock_llama
    ):
        CPUBackend()

        mock_download.assert_called_once_with(
            repo_id=GENERATION_MODEL_REPO,
            filename=GENERATION_MODEL_FILENAME,
            revision=GENERATION_MODEL_REVISION,
        )

    def test_loads_llama_from_the_downloaded_path_with_the_configured_context_window(
        self, mock_download, mock_llama
    ):
        CPUBackend()

        mock_llama.assert_called_once_with(
            model_path="/fake/model.gguf", n_ctx=GENERATION_CONTEXT_WINDOW, verbose=False
        )

    def test_generate_sends_the_prompt_as_a_single_user_message(self, mock_download, mock_llama):
        mock_llama.return_value.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "the answer"}}]
        }
        backend = CPUBackend()

        result = backend.generate("my prompt", max_tokens=64)

        assert result == "the answer"
        mock_llama.return_value.create_chat_completion.assert_called_once_with(
            messages=[{"role": "user", "content": "my prompt"}], max_tokens=64
        )


@pytest.mark.integration
def test_real_model_generates_a_completion_for_a_fixed_prompt():
    backend = CPUBackend()

    result = backend.generate("Say the word 'hello' and nothing else.", max_tokens=16)

    assert isinstance(result, str)
    assert result.strip()
