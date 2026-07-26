from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config import ZEROGPU_MODEL_ID, ZEROGPU_MODEL_REVISION
from src.generation.zerogpu_backend import ZeroGPUBackend


@pytest.fixture
def mock_tokenizer():
    with patch("src.generation.zerogpu_backend.AutoTokenizer") as mock_tokenizer:
        yield mock_tokenizer


@pytest.fixture
def mock_model():
    with patch("src.generation.zerogpu_backend.AutoModelForCausalLM") as mock_model:
        yield mock_model


class TestZeroGPUBackend:
    def test_pins_model_id_and_revision_for_the_tokenizer(self, mock_tokenizer, mock_model):
        ZeroGPUBackend()

        mock_tokenizer.from_pretrained.assert_called_once_with(
            ZEROGPU_MODEL_ID, revision=ZEROGPU_MODEL_REVISION
        )

    def test_pins_model_id_and_revision_for_the_model_and_moves_it_to_cuda(
        self, mock_tokenizer, mock_model
    ):
        ZeroGPUBackend()

        mock_model.from_pretrained.assert_called_once_with(
            ZEROGPU_MODEL_ID, revision=ZEROGPU_MODEL_REVISION, torch_dtype=torch.bfloat16
        )
        mock_model.from_pretrained.return_value.to.assert_called_once_with("cuda")

    def test_generate_decodes_the_models_output_and_passes_max_tokens_through(
        self, mock_tokenizer, mock_model
    ):
        class _FakeBatchEncoding(dict):
            def to(self, device):
                return self

        backend = ZeroGPUBackend()
        fake_inputs = _FakeBatchEncoding({"input_ids": MagicMock(shape=(1, 3))})
        mock_tokenizer.from_pretrained.return_value.return_value = fake_inputs
        mock_tokenizer.from_pretrained.return_value.decode.return_value = "the answer"
        mock_model.from_pretrained.return_value.to.return_value.generate.return_value = [
            MagicMock()
        ]

        result = backend.generate("a prompt", max_tokens=16)

        assert result == "the answer"
        generate_mock = mock_model.from_pretrained.return_value.to.return_value.generate
        assert generate_mock.call_args.kwargs["max_new_tokens"] == 16

    def test_context_window_reads_the_models_own_config(self, mock_tokenizer, mock_model):
        moved_model = mock_model.from_pretrained.return_value.to.return_value
        moved_model.config.max_position_embeddings = 32768
        backend = ZeroGPUBackend()

        assert backend.context_window == 32768

    def test_count_tokens_delegates_to_the_loaded_tokenizer(self, mock_tokenizer, mock_model):
        mock_tokenizer.from_pretrained.return_value.encode.return_value = [1, 2, 3]
        backend = ZeroGPUBackend()

        assert backend.count_tokens("some text") == 3
        mock_tokenizer.from_pretrained.return_value.encode.assert_called_once_with("some text")
