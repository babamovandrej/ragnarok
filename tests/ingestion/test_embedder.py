from unittest.mock import patch

import numpy as np
import pytest

from src.config import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION, EMBEDDING_QUERY_PROMPT
from src.ingestion.embedder import Embedder


@pytest.fixture
def mock_model():
    with patch("src.ingestion.embedder.SentenceTransformer") as sentence_transformer:
        yield sentence_transformer


class TestEmbedder:
    def test_pins_model_id_and_revision_on_load(self, mock_model):
        Embedder(device="cpu")

        mock_model.assert_called_once_with(
            EMBEDDING_MODEL_ID, revision=EMBEDDING_MODEL_REVISION, device="cpu"
        )

    def test_falls_back_to_the_configured_default_device(self, mock_model, monkeypatch):
        monkeypatch.setattr("src.ingestion.embedder.DEFAULT_DEVICE", "cpu")

        Embedder()

        assert mock_model.call_args.kwargs["device"] == "cpu"

    def test_an_explicit_device_overrides_the_configured_default(self, mock_model, monkeypatch):
        monkeypatch.setattr("src.ingestion.embedder.DEFAULT_DEVICE", "cpu")

        Embedder(device="mps")

        assert mock_model.call_args.kwargs["device"] == "mps"

    def test_embeds_chunk_texts_without_a_prompt(self, mock_model):
        mock_model.return_value.encode.return_value = np.zeros((2, 4), dtype=np.float32)

        embeddings = Embedder(batch_size=2).embed(["a", "b"])

        assert embeddings.shape == (2, 4)
        assert "prompt" not in mock_model.return_value.encode.call_args.kwargs
        assert mock_model.return_value.encode.call_args.kwargs["batch_size"] == 2

    def test_embeds_a_query_with_the_code_search_instruction(self, mock_model):
        mock_model.return_value.encode.return_value = np.ones((1, 4), dtype=np.float32)

        embedding = Embedder().embed_query("where is auth handled?")

        assert embedding.shape == (4,)
        assert mock_model.return_value.encode.call_args.args == (["where is auth handled?"],)
        assert mock_model.return_value.encode.call_args.kwargs["prompt"] == EMBEDDING_QUERY_PROMPT


@pytest.mark.integration
def test_embed_returns_correct_shape_with_no_nans():
    embedder = Embedder()
    texts = [
        "def add(a, b):\n    return a + b",
        "class Foo:\n    pass",
        "def sub(a, b):\n    return a - b",
        "def mul(a, b):\n    return a * b",
        "def div(a, b):\n    return a / b",
    ]

    embeddings = embedder.embed(texts)

    assert embeddings.shape == (5, embeddings.shape[1])
    assert embeddings.shape[1] > 0
    assert not np.isnan(embeddings).any()
