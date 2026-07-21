import numpy as np
import pytest

from src.ingestion.embedder import Embedder


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
