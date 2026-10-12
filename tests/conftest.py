import numpy as np
import pytest


class _FakeEmbedder:
    """Stands in for the real (multi-GB) Embedder wherever only ingestion wiring, not
    actual embedding quality, is under test."""

    def __init__(self, device=None, batch_size=None):
        self.device = device
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.random.default_rng(0).random((len(texts), 4), dtype=np.float32)


@pytest.fixture
def fake_embedder(monkeypatch):
    """Patch scripts.build_index.Embedder so ingest_repository never loads a real model."""
    monkeypatch.setattr("scripts.build_index.Embedder", _FakeEmbedder)
