"""Sentence embedding wrapper (sentence-transformers, batched, lazily loaded)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of sentence embeddings."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install ML extras: uv pip install -e '.[ml]'"
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)
