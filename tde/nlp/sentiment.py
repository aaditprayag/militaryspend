"""FinBERT-tone sentence sentiment wrapper (batched, lazily loaded)."""

from __future__ import annotations

from typing import Protocol

MODEL_NAME = "yiyanghkust/finbert-tone"
LABELS = ("Positive", "Negative", "Neutral")


class SentimentModel(Protocol):
    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        """Return (label, score) per text; labels in {Positive, Negative, Neutral}."""
        ...


class FinbertTone:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            try:
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                    pipeline,
                )
            except ImportError as e:
                raise RuntimeError(
                    "transformers/torch are not installed. "
                    "Install ML extras: uv pip install -e '.[ml]'"
                ) from e
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self._pipe = pipeline(
                "text-classification", model=model, tokenizer=tok, truncation=True, max_length=512
            )
        return self._pipe

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for res in self.pipe(batch, batch_size=self.batch_size):
                out.append((res["label"], float(res["score"])))
        return out
