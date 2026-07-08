"""Shared fixtures: fixture transcripts, mini LM dictionary, fake ML backends."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from tde.nlp.lexicon import LMDictionary
from tde.sources.base import EventRef, RawDoc

FIXTURES = Path(__file__).parent / "fixtures"


def load_rawdoc(name: str) -> RawDoc:
    payload = json.loads((FIXTURES / name).read_text())
    ev = payload["event"]
    return RawDoc(
        event=EventRef(
            source_event_id=ev["source_event_id"],
            call_date=ev["call_date"],
            fiscal_label=ev["fiscal_label"],
        ),
        doc_kind="verbatim",
        source_version="fixture",
        components=payload["components"],
        payload=payload,
    )


@pytest.fixture
def doc_q1() -> RawDoc:
    return load_rawdoc("call_q1.json")


@pytest.fixture
def doc_q2() -> RawDoc:
    return load_rawdoc("call_q2.json")


@pytest.fixture
def lm_mini() -> LMDictionary:
    return LMDictionary.from_csv(FIXTURES / "lm_mini.csv")


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (identical text -> identical
    vector; shared vocabulary -> high cosine). Stands in for sentence-transformers."""

    dim = 64

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z']+", text.lower()):
                out[i, hash(word) % self.dim] += 1.0
        norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out / norms


class FakeSentiment:
    """Keyword sentiment stand-in for finbert-tone."""

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        out = []
        for t in texts:
            lowered = t.lower()
            if any(w in lowered for w in ("strong", "grew", "up ", "wins")):
                out.append(("Positive", 0.9))
            elif any(w in lowered for w in ("decline", "uncertainty", "pushout", "headwind")):
                out.append(("Negative", 0.9))
            else:
                out.append(("Neutral", 0.9))
        return out


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_sentiment() -> FakeSentiment:
    return FakeSentiment()
