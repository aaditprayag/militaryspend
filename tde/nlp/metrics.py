"""All metric formulas.

Every function here is pure (numpy in, floats out) so it can be unit-tested
against hand-computed fixtures. Z-scores use population std (ddof=0) over the
trailing window and are None when history is short or degenerate.
"""

from __future__ import annotations

import re

import numpy as np

WORD_RE = re.compile(r"[a-z']+")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def doc_cosine_prev(cur: np.ndarray, prev: np.ndarray) -> float | None:
    """Cosine between mean sentence embeddings of current and prior call."""
    if cur.size == 0 or prev.size == 0:
        return None
    return cosine(cur.mean(axis=0), prev.mean(axis=0))


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    words = WORD_RE.findall(text.lower())
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def jaccard_trigram_prev(cur_text: str, prev_text: str) -> float | None:
    """Jaccard similarity of word-trigram sets (Lazy Prices-style lexical measure)."""
    a, b = _trigrams(cur_text), _trigrams(prev_text)
    if not a and not b:
        return None
    union = a | b
    return len(a & b) / len(union) if union else None


def best_match_sims(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """For each row of `cur`, max cosine similarity against rows of `prev`."""
    if cur.size == 0:
        return np.zeros(0)
    if prev.size == 0:
        return np.zeros(cur.shape[0])
    cn = cur / np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
    pn = prev / np.maximum(np.linalg.norm(prev, axis=1, keepdims=True), 1e-12)
    return (cn @ pn.T).max(axis=1)


def novelty_pct(cur: np.ndarray, prev: np.ndarray, threshold: float) -> float | None:
    """Share of current sentences whose best match in the prior call < threshold."""
    if cur.size == 0 or prev.size == 0:
        return None
    sims = best_match_sims(cur, prev)
    return float((sims < threshold).mean())


def dropped_pct(cur: np.ndarray, prev: np.ndarray, threshold: float) -> float | None:
    """Symmetric: share of prior sentences with no good match in the current call."""
    return novelty_pct(prev, cur, threshold)


def lm_density_per_1k(category_count: int, total_words: int) -> float | None:
    if total_words <= 0:
        return None
    return 1000.0 * category_count / total_words


def guidance_strength(strong_modal: int, weak_modal: int) -> float | None:
    denom = strong_modal + weak_modal
    if denom == 0:
        return None
    return strong_modal / denom


def numeric_specificity(numeric_tokens: int, total_words: int) -> float | None:
    if total_words <= 0:
        return None
    return 100.0 * numeric_tokens / total_words


def finbert_net(n_positive: int, n_negative: int, n_total: int) -> float | None:
    if n_total <= 0:
        return None
    return (n_positive - n_negative) / n_total


def qa_gap(finbert_net_prepared: float | None, finbert_net_qa: float | None) -> float | None:
    if finbert_net_prepared is None or finbert_net_qa is None:
        return None
    return finbert_net_prepared - finbert_net_qa


def theme_mentions(text_lower: str, phrases: list[str]) -> int:
    """Total occurrences of any phrase (case-insensitive substring) in the text."""
    return sum(text_lower.count(p.lower()) for p in phrases)


def theme_density_per_10k(mentions: int, total_words: int) -> float | None:
    if total_words <= 0:
        return None
    return 10000.0 * mentions / total_words


def analyst_theme_share(hit_count: int, analyst_sentence_count: int) -> float | None:
    """Share of analyst Q&A sentences that touch a theme."""
    if analyst_sentence_count <= 0:
        return None
    return hit_count / analyst_sentence_count


def zscore(value: float | None, history: list[float], min_quarters: int) -> float | None:
    """Z of `value` vs trailing same-company history (excludes current value)."""
    if value is None or len(history) < min_quarters:
        return None
    arr = np.asarray(history, dtype=np.float64)
    std = float(arr.std())  # population std, ddof=0
    if std == 0.0:
        return None
    return float((value - float(arr.mean())) / std)
