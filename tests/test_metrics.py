"""Hand-computed expected values for every metric formula (spec metrics 1-11)."""

import math

import numpy as np
import pytest

from tde.nlp import metrics as M


def test_doc_cosine_prev():
    cur = np.array([[1.0, 0.0], [0.0, 1.0]])  # mean = [0.5, 0.5]
    prev = np.array([[1.0, 0.0]])  # mean = [1, 0]
    # cos = 0.5 / (sqrt(0.5) * 1) = 1/sqrt(2)
    assert M.doc_cosine_prev(cur, prev) == pytest.approx(1 / math.sqrt(2))
    assert M.doc_cosine_prev(np.zeros((0, 2)), prev) is None


def test_jaccard_trigram_prev():
    cur = "the quick brown fox jumps"  # trigrams: {tqb, qbf, bfj}
    prev = "the quick brown cat jumps"  # trigrams: {tqb, qbc, bcj}
    # intersection = 1 (the quick brown), union = 5
    assert M.jaccard_trigram_prev(cur, prev) == pytest.approx(1 / 5)
    assert M.jaccard_trigram_prev("one two", "one two") is None  # too short for trigrams
    assert M.jaccard_trigram_prev("a b c", "a b c") == pytest.approx(1.0)


def test_novelty_and_dropped_pct():
    cur = np.array([[1.0, 0.0], [0.0, 1.0]])
    prev = np.array([[1.0, 0.0]])
    # cur sims vs prev: [1.0, 0.0] -> one below 0.75 -> 50% novel
    assert M.novelty_pct(cur, prev, 0.75) == pytest.approx(0.5)
    # prev sims vs cur: [1.0] -> nothing dropped
    assert M.dropped_pct(cur, prev, 0.75) == pytest.approx(0.0)
    assert M.novelty_pct(np.zeros((0, 2)), prev, 0.75) is None


def test_lm_density_per_1k():
    assert M.lm_density_per_1k(3, 1500) == pytest.approx(2.0)
    assert M.lm_density_per_1k(3, 0) is None


def test_guidance_strength():
    assert M.guidance_strength(3, 1) == pytest.approx(0.75)
    assert M.guidance_strength(0, 0) is None  # NaN-guarded


def test_numeric_specificity():
    assert M.numeric_specificity(5, 250) == pytest.approx(2.0)
    assert M.numeric_specificity(5, 0) is None


def test_finbert_net():
    assert M.finbert_net(4, 1, 10) == pytest.approx(0.3)
    assert M.finbert_net(0, 0, 0) is None


def test_qa_gap():
    assert M.qa_gap(0.3, 0.1) == pytest.approx(0.2)
    assert M.qa_gap(None, 0.1) is None


def test_theme_density():
    text = "ai server demand and ai server backlog"
    assert M.theme_mentions(text, ["AI server", "backlog"]) == 3
    assert M.theme_density_per_10k(3, 5000) == pytest.approx(6.0)


def test_analyst_theme_share():
    assert M.analyst_theme_share(2, 8) == pytest.approx(0.25)
    assert M.analyst_theme_share(0, 0) is None


def test_zscore():
    # history [2,4,6,8]: mean 5, population std = sqrt(5)
    assert M.zscore(10.0, [2, 4, 6, 8], min_quarters=4) == pytest.approx(5 / math.sqrt(5))
    assert M.zscore(10.0, [2, 4, 6], min_quarters=4) is None  # short history
    assert M.zscore(10.0, [5, 5, 5, 5], min_quarters=4) is None  # zero variance
    assert M.zscore(None, [2, 4, 6, 8], min_quarters=4) is None
