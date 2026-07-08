from tde.nlp.lexicon import CATEGORIES


def test_counts(lm_mini):
    counts = lm_mini.count(
        "We WILL see strong gains, but results may decline amid uncertainty and a lawsuit."
    )
    assert counts["strong_modal"] == 1  # WILL
    assert counts["weak_modal"] == 1  # may
    assert counts["positive"] == 1  # strong (GAINS is not GAIN -- exact word match)
    assert counts["negative"] == 1  # decline
    assert counts["uncertainty"] == 1
    assert counts["litigious"] == 1
    assert counts["constraining"] == 0
    assert set(counts) == set(CATEGORIES)


def test_case_insensitive(lm_mini):
    assert lm_mini.count("will Will WILL")["strong_modal"] == 3
