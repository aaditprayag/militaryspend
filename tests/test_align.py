import numpy as np

from tde.diff.align import align_section, rank_for_llm


def _meta(n, wc=10, role="exec"):
    return [(wc, role)] * n


def test_align_flags_new_and_dropped():
    cur = np.array([[1.0, 0.0], [0.0, 1.0]])
    prev = np.array([[1.0, 0.0], [0.7071, 0.7071]])
    out = align_section([11, 12], cur, _meta(2), [21, 22], prev, _meta(2), threshold=0.75)
    by = {(a.kind, a.sentence_id): a for a in out}
    # cur[0] matches prev[0] exactly -> not new. cur[1] best match is prev[1] at cos 0.7071 -> new.
    assert ("new", 12) in by and ("new", 11) not in by
    # prev[1] best vs cur = 0.7071 -> dropped; prev[0] matched -> not dropped.
    assert ("dropped", 22) in by and ("dropped", 21) not in by


def test_align_ignores_short_and_operator():
    cur = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    meta = [(3, "exec"), (10, "operator"), (10, "exec")]
    prev = np.array([[0.0, 1.0]])
    out = align_section([1, 2, 3], cur, meta, [9], prev, [(10, "exec")], threshold=0.75)
    flagged = {a.sentence_id for a in out if a.kind == "new"}
    assert 1 not in flagged  # too short
    assert 2 not in flagged  # operator
    assert 3 in flagged


def test_rank_for_llm():
    rows = [
        {"id": 1, "best_match_sim": 0.5, "word_count": 10},  # score 5.0
        {"id": 2, "best_match_sim": 0.1, "word_count": 4},  # score 3.6
        {"id": 3, "best_match_sim": 0.7, "word_count": 30},  # score 9.0
    ]
    ranked = rank_for_llm(rows, top_k=2)
    assert [r["id"] for r in ranked] == [3, 1]
