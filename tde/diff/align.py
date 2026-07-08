"""QoQ sentence alignment: flag new and dropped sentences per section."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tde.nlp.metrics import best_match_sims

MIN_WORDS = 6


@dataclass
class AlignedSentence:
    sentence_id: int
    kind: str  # 'new' | 'dropped'
    best_match_sim: float


def _eligible(word_count: int, speaker_role: str) -> bool:
    return word_count >= MIN_WORDS and speaker_role != "operator"


def align_section(
    cur_ids: list[int],
    cur_embeddings: np.ndarray,
    cur_meta: list[tuple[int, str]],  # (word_count, speaker_role) per current sentence
    prev_ids: list[int],
    prev_embeddings: np.ndarray,
    prev_meta: list[tuple[int, str]],
    threshold: float,
) -> list[AlignedSentence]:
    """Align one section of Q_t against Q_(t-1).

    Ineligible sentences (short or operator boilerplate) are excluded from both
    sides: they neither get flagged nor serve as match targets.
    """
    cur_keep = [i for i, m in enumerate(cur_meta) if _eligible(*m)]
    prev_keep = [i for i, m in enumerate(prev_meta) if _eligible(*m)]
    out: list[AlignedSentence] = []

    cur_e = cur_embeddings[cur_keep] if cur_keep else np.zeros((0, 1))
    prev_e = prev_embeddings[prev_keep] if prev_keep else np.zeros((0, 1))

    if cur_keep:
        sims = best_match_sims(cur_e, prev_e)
        for pos, sim in zip(cur_keep, sims):
            if sim < threshold:
                out.append(AlignedSentence(cur_ids[pos], "new", float(sim)))
    if prev_keep:
        sims = best_match_sims(prev_e, cur_e)
        for pos, sim in zip(prev_keep, sims):
            if sim < threshold:
                out.append(AlignedSentence(prev_ids[pos], "dropped", float(sim)))
    return out


def rank_for_llm(rows: list[dict], top_k: int) -> list[dict]:
    """Rank diff rows by (1 - best_match_sim) * sentence length, descending."""
    return sorted(
        rows,
        key=lambda r: (1.0 - float(r["best_match_sim"])) * int(r["word_count"]),
        reverse=True,
    )[:top_k]
