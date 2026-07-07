"""Normalize raw transcript payloads into sectioned, speaker-tagged sentences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from tde.sources.base import RawDoc

NUMERIC_RE = re.compile(
    r"\d[\d,\.]*|%|\$|\b(?:bn|mn|million|billion|trillion|bps)\b", re.IGNORECASE
)
QA_HEADING_RE = re.compile(
    r"^\s*(?:question[- ]and[- ]answer|questions? and answers?|q\s*&\s*a)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
SPEAKER_TURN_RE = re.compile(r"^([A-Z][\w\.\- ']{1,60}):\s+", re.MULTILINE)
OPERATOR_BOILERPLATE = ("operator", "moderator")


@dataclass
class Sentence:
    section: str  # 'prepared' | 'qa'
    speaker: str | None
    speaker_role: str  # 'exec' | 'analyst' | 'operator' | 'unknown'
    idx: int
    text: str
    word_count: int
    numeric_tokens: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@lru_cache(maxsize=1)
def _segmenter():
    import pysbd

    return pysbd.Segmenter(language="en", clean=False)


def segment(text: str) -> list[str]:
    return [s.strip() for s in _segmenter().segment(text) if s.strip()]


def count_numeric_tokens(text: str) -> int:
    return len(NUMERIC_RE.findall(text))


def _role_from_component_type(component_type: str) -> tuple[str, str]:
    """Map a CIQ component_type to (section, speaker_role)."""
    ct = component_type.lower()
    if "operator" in ct:
        section = "qa" if "question" in ct else "prepared"
        return section, "operator"
    if "presenter" in ct or "presentation" in ct:
        return "prepared", "exec"
    if "question" in ct:
        return "qa", "analyst"
    if "answer" in ct:
        return "qa", "exec"
    return "prepared", "unknown"


def normalize(doc: RawDoc) -> list[Sentence]:
    """Produce ordered sentences from either structured components or raw text."""
    if doc.components:
        turns = [
            (c.get("person_name") or None, *_role_from_component_type(c.get("component_type", "")))
            + (c.get("text", ""),)
            for c in doc.components
        ]
        # -> list of (speaker, section, role, text)
    elif doc.raw_text:
        turns = _turns_from_raw_text(doc.raw_text)
    else:
        return []

    sentences: list[Sentence] = []
    idx = 0
    for speaker, section, role, text in turns:
        for sent in segment(text):
            sentences.append(
                Sentence(
                    section=section,
                    speaker=speaker,
                    speaker_role=role,
                    idx=idx,
                    text=sent,
                    word_count=len(sent.split()),
                    numeric_tokens=count_numeric_tokens(sent),
                )
            )
            idx += 1
    return sentences


def _turns_from_raw_text(raw_text: str) -> list[tuple[str | None, str, str, str]]:
    """Fallback parser: split prepared vs Q&A on a heading, then on 'Name:' turns.

    Heuristics for roles: 'Operator'/'Moderator' -> operator; speakers first seen
    in the prepared section are execs; new speakers appearing only in Q&A are
    analysts (they get introduced to ask questions); unknown otherwise.
    """
    m = QA_HEADING_RE.search(raw_text)
    if m:
        parts = [("prepared", raw_text[: m.start()]), ("qa", raw_text[m.end() :])]
    else:
        parts = [("prepared", raw_text)]

    prepared_speakers: set[str] = set()
    turns: list[tuple[str | None, str, str, str]] = []
    for section, block in parts:
        matches = list(SPEAKER_TURN_RE.finditer(block))
        if not matches:
            if block.strip():
                turns.append((None, section, "unknown", block.strip()))
            continue
        for i, sm in enumerate(matches):
            speaker = sm.group(1).strip()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
            text = block[sm.end() : end].strip()
            if not text:
                continue
            lowered = speaker.lower()
            if any(b in lowered for b in OPERATOR_BOILERPLATE):
                role = "operator"
            elif section == "prepared":
                prepared_speakers.add(speaker)
                role = "exec"
            elif speaker in prepared_speakers:
                role = "exec"
            else:
                role = "analyst"
            turns.append((speaker, section, role, text))
    return turns
