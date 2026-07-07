"""Loughran-McDonald Master Dictionary loader and per-sentence category counters."""

from __future__ import annotations

import csv
import re
from pathlib import Path

CATEGORIES: dict[str, str] = {
    # metric key -> CSV column name
    "negative": "Negative",
    "positive": "Positive",
    "uncertainty": "Uncertainty",
    "litigious": "Litigious",
    "strong_modal": "Strong_Modal",
    "weak_modal": "Weak_Modal",
    "constraining": "Constraining",
}

TOKEN_RE = re.compile(r"\b[A-Za-z']+\b")

SRAF_PAGE = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"


class LMDictionary:
    """Category -> set of UPPERCASE words."""

    def __init__(self, words_by_category: dict[str, set[str]]):
        self.words_by_category = words_by_category

    def count(self, text: str) -> dict[str, int]:
        tokens = [t.upper() for t in TOKEN_RE.findall(text)]
        return {
            cat: sum(1 for t in tokens if t in words)
            for cat, words in self.words_by_category.items()
        }

    @classmethod
    def from_csv(cls, path: Path) -> LMDictionary:
        words: dict[str, set[str]] = {cat: set() for cat in CATEGORIES}
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "Word" not in reader.fieldnames:
                raise ValueError(f"{path} does not look like the LM Master Dictionary CSV")
            for row in reader:
                word = row["Word"].strip().upper()
                for cat, col in CATEGORIES.items():
                    # LM convention: nonzero value = year the word entered the category.
                    if row.get(col, "0").strip() not in ("", "0"):
                        words[cat].add(word)
        return cls(words)


def find_or_download(lexicon_dir: Path, url: str = "") -> Path:
    """Locate a cached LM Master Dictionary CSV, downloading it if a URL is configured."""
    lexicon_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(lexicon_dir.glob("*aster*ictionary*.csv")) + sorted(
        lexicon_dir.glob("lm_master*.csv")
    )
    if cached:
        return cached[0]
    if url:
        import requests

        dest = lexicon_dir / "lm_master_dictionary.csv"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    raise FileNotFoundError(
        f"LM Master Dictionary not found in {lexicon_dir}. Download the Master Dictionary "
        f"CSV from {SRAF_PAGE} into that directory, or set lm_dictionary_url in "
        "config/settings.yaml to a direct-download URL."
    )


def load(lexicon_dir: Path, url: str = "") -> LMDictionary:
    return LMDictionary.from_csv(find_or_download(lexicon_dir, url))
