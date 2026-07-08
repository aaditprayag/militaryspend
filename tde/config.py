"""Configuration loading: settings, universe, and taxonomy merge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path("config")


@dataclass
class Settings:
    alignment_threshold: float = 0.75
    zscore_window_quarters: int = 8
    zscore_min_quarters: int = 4
    top_k_delta_sentences: int = 25
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 4000
    embedding_model: str = "all-mpnet-base-v2"
    lm_dictionary_url: str = ""
    db_path: str = "data/db.sqlite"
    raw_dir: str = "data/raw"
    lexicon_dir: str = "data/lexicons"
    reports_dir: str = "data/reports"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_dir: Path = DEFAULT_CONFIG_DIR) -> Settings:
        path = config_dir / "settings.yaml"
        raw: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(**kwargs, extra=extra)


@dataclass
class CompanyConfig:
    ticker: str
    name: str
    source: str
    exchange: str | None = None


def load_universe(config_dir: Path = DEFAULT_CONFIG_DIR) -> list[CompanyConfig]:
    raw = yaml.safe_load((config_dir / "universe.yaml").read_text())
    out: list[CompanyConfig] = []
    for row in raw["companies"]:
        out.append(
            CompanyConfig(
                ticker=str(row["ticker"]),
                name=str(row["name"]),
                source=str(row["source"]),
                exchange=str(row["exchange"]) if row.get("exchange") else None,
            )
        )
    return out


def load_taxonomy(ticker: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, list[str]]:
    """Merge global taxonomy with the per-name file (per-name adds/overrides themes)."""
    tax_dir = config_dir / "taxonomies"
    merged: dict[str, list[str]] = {}
    global_path = tax_dir / "global.yaml"
    if global_path.exists():
        merged.update(yaml.safe_load(global_path.read_text()) or {})
    per_name = tax_dir / f"{ticker}.yaml"
    if per_name.exists():
        merged.update(yaml.safe_load(per_name.read_text()) or {})
    return {theme: [str(p) for p in phrases] for theme, phrases in merged.items()}
