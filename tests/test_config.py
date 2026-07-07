from pathlib import Path

from tde.config import Settings, load_taxonomy, load_universe

REPO_CONFIG = Path(__file__).parent.parent / "config"


def test_settings_load():
    s = Settings.load(REPO_CONFIG)
    assert s.alignment_threshold == 0.75
    assert s.zscore_window_quarters == 8
    assert s.llm_model.startswith("claude-")


def test_universe_load():
    companies = load_universe(REPO_CONFIG)
    tickers = {c.ticker for c in companies}
    assert {"QCOM", "NVDA", "6981", "6976", "6762"} <= tickers
    murata = next(c for c in companies if c.ticker == "6981")
    assert murata.source == "ir_murata" and murata.exchange == "TSE"


def test_taxonomy_merge():
    base = load_taxonomy("NVDA", REPO_CONFIG)  # no per-name file
    assert "ai_datacenter" in base and "dragonfly" not in base
    qcom = load_taxonomy("QCOM", REPO_CONFIG)
    assert "dragonfly" in qcom and "ai_datacenter" in qcom
    assert "Dragonfly" in qcom["dragonfly"]
