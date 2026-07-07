"""Offline end-to-end: ingest fixtures -> analyze -> diff -> report (no LLM, fake models)."""

from pathlib import Path

from tde import db
from tde.config import CompanyConfig, Settings
from tde.pipeline import (
    analyze_company,
    diff_company,
    ingest_company,
    report_company,
)
from tde.report.render import render_index

REPO_CONFIG = Path(__file__).parent.parent / "config"


class FixtureAdapter:
    name = "fixture"

    def __init__(self, docs):
        self.docs = {d.event.source_event_id: d for d in docs}

    def list_events(self, company):
        return sorted((d.event for d in self.docs.values()), key=lambda e: e.call_date)

    def fetch(self, company, event):
        return self.docs[event.source_event_id]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=str(tmp_path / "db.sqlite"),
        raw_dir=str(tmp_path / "raw"),
        lexicon_dir=str(tmp_path / "lexicons"),
        reports_dir=str(tmp_path / "reports"),
        zscore_min_quarters=1,  # tiny fixture history
    )


def test_end_to_end(tmp_path, doc_q1, doc_q2, lm_mini, fake_embedder, fake_sentiment):
    settings = _settings(tmp_path)
    conn = db.get_conn(settings.db_path)
    db.init_db(conn)
    company = CompanyConfig(ticker="TEST", name="Test Semi", source="kfinance")
    adapter = FixtureAdapter([doc_q1, doc_q2])

    tids = ingest_company(conn, company, settings, quarters=8, adapter=adapter)
    assert len(tids) == 2
    # raw payloads saved untouched
    assert len(list((tmp_path / "raw" / "TEST").glob("*.json"))) == 2
    # idempotent re-ingest
    assert ingest_company(conn, company, settings, quarters=8, adapter=adapter) == tids

    comp = db.get_company(conn, "TEST")
    transcripts = db.transcripts_for_company(conn, comp["id"])
    sentences = db.sentences_for_transcript(conn, transcripts[0]["id"])
    assert sentences and all(s["speaker"] for s in sentences)
    prepared_share = sum(1 for s in sentences if s["section"] == "prepared") / len(sentences)
    assert 0.2 <= prepared_share <= 0.6

    n = analyze_company(
        conn, company, settings, lm_mini, fake_embedder, fake_sentiment, REPO_CONFIG
    )
    assert n == 2
    q2_metrics = {
        (m["section"], m["name"]): m for m in db.metrics_for_transcript(conn, transcripts[1]["id"])
    }
    assert q2_metrics[("all", "doc_cosine_prev")]["value"] is not None
    assert 0 <= q2_metrics[("all", "novelty_pct")]["value"] <= 1
    assert q2_metrics[("all", "jaccard_trigram_prev")]["value"] is not None
    assert q2_metrics[("transcript", "qa_gap")]["value"] is not None
    assert ("all", "theme_ai_datacenter_per_10k") in q2_metrics
    assert q2_metrics[("all", "theme_china_geo_per_10k")]["value"] > 0  # Q2 mentions China/tariff
    # With a single-point history the trailing std is zero, so z is null by design;
    # z-score math itself is covered in test_metrics.py::test_zscore.
    assert q2_metrics[("all", "lm_uncertainty_per_1k")]["zscore"] is None

    pairs = diff_company(conn, company, settings)
    assert pairs == 1
    diffs = db.diffs_for_transcript(conn, transcripts[1]["id"])
    kinds = {d["kind"] for d in diffs}
    assert "new" in kinds  # Falcon accelerator / export-control content is new
    new_texts = " ".join(d["text"] for d in diffs if d["kind"] == "new")
    assert "Falcon" in new_texts

    paths = report_company(conn, company, settings, with_llm=False)
    assert len(paths) == 1
    report_path = tmp_path / "reports"
    html_files = list(report_path.glob("TEST_*.html"))
    assert len(html_files) == 1
    html = html_files[0].read_text()
    assert "Test Semi" in html and "doc_cosine_prev" in html

    idx = render_index(conn, report_path, settings.zscore_window_quarters, paths)
    assert idx.exists() and "TEST" in idx.read_text()
