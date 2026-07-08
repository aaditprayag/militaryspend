"""Orchestration: ingest -> enrich -> metrics -> diff -> synthesize -> report.

CLI commands in tde.cli are thin wrappers over these functions so each stage is
independently testable with injected model backends.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from tde import db
from tde.config import CompanyConfig, Settings, load_taxonomy
from tde.diff.align import align_section, rank_for_llm
from tde.nlp import metrics as M
from tde.nlp.embed import Embedder
from tde.nlp.lexicon import CATEGORIES, LMDictionary
from tde.parsing.normalize import normalize
from tde.sources.base import RawDoc, SourceAdapter

console = Console()

SECTIONS = ("prepared", "qa", "all")


def get_adapter(company: CompanyConfig) -> SourceAdapter:
    if company.source == "kfinance":
        from tde.sources.kfinance_source import KFinanceSource

        return KFinanceSource()
    if company.source == "ir_murata":
        from tde.sources.ir_murata import MurataIRSource

        return MurataIRSource()
    if company.source == "ir_taiyo":
        from tde.sources.ir_taiyo import TaiyoIRSource

        return TaiyoIRSource()
    raise ValueError(f"Unknown source adapter: {company.source}")


# ----------------------------------------------------------------- ingestion


def save_raw(doc: RawDoc, raw_dir: Path, ticker: str) -> Path:
    out_dir = raw_dir / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc.event.call_date}_{doc.event.source_event_id}.json"
    payload = doc.payload if doc.payload is not None else {"raw_text": doc.raw_text}
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def ingest_doc(
    conn: sqlite3.Connection, company_id: int, company: CompanyConfig, doc: RawDoc, raw_path: Path
) -> int:
    sentences = [s.as_dict() for s in normalize(doc)]
    word_count = sum(s["word_count"] for s in sentences)
    transcript_id, _created = db.upsert_transcript(
        conn,
        company_id,
        doc.event.source_event_id,
        fiscal_label=doc.event.fiscal_label,
        call_date=doc.event.call_date,
        doc_kind=doc.doc_kind,
        source=company.source,
        source_version=doc.source_version,
        raw_path=str(raw_path),
        word_count=word_count,
    )
    db.replace_sentences(conn, transcript_id, sentences)
    return transcript_id


def ingest_company(
    conn: sqlite3.Connection,
    company: CompanyConfig,
    settings: Settings,
    *,
    quarters: int | None = None,
    latest_only: bool = False,
    new_only: bool = False,
    adapter: SourceAdapter | None = None,
) -> list[int]:
    """Fetch and store events. Returns transcript ids ingested (chronological)."""
    adapter = adapter or get_adapter(company)
    source_identifier = None
    if hasattr(adapter, "resolve_identifier"):
        try:
            source_identifier = adapter.resolve_identifier(company)
        except Exception as e:  # non-fatal: identifier is informational
            console.print(f"[yellow]{company.ticker}: identifier resolution failed: {e}[/]")
    company_id = db.upsert_company(
        conn, company.ticker, company.name, company.source, company.exchange, source_identifier
    )

    events = adapter.list_events(company)  # chronological
    if new_only:
        last = conn.execute(
            "SELECT MAX(call_date) AS d FROM transcripts WHERE company_id = ?", (company_id,)
        ).fetchone()["d"]
        if last:
            events = [e for e in events if e.call_date > last]
    if latest_only:
        events = events[-1:]
    elif quarters is not None:
        events = events[-quarters:]

    ingested: list[int] = []
    for event in events:
        existing = conn.execute(
            "SELECT id FROM transcripts WHERE company_id = ? AND source_event_id = ?",
            (company_id, event.source_event_id),
        ).fetchone()
        if existing and not latest_only:
            ingested.append(int(existing["id"]))
            continue
        doc = adapter.fetch(company, event)
        raw_path = save_raw(doc, Path(settings.raw_dir), company.ticker)
        tid = ingest_doc(conn, company_id, company, doc, raw_path)
        ingested.append(tid)
        console.print(f"  ingested {company.ticker} {event.call_date} ({event.fiscal_label})")
    return ingested


# ---------------------------------------------------------------- enrichment


def enrich_transcript(
    conn: sqlite3.Connection,
    transcript_id: int,
    lm: LMDictionary,
    embedder: Embedder | None,
    sentiment: Any | None,
) -> None:
    """Fill lm_counts, embedding, finbert for sentences that lack them."""
    rows = db.sentences_for_transcript(conn, transcript_id)

    for r in rows:
        if r["lm_counts"] is None:
            db.update_sentence_enrichment(conn, r["id"], lm_counts=lm.count(r["text"]))

    need_embed = [r for r in rows if r["embedding"] is None]
    if need_embed and embedder is not None:
        vecs = embedder.encode([r["text"] for r in need_embed])
        for r, vec in zip(need_embed, vecs):
            db.update_sentence_enrichment(conn, r["id"], embedding=db.pack_embedding(vec))

    need_sent = [r for r in rows if r["finbert_label"] is None]
    if need_sent and sentiment is not None:
        results = sentiment.classify([r["text"] for r in need_sent])
        for r, (label, score) in zip(need_sent, results):
            db.update_sentence_enrichment(conn, r["id"], finbert_label=label, finbert_score=score)
    conn.commit()


# ------------------------------------------------------------------- metrics


def _section_rows(rows: list[sqlite3.Row], section: str) -> list[sqlite3.Row]:
    return list(rows) if section == "all" else [r for r in rows if r["section"] == section]


def _embeddings(rows: list[sqlite3.Row]) -> np.ndarray:
    vecs = [db.unpack_embedding(r["embedding"]) for r in rows if r["embedding"] is not None]
    return np.vstack(vecs) if vecs else np.zeros((0, 1))


def _lm_totals(rows: list[sqlite3.Row]) -> dict[str, int]:
    totals = dict.fromkeys(CATEGORIES, 0)
    for r in rows:
        if r["lm_counts"]:
            for cat, n in json.loads(r["lm_counts"]).items():
                totals[cat] = totals.get(cat, 0) + int(n)
    return totals


def compute_section_metrics(
    cur_rows: list[sqlite3.Row],
    prev_rows: list[sqlite3.Row] | None,
    taxonomy: dict[str, list[str]],
    threshold: float,
) -> dict[str, float | None]:
    """All per-section metric values for one transcript section."""
    out: dict[str, float | None] = {}
    total_words = sum(r["word_count"] for r in cur_rows)
    text = " ".join(r["text"] for r in cur_rows)
    text_lower = text.lower()

    cur_e = _embeddings(cur_rows)
    if prev_rows is not None:
        prev_e = _embeddings(prev_rows)
        prev_text = " ".join(r["text"] for r in prev_rows)
        out["doc_cosine_prev"] = M.doc_cosine_prev(cur_e, prev_e)
        out["jaccard_trigram_prev"] = M.jaccard_trigram_prev(text, prev_text)
        out["novelty_pct"] = M.novelty_pct(cur_e, prev_e, threshold)
        out["dropped_pct"] = M.dropped_pct(cur_e, prev_e, threshold)

    lm_totals = _lm_totals(cur_rows)
    for cat in CATEGORIES:
        out[f"lm_{cat}_per_1k"] = M.lm_density_per_1k(lm_totals.get(cat, 0), total_words)
    out["guidance_strength"] = M.guidance_strength(
        lm_totals.get("strong_modal", 0), lm_totals.get("weak_modal", 0)
    )
    out["numeric_specificity"] = M.numeric_specificity(
        sum(r["numeric_tokens"] for r in cur_rows), total_words
    )
    n_pos = sum(1 for r in cur_rows if r["finbert_label"] == "Positive")
    n_neg = sum(1 for r in cur_rows if r["finbert_label"] == "Negative")
    n_lab = sum(1 for r in cur_rows if r["finbert_label"] is not None)
    out["finbert_net"] = M.finbert_net(n_pos, n_neg, n_lab)

    for theme, phrases in taxonomy.items():
        mentions = M.theme_mentions(text_lower, phrases)
        out[f"theme_{theme}_per_10k"] = M.theme_density_per_10k(mentions, total_words)
    return out


def compute_transcript_metrics(
    conn: sqlite3.Connection,
    company_id: int,
    transcript: sqlite3.Row,
    prev_transcript: sqlite3.Row | None,
    taxonomy: dict[str, list[str]],
    settings: Settings,
) -> None:
    cur_all = db.sentences_for_transcript(conn, transcript["id"])
    prev_all = db.sentences_for_transcript(conn, prev_transcript["id"]) if prev_transcript else None

    finbert_by_section: dict[str, float | None] = {}
    for section in SECTIONS:
        cur_rows = _section_rows(cur_all, section)
        prev_rows = _section_rows(prev_all, section) if prev_all is not None else None
        values = compute_section_metrics(
            cur_rows, prev_rows, taxonomy, settings.alignment_threshold
        )
        finbert_by_section[section] = values.get("finbert_net")
        for name, value in values.items():
            z = M.zscore(
                value,
                db.metric_history(
                    conn,
                    company_id,
                    section,
                    name,
                    transcript["call_date"],
                    settings.zscore_window_quarters,
                ),
                settings.zscore_min_quarters,
            )
            db.write_metric(conn, transcript["id"], section, name, value, z)

    # Transcript-level metrics.
    gap = M.qa_gap(finbert_by_section.get("prepared"), finbert_by_section.get("qa"))
    z = M.zscore(
        gap,
        db.metric_history(
            conn,
            company_id,
            "transcript",
            "qa_gap",
            transcript["call_date"],
            settings.zscore_window_quarters,
        ),
        settings.zscore_min_quarters,
    )
    db.write_metric(conn, transcript["id"], "transcript", "qa_gap", gap, z)

    # Analyst question theme mix (share of analyst Q&A sentences hitting each theme).
    analyst_rows = [r for r in cur_all if r["section"] == "qa" and r["speaker_role"] == "analyst"]
    for theme, phrases in taxonomy.items():
        hits = sum(1 for r in analyst_rows if any(p.lower() in r["text"].lower() for p in phrases))
        share = M.analyst_theme_share(hits, len(analyst_rows))
        name = f"analyst_theme_{theme}_share"
        z = M.zscore(
            share,
            db.metric_history(
                conn,
                company_id,
                "qa",
                name,
                transcript["call_date"],
                settings.zscore_window_quarters,
            ),
            settings.zscore_min_quarters,
        )
        db.write_metric(conn, transcript["id"], "qa", name, share, z)
    conn.commit()


def analyze_company(
    conn: sqlite3.Connection,
    company: CompanyConfig,
    settings: Settings,
    lm: LMDictionary,
    embedder: Embedder | None,
    sentiment: Any | None,
    config_dir: Path = Path("config"),
) -> int:
    comp = db.get_company(conn, company.ticker)
    if comp is None:
        raise RuntimeError(f"{company.ticker} not ingested yet -- run `tde backfill` first")
    taxonomy = load_taxonomy(company.ticker, config_dir)
    transcripts = db.transcripts_for_company(conn, comp["id"])
    prev: sqlite3.Row | None = None
    for t in transcripts:
        enrich_transcript(conn, t["id"], lm, embedder, sentiment)
        compute_transcript_metrics(conn, comp["id"], t, prev, taxonomy, settings)
        prev = t
    return len(transcripts)


# ---------------------------------------------------------------------- diff


def diff_company(conn: sqlite3.Connection, company: CompanyConfig, settings: Settings) -> int:
    comp = db.get_company(conn, company.ticker)
    transcripts = db.transcripts_for_company(conn, comp["id"])
    pairs = 0
    for prev_t, cur_t in zip(transcripts, transcripts[1:]):
        rows: list[tuple[str, int, float]] = []
        for section in ("prepared", "qa"):
            cur_rows = db.sentences_for_transcript(conn, cur_t["id"], section)
            prev_rows = db.sentences_for_transcript(conn, prev_t["id"], section)
            cur_with_e = [r for r in cur_rows if r["embedding"] is not None]
            prev_with_e = [r for r in prev_rows if r["embedding"] is not None]
            if not cur_with_e or not prev_with_e:
                continue
            aligned = align_section(
                [r["id"] for r in cur_with_e],
                _embeddings(cur_with_e),
                [(r["word_count"], r["speaker_role"]) for r in cur_with_e],
                [r["id"] for r in prev_with_e],
                _embeddings(prev_with_e),
                [(r["word_count"], r["speaker_role"]) for r in prev_with_e],
                settings.alignment_threshold,
            )
            rows.extend((a.kind, a.sentence_id, a.best_match_sim) for a in aligned)
        db.replace_diffs(conn, cur_t["id"], prev_t["id"], rows)
        pairs += 1
    return pairs


# -------------------------------------------------------------------- report


def _llm_inputs(
    conn: sqlite3.Connection, transcript: sqlite3.Row, prev_transcript: sqlite3.Row, top_k: int
) -> dict[str, Any]:
    diffs = [dict(r) for r in db.diffs_for_transcript(conn, transcript["id"])]
    new = rank_for_llm([d for d in diffs if d["kind"] == "new"], top_k)
    dropped = rank_for_llm([d for d in diffs if d["kind"] == "dropped"], top_k)
    slim = lambda d: {  # noqa: E731
        "text": d["text"],
        "speaker": d["speaker"],
        "section": d["section"],
        "best_match_sim": round(float(d["best_match_sim"]), 3),
    }
    metrics_rows = [
        {
            "section": r["section"],
            "name": r["name"],
            "value": round(r["value"], 4) if r["value"] is not None else None,
            "zscore": round(r["zscore"], 2) if r["zscore"] is not None else None,
        }
        for r in db.metrics_for_transcript(conn, transcript["id"])
        if not r["name"].startswith("theme_")
    ]
    cur_theme = {
        r["name"]: r["value"]
        for r in db.metrics_for_transcript(conn, transcript["id"])
        if r["section"] == "all" and r["name"].startswith("theme_")
    }
    prev_theme = {
        r["name"]: r["value"]
        for r in db.metrics_for_transcript(conn, prev_transcript["id"])
        if r["section"] == "all" and r["name"].startswith("theme_")
    }
    theme_deltas = [
        {
            "theme": name,
            "current": round(cur_theme.get(name) or 0.0, 2),
            "prior": round(prev_theme.get(name) or 0.0, 2),
        }
        for name in sorted(cur_theme)
    ]

    def modal_counts(tid: int) -> dict[str, int]:
        totals = _lm_totals(db.sentences_for_transcript(conn, tid))
        return {
            "strong_modal": totals.get("strong_modal", 0),
            "weak_modal": totals.get("weak_modal", 0),
        }

    return {
        "metrics_rows": metrics_rows,
        "new_sentences": [slim(d) for d in new],
        "dropped_sentences": [slim(d) for d in dropped],
        "theme_deltas": theme_deltas,
        "guidance_shift": {
            "current": modal_counts(transcript["id"]),
            "prior": modal_counts(prev_transcript["id"]),
        },
    }


def report_company(
    conn: sqlite3.Connection,
    company: CompanyConfig,
    settings: Settings,
    *,
    with_llm: bool = True,
) -> dict[int, str]:
    """Render reports for every transcript that has a prior. Returns {transcript_id: path}."""
    import os

    from tde.report.render import render_transcript_report

    comp = db.get_company(conn, company.ticker)
    transcripts = db.transcripts_for_company(conn, comp["id"])
    reports_dir = Path(settings.reports_dir)
    paths: dict[int, str] = {}
    llm_available = with_llm and bool(os.environ.get("ANTHROPIC_API_KEY"))
    if with_llm and not llm_available:
        console.print("[yellow]ANTHROPIC_API_KEY not set -- rendering without LLM synthesis[/]")

    for prev_t, cur_t in zip(transcripts, transcripts[1:]):
        synthesis = None
        if llm_available:
            from tde.synth.llm import synthesize

            inputs = _llm_inputs(conn, cur_t, prev_t, settings.top_k_delta_sentences)
            synthesis = synthesize(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                company_name=comp["name"],
                fiscal_label=cur_t["fiscal_label"] or cur_t["call_date"],
                call_date=cur_t["call_date"],
                doc_kind=cur_t["doc_kind"],
                **inputs,
            )
        path = render_transcript_report(
            conn,
            comp,
            cur_t,
            prev_t["id"],
            synthesis,
            settings.alignment_threshold,
            reports_dir,
        )
        db.save_report(
            conn,
            cur_t["id"],
            str(path),
            json.dumps(synthesis) if synthesis else None,
            datetime.now(UTC).isoformat(),
        )
        paths[cur_t["id"]] = path.name
        console.print(f"  report: {path}")
    return paths
