"""Render per-transcript HTML briefs and the cross-sectional index."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from tde import db

INDEX_METRICS = [
    "doc_cosine_prev",
    "novelty_pct",
    "lm_uncertainty_per_1k",
    "guidance_strength",
    "qa_gap",
]
# Direction: +1 means a HIGH z is the caution signal (red), -1 means LOW is.
INDEX_DIRECTION = {
    "doc_cosine_prev": -1,  # low cosine = big language change
    "novelty_pct": 1,
    "lm_uncertainty_per_1k": 1,
    "guidance_strength": -1,  # weakening modality = caution
    "qa_gap": 1,  # prepared much rosier than Q&A = caution
}


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("tde.report", "templates"),
        autoescape=select_autoescape(["html"]),
    )


def _z_color(z: float | None, direction: int) -> str:
    if z is None:
        return "transparent"
    signal = z * direction  # positive = caution
    mag = min(abs(signal) / 3.0, 1.0)
    alpha = round(0.15 + 0.55 * mag, 2)
    return f"rgba(214, 39, 40, {alpha})" if signal > 0 else f"rgba(44, 160, 44, {alpha})"


def build_metric_rows(
    conn: sqlite3.Connection, transcript_id: int, prev_transcript_id: int | None
) -> list[dict[str, Any]]:
    rows = db.metrics_for_transcript(conn, transcript_id)
    prev = {}
    if prev_transcript_id is not None:
        prev = {
            (r["section"], r["name"]): r["value"]
            for r in db.metrics_for_transcript(conn, prev_transcript_id)
        }
    out = []
    for r in rows:
        pv = prev.get((r["section"], r["name"]))
        delta = r["value"] - pv if (r["value"] is not None and pv is not None) else None
        out.append(
            {
                "section": r["section"],
                "name": r["name"],
                "value": r["value"],
                "delta": delta,
                "zscore": r["zscore"],
            }
        )
    return out


def build_theme_history(
    conn: sqlite3.Connection, company_id: int, upto_transcript_id: int, quarters: int = 8
) -> list[dict[str, Any]]:
    transcripts = db.transcripts_for_company(conn, company_id)
    ids = [t["id"] for t in transcripts]
    if upto_transcript_id in ids:
        ids = ids[: ids.index(upto_transcript_id) + 1]
    ids = ids[-quarters:]
    themes: dict[str, dict[int, float]] = {}
    for tid in ids:
        for r in db.metrics_for_transcript(conn, tid):
            if r["section"] == "all" and r["name"].startswith("theme_"):
                theme = r["name"].removeprefix("theme_").removesuffix("_per_10k")
                themes.setdefault(theme, {})[tid] = r["value"]
    out = []
    for theme, by_tid in sorted(themes.items()):
        vals = [f"{by_tid[t]:.1f}" if t in by_tid and by_tid[t] is not None else "·" for t in ids]
        latest = vals[-1] if vals else "—"
        out.append({"theme": theme, "vals": vals, "latest": latest})
    return out


def render_transcript_report(
    conn: sqlite3.Connection,
    company: sqlite3.Row,
    transcript: sqlite3.Row,
    prev_transcript_id: int | None,
    synthesis: dict[str, Any] | None,
    threshold: float,
    reports_dir: Path,
) -> Path:
    diffs = db.diffs_for_transcript(conn, transcript["id"])
    diff_counts = {
        "new": sum(1 for d in diffs if d["kind"] == "new"),
        "dropped": sum(1 for d in diffs if d["kind"] == "dropped"),
    }
    html = (
        _env()
        .get_template("transcript.html.j2")
        .render(
            company=dict(company),
            transcript=dict(transcript),
            synthesis=synthesis,
            metric_rows=build_metric_rows(conn, transcript["id"], prev_transcript_id),
            theme_history=build_theme_history(conn, company["id"], transcript["id"]),
            diff_counts=diff_counts,
            threshold=threshold,
        )
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    label = (transcript["fiscal_label"] or transcript["call_date"] or "unknown").replace(" ", "_")
    path = reports_dir / f"{company['ticker']}_{label}.html"
    path.write_text(html)
    return path


def render_index(
    conn: sqlite3.Connection, reports_dir: Path, window: int, report_paths: dict[int, str]
) -> Path:
    """One row per company: latest transcript's key z-scores."""
    rows = []
    for comp in conn.execute("SELECT * FROM companies ORDER BY ticker").fetchall():
        transcripts = db.transcripts_for_company(conn, comp["id"])
        if not transcripts:
            continue
        latest = transcripts[-1]
        zmap = {
            (r["section"], r["name"]): r["zscore"]
            for r in db.metrics_for_transcript(conn, latest["id"])
        }
        cells = []
        for name in INDEX_METRICS:
            section = "transcript" if name == "qa_gap" else "all"
            z = zmap.get((section, name))
            cells.append({"z": z, "color": _z_color(z, INDEX_DIRECTION[name])})
        rows.append(
            {
                "ticker": comp["ticker"],
                "name": comp["name"],
                "fiscal_label": latest["fiscal_label"],
                "call_date": latest["call_date"],
                "href": report_paths.get(latest["id"], "#"),
                "cells": cells,
            }
        )
    html = (
        _env()
        .get_template("index.html.j2")
        .render(
            columns=INDEX_METRICS,
            rows=rows,
            window=window,
            generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        )
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "index.html"
    path.write_text(html)
    return path
