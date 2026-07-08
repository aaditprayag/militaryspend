"""SQLite schema and data access."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
  id INTEGER PRIMARY KEY, ticker TEXT UNIQUE, name TEXT,
  source TEXT, exchange TEXT, source_identifier TEXT
);
CREATE TABLE IF NOT EXISTS transcripts (
  id INTEGER PRIMARY KEY, company_id INT REFERENCES companies(id),
  event_type TEXT DEFAULT 'earnings',
  fiscal_label TEXT,
  call_date TEXT,
  doc_kind TEXT,
  source TEXT, source_event_id TEXT, source_version TEXT,
  raw_path TEXT, word_count INT,
  UNIQUE(company_id, source_event_id)
);
CREATE TABLE IF NOT EXISTS sentences (
  id INTEGER PRIMARY KEY, transcript_id INT REFERENCES transcripts(id),
  section TEXT,
  speaker TEXT, speaker_role TEXT,
  idx INT, text TEXT,
  embedding BLOB,
  lm_counts TEXT,
  finbert_label TEXT, finbert_score REAL,
  numeric_tokens INT, word_count INT
);
CREATE INDEX IF NOT EXISTS idx_sentences_transcript ON sentences(transcript_id);
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY, transcript_id INT, section TEXT,
  name TEXT, value REAL, zscore REAL,
  UNIQUE(transcript_id, section, name)
);
CREATE TABLE IF NOT EXISTS diffs (
  id INTEGER PRIMARY KEY, transcript_id INT, prev_transcript_id INT,
  kind TEXT,
  sentence_id INT,
  best_match_sim REAL
);
CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY, transcript_id INT, html_path TEXT,
  llm_json TEXT, created_at TEXT
);
"""


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def pack_embedding(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def upsert_company(
    conn: sqlite3.Connection,
    ticker: str,
    name: str,
    source: str,
    exchange: str | None = None,
    source_identifier: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO companies (ticker, name, source, exchange, source_identifier)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
          name=excluded.name, source=excluded.source, exchange=excluded.exchange,
          source_identifier=COALESCE(excluded.source_identifier, companies.source_identifier)
        """,
        (ticker, name, source, exchange, source_identifier),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM companies WHERE ticker = ?", (ticker,)).fetchone()
    return int(row["id"])


def get_company(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM companies WHERE ticker = ?", (ticker,)).fetchone()


def upsert_transcript(
    conn: sqlite3.Connection,
    company_id: int,
    source_event_id: str,
    *,
    event_type: str = "earnings",
    fiscal_label: str | None = None,
    call_date: str | None = None,
    doc_kind: str = "verbatim",
    source: str | None = None,
    source_version: str | None = None,
    raw_path: str | None = None,
    word_count: int | None = None,
) -> tuple[int, bool]:
    """Insert or update a transcript row. Returns (transcript_id, created)."""
    existing = conn.execute(
        "SELECT id FROM transcripts WHERE company_id = ? AND source_event_id = ?",
        (company_id, source_event_id),
    ).fetchone()
    conn.execute(
        """
        INSERT INTO transcripts (company_id, event_type, fiscal_label, call_date, doc_kind,
                                 source, source_event_id, source_version, raw_path, word_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(company_id, source_event_id) DO UPDATE SET
          fiscal_label=excluded.fiscal_label, call_date=excluded.call_date,
          doc_kind=excluded.doc_kind, source_version=excluded.source_version,
          raw_path=excluded.raw_path, word_count=excluded.word_count
        """,
        (
            company_id,
            event_type,
            fiscal_label,
            call_date,
            doc_kind,
            source,
            source_event_id,
            source_version,
            raw_path,
            word_count,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM transcripts WHERE company_id = ? AND source_event_id = ?",
        (company_id, source_event_id),
    ).fetchone()
    return int(row["id"]), existing is None


def replace_sentences(
    conn: sqlite3.Connection, transcript_id: int, sentences: list[dict[str, Any]]
) -> None:
    """Replace all sentence rows for a transcript (idempotent re-ingest)."""
    conn.execute("DELETE FROM sentences WHERE transcript_id = ?", (transcript_id,))
    conn.executemany(
        """
        INSERT INTO sentences (transcript_id, section, speaker, speaker_role, idx, text,
                               numeric_tokens, word_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                transcript_id,
                s["section"],
                s.get("speaker"),
                s.get("speaker_role", "unknown"),
                s["idx"],
                s["text"],
                s.get("numeric_tokens", 0),
                s.get("word_count", 0),
            )
            for s in sentences
        ],
    )
    conn.commit()


def transcripts_for_company(conn: sqlite3.Connection, company_id: int) -> list[sqlite3.Row]:
    """Transcripts ordered by call_date ascending."""
    return conn.execute(
        "SELECT * FROM transcripts WHERE company_id = ? ORDER BY call_date ASC, id ASC",
        (company_id,),
    ).fetchall()


def sentences_for_transcript(
    conn: sqlite3.Connection, transcript_id: int, section: str | None = None
) -> list[sqlite3.Row]:
    if section:
        return conn.execute(
            "SELECT * FROM sentences WHERE transcript_id = ? AND section = ? ORDER BY idx",
            (transcript_id, section),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM sentences WHERE transcript_id = ? ORDER BY idx", (transcript_id,)
    ).fetchall()


def update_sentence_enrichment(
    conn: sqlite3.Connection,
    sentence_id: int,
    *,
    embedding: bytes | None = None,
    lm_counts: dict[str, int] | None = None,
    finbert_label: str | None = None,
    finbert_score: float | None = None,
) -> None:
    sets: list[str] = []
    params: list[Any] = []
    if embedding is not None:
        sets.append("embedding = ?")
        params.append(embedding)
    if lm_counts is not None:
        sets.append("lm_counts = ?")
        params.append(json.dumps(lm_counts))
    if finbert_label is not None:
        sets.append("finbert_label = ?, finbert_score = ?")
        params.extend([finbert_label, finbert_score])
    if not sets:
        return
    params.append(sentence_id)
    conn.execute(f"UPDATE sentences SET {', '.join(sets)} WHERE id = ?", params)


def write_metric(
    conn: sqlite3.Connection,
    transcript_id: int,
    section: str,
    name: str,
    value: float | None,
    zscore: float | None,
) -> None:
    conn.execute(
        """
        INSERT INTO metrics (transcript_id, section, name, value, zscore)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(transcript_id, section, name) DO UPDATE SET
          value=excluded.value, zscore=excluded.zscore
        """,
        (transcript_id, section, name, value, zscore),
    )


def metric_history(
    conn: sqlite3.Connection,
    company_id: int,
    section: str,
    name: str,
    before_date: str,
    window: int,
) -> list[float]:
    """Trailing metric values for the same company+section+metric, most recent last."""
    rows = conn.execute(
        """
        SELECT m.value FROM metrics m
        JOIN transcripts t ON t.id = m.transcript_id
        WHERE t.company_id = ? AND m.section = ? AND m.name = ?
          AND t.call_date < ? AND m.value IS NOT NULL
        ORDER BY t.call_date DESC LIMIT ?
        """,
        (company_id, section, name, before_date, window),
    ).fetchall()
    return [float(r["value"]) for r in reversed(rows)]


def metrics_for_transcript(conn: sqlite3.Connection, transcript_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM metrics WHERE transcript_id = ? ORDER BY section, name",
        (transcript_id,),
    ).fetchall()


def replace_diffs(
    conn: sqlite3.Connection,
    transcript_id: int,
    prev_transcript_id: int,
    rows: list[tuple[str, int, float]],
) -> None:
    """rows: (kind, sentence_id, best_match_sim)."""
    conn.execute(
        "DELETE FROM diffs WHERE transcript_id = ? AND prev_transcript_id = ?",
        (transcript_id, prev_transcript_id),
    )
    conn.executemany(
        """
        INSERT INTO diffs (transcript_id, prev_transcript_id, kind, sentence_id, best_match_sim)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(transcript_id, prev_transcript_id, k, sid, sim) for k, sid, sim in rows],
    )
    conn.commit()


def diffs_for_transcript(conn: sqlite3.Connection, transcript_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT d.*, s.text, s.speaker, s.section, s.word_count
        FROM diffs d JOIN sentences s ON s.id = d.sentence_id
        WHERE d.transcript_id = ?
        """,
        (transcript_id,),
    ).fetchall()


def save_report(
    conn: sqlite3.Connection,
    transcript_id: int,
    html_path: str,
    llm_json: str | None,
    created_at: str,
) -> None:
    conn.execute(
        "INSERT INTO reports (transcript_id, html_path, llm_json, created_at) VALUES (?, ?, ?, ?)",
        (transcript_id, html_path, llm_json, created_at),
    )
    conn.commit()
