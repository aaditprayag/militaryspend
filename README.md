# Transcript Delta Engine (TDE)

A local CLI pipeline that ingests earnings-call transcripts for a semis / neocloud /
AI-infrastructure coverage universe, computes quarter-over-quarter language-change
metrics, semantically diffs each call against the prior one, and renders an
analyst-grade HTML brief per name plus a cross-sectional dashboard.

Rationale: deliberate changes in corporate language predict fundamentals and returns,
and the market underreacts to them (Cohen/Malloy/Nguyen, *Lazy Prices*, JF 2020).
Prepared remarks and Q&A are analyzed separately; all metrics are z-scored against
each company's **own** trailing history, because cross-company levels are noise and
within-company deltas are signal.

## Status (MVP)

| Phase | What | Status |
|---|---|---|
| 0 | kFinance entitlement smoke test | **Built; blocked on credentials.** `scripts/smoke_kfinance.py` introspects the real API surface and STOPs with a clear message until kFinance credentials are added to `.env`. Live QCOM fetch not yet verified. |
| 1 | Ingestion + backfill (kFinance adapter, normalization, SQLite) | Built and unit/integration tested against fixture transcripts. Live backfill pending Phase 0 credentials. |
| 2 | Metrics engine (LM, finbert, embeddings, all 11 metric families, z-scores) | Built; every formula has a hand-computed pytest. Real model inference requires `pip install -e '.[ml]'`. |
| 3 | Diff engine, Claude synthesis, HTML reports | Built; quote validation and alignment tested offline; synthesis needs `ANTHROPIC_API_KEY`. |
| 4 | Murata / Taiyo Yuden IR PDF adapters | Built (defensive parsing, polite 2s throttle, cache-and-skip); not yet exercised against the live sites. |
| 5 | Full-universe operations (`run-all --new-only`) | CLI built; awaiting live data. |

**28/28 tests pass offline; `ruff check` clean.** The end-to-end pipeline
(ingest → analyze → diff → report → index) runs in tests with fixture transcripts and
injected model backends, so the plumbing is verified even without live data access.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"        # core + tests
uv pip install -e ".[ml]"                    # sentence-transformers + finbert (needed for analyze)
uv pip install -e ".[pdf]"                   # pdfplumber (needed for the Japan IR adapters)
cp .env.example .env                          # then fill in credentials
```

`.env` keys:

- `KFINANCE_REFRESH_TOKEN` **or** `KFINANCE_CLIENT_ID` + `KFINANCE_PRIVATE_KEY` — kFinance auth
- `ANTHROPIC_API_KEY` — Claude synthesis (reports render without it via `--no-llm`, minus the narrative sections)

**Loughran-McDonald dictionary:** sraf.nd.edu serves the Master Dictionary CSV via a
Google Drive link that changes between releases, so no direct URL is hardcoded.
Download it from <https://sraf.nd.edu/loughranmcdonald-master-dictionary/> into
`data/lexicons/`, or paste a direct-download URL into `lm_dictionary_url` in
`config/settings.yaml`.

## First run

```bash
python scripts/smoke_kfinance.py      # Phase 0: verify transcript entitlement (QCOM + TDK)
tde init
tde backfill --ticker QCOM --quarters 8
tde analyze --ticker QCOM
tde diff --ticker QCOM
tde report --ticker QCOM             # add --no-llm to skip Claude synthesis
open data/reports/index.html
```

Or the whole thing at once: `tde run-all --ticker QCOM`.

## Earnings-morning watch

On earnings mornings run:

```bash
tde run-all --new-only
```

For each name it checks for a new event since the last stored transcript; if found it
ingests → analyzes → diffs → reports and rebuilds `data/reports/index.html`, printing a
one-line summary per name. Optional cron snippet (not installed automatically):

```cron
# 7:00 am weekdays
0 7 * * 1-5  cd /path/to/tde && .venv/bin/tde run-all --new-only >> data/watch.log 2>&1
```

## Layout

```
config/            universe.yaml, settings.yaml, taxonomies/ (global + per-name themes)
tde/sources/       kfinance adapter + Murata/Taiyo IR PDF scrapers (SourceAdapter protocol)
tde/parsing/       speaker/section parsing, pysbd sentence segmentation
tde/nlp/           LM lexicon, finbert wrapper, embeddings, all metric formulas
tde/diff/          QoQ sentence alignment (cosine, threshold in settings)
tde/synth/         Claude synthesis (temperature 0, JSON-validated, quotes verified verbatim)
tde/report/        jinja2 templates -> static HTML per transcript + index heat table
tde/pipeline.py    stage orchestration (CLI commands are thin wrappers)
tde/db.py          SQLite schema + data access; embeddings as float32 BLOBs
scripts/           smoke_kfinance.py (Phase 0)
tests/             pytest suite w/ fixture transcripts and hand-computed metric values
data/              (gitignored) raw payloads, lexicons, db.sqlite, reports
```

## Design notes & deviations from spec

- **`tde/pipeline.py`** was added beyond the spec layout so each stage is testable with
  injected backends (fake embedder/sentiment in tests); the CLI stays thin.
- ML dependencies are **optional extras** (`.[ml]`) so ingestion, tests, and report
  rendering work without a multi-GB torch install. `tde analyze` requires them.
- The current `kensho-kfinance` client exposes a single transcript per `key_dev_id`
  (no spot/edited/proofed versioning endpoint), so `source_version` records `latest`;
  revisit if the entitled payload turns out to carry version fields.
- Z-scores use population std over the trailing window (max 8 quarters), null when
  history < 4 quarters (`zscore_min_quarters`) or variance is zero.
- Transcripts are licensed content: raw payloads stay in gitignored `data/`, and
  reports only quote sub-25-word excerpts with speaker attribution, validated
  programmatically as verbatim substrings.

## Validation protocol (pending live data)

After Phase 5 backfill: (1) replay test on Murata/Taiyo FY2025-26 and QCOM's June-2026
Investor Day quarter vs known guidance inflections; (2) negative control on two
uneventful quarters; (3) novelty threshold sweep at 0.70/0.75/0.80 on QCOM
(`alignment_threshold` in settings).
