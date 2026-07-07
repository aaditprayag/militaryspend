"""Typer CLI: init, backfill, ingest, analyze, diff, report, run-all."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from tde import db
from tde.config import CompanyConfig, Settings, load_universe

app = typer.Typer(help="Transcript Delta Engine", no_args_is_help=True)
console = Console()

CONFIG_DIR = Path("config")


def _setup() -> Settings:
    load_dotenv()
    return Settings.load(CONFIG_DIR)


def _conn(settings: Settings):
    conn = db.get_conn(settings.db_path)
    db.init_db(conn)
    return conn


def _companies(ticker: str | None) -> list[CompanyConfig]:
    universe = load_universe(CONFIG_DIR)
    if ticker is None:
        return universe
    matches = [c for c in universe if c.ticker == ticker]
    if not matches:
        raise typer.BadParameter(f"{ticker} not in config/universe.yaml")
    return matches


def _models(settings: Settings):
    """Load LM dictionary + ML models needed by analyze."""
    from tde.nlp.embed import SentenceTransformerEmbedder
    from tde.nlp.lexicon import load as load_lm
    from tde.nlp.sentiment import FinbertTone

    lm = load_lm(Path(settings.lexicon_dir), settings.lm_dictionary_url)
    return lm, SentenceTransformerEmbedder(settings.embedding_model), FinbertTone()


@app.command()
def init() -> None:
    """Create data directories and the SQLite schema."""
    settings = _setup()
    for d in (settings.raw_dir, settings.lexicon_dir, settings.reports_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    _conn(settings).close()
    console.print(f"initialized db at {settings.db_path}")


@app.command()
def backfill(
    ticker: str = typer.Option(..., "--ticker"),
    quarters: int = typer.Option(8, "--quarters"),
) -> None:
    """Backfill N quarters of transcripts for one ticker."""
    settings = _setup()
    conn = _conn(settings)
    from tde.pipeline import ingest_company

    for company in _companies(ticker):
        n = len(ingest_company(conn, company, settings, quarters=quarters))
        console.print(f"{company.ticker}: {n} transcripts stored")


@app.command()
def ingest(
    ticker: str = typer.Option(..., "--ticker"),
    latest: bool = typer.Option(False, "--latest"),
) -> None:
    """Ingest the latest (or all new) events for one ticker."""
    settings = _setup()
    conn = _conn(settings)
    from tde.pipeline import ingest_company

    for company in _companies(ticker):
        n = len(ingest_company(conn, company, settings, latest_only=latest, new_only=not latest))
        console.print(f"{company.ticker}: {n} transcripts ingested")


@app.command()
def analyze(ticker: str = typer.Option(..., "--ticker")) -> None:
    """Enrich sentences (LM, finbert, embeddings) and compute all metrics."""
    settings = _setup()
    conn = _conn(settings)
    from tde.pipeline import analyze_company

    lm, embedder, sentiment = _models(settings)
    for company in _companies(ticker):
        n = analyze_company(conn, company, settings, lm, embedder, sentiment, CONFIG_DIR)
        console.print(f"{company.ticker}: metrics computed for {n} transcripts")


@app.command()
def diff(ticker: str = typer.Option(..., "--ticker")) -> None:
    """Align sentences QoQ and persist new/dropped diffs."""
    settings = _setup()
    conn = _conn(settings)
    from tde.pipeline import diff_company

    for company in _companies(ticker):
        n = diff_company(conn, company, settings)
        console.print(f"{company.ticker}: {n} transcript pairs diffed")


@app.command()
def report(
    ticker: str = typer.Option(..., "--ticker"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip Claude synthesis"),
) -> None:
    """Render HTML reports (and the index) for one ticker."""
    settings = _setup()
    conn = _conn(settings)
    from tde.pipeline import report_company
    from tde.report.render import render_index

    paths: dict[int, str] = {}
    for company in _companies(ticker):
        paths |= report_company(conn, company, settings, with_llm=not no_llm)
    idx = render_index(conn, Path(settings.reports_dir), settings.zscore_window_quarters, paths)
    console.print(f"index: {idx}")


@app.command("run-all")
def run_all(
    ticker: str = typer.Option(None, "--ticker", help="Restrict to one ticker"),
    quarters: int = typer.Option(8, "--quarters"),
    new_only: bool = typer.Option(False, "--new-only", help="Only process new events"),
    no_llm: bool = typer.Option(False, "--no-llm"),
) -> None:
    """Full pipeline: ingest -> analyze -> diff -> report -> index."""
    settings = _setup()
    conn = _conn(settings)
    from tde.pipeline import analyze_company, diff_company, ingest_company, report_company
    from tde.report.render import render_index

    lm, embedder, sentiment = _models(settings)
    all_paths: dict[int, str] = {}
    for company in _companies(ticker):
        try:
            new_ids = ingest_company(
                conn, company, settings, quarters=None if new_only else quarters, new_only=new_only
            )
            if new_only and not new_ids:
                console.print(f"{company.ticker}: no new events")
                continue
            analyze_company(conn, company, settings, lm, embedder, sentiment, CONFIG_DIR)
            diff_company(conn, company, settings)
            all_paths |= report_company(conn, company, settings, with_llm=not no_llm)
            console.print(f"[green]{company.ticker}: pipeline complete[/]")
        except Exception as e:
            console.print(f"[red]{company.ticker}: FAILED -- {e}[/]")
    idx = render_index(conn, Path(settings.reports_dir), settings.zscore_window_quarters, all_paths)
    console.print(f"index: {idx}")


if __name__ == "__main__":
    app()
