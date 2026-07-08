"""Murata (TSE 6981) IR adapter.

Earnings-call conference materials (full English transcripts, presentation +
Q&A, as PDFs) are published under:
https://corporate.murata.com/en-us/ir/library/meetings

Turn format inside the PDFs: `Name:` speaker turns with a moderator.
doc_kind = 'verbatim'.
"""

from __future__ import annotations

import re
from pathlib import Path

from tde.config import CompanyConfig
from tde.sources.base import EventRef, RawDoc
from tde.sources.ir_base import PoliteFetcher, extract_pdf_text, find_pdf_links

MEETINGS_URL = "https://corporate.murata.com/en-us/ir/library/meetings"

# e.g. "FY2025 3Q Results Meeting Transcript" / "Financial Results ... 3rd Quarter FY2025"
QUARTER_RE = re.compile(r"FY\s*(\d{4})[^\d]{0,20}([1-4])(?:Q|st|nd|rd|th)", re.IGNORECASE)
TRANSCRIPT_HINTS = ("transcript", "q&a", "results meeting", "earnings")


class MurataIRSource:
    name = "ir_murata"

    def __init__(self, raw_dir: Path = Path("data/raw/6981")):
        self.fetcher = PoliteFetcher(raw_dir)

    def list_events(self, company: CompanyConfig) -> list[EventRef]:
        html = self.fetcher.get(MEETINGS_URL).decode("utf-8", errors="replace")
        events: dict[str, EventRef] = {}
        for url, text in find_pdf_links(html, MEETINGS_URL):
            lowered = text.lower()
            if not any(h in lowered for h in TRANSCRIPT_HINTS):
                continue
            m = QUARTER_RE.search(text) or QUARTER_RE.search(url)
            if not m:
                continue
            fy, q = m.group(1), m.group(2)
            event_id = f"murata_fy{fy}_q{q}"
            # Murata's FY ends in March: Q1 call ~Jul, Q2 ~Oct, Q3 ~Jan(+1y), Q4 ~Apr(+1y).
            month = {1: "07", 2: "10", 3: "01", 4: "04"}[int(q)]
            year = int(fy) + (1 if int(q) >= 3 else 0)
            ref = events.setdefault(
                event_id,
                EventRef(
                    source_event_id=event_id,
                    call_date=f"{year}-{month}-28",  # approximate; ordering key only
                    name=text,
                    fiscal_label=f"Q{q} FY{fy}",
                ),
            )
            ref.extra.setdefault("pdf_urls", []).append(url)
        return sorted(events.values(), key=lambda e: e.call_date)

    def fetch(self, company: CompanyConfig, event: EventRef) -> RawDoc:
        texts = []
        for url in event.extra.get("pdf_urls", []):
            fname = f"{event.source_event_id}_{url.rsplit('/', 1)[-1]}"
            pdf_path = self.fetcher.download(url, fname)
            texts.append(extract_pdf_text(pdf_path))
        raw_text = "\n\n".join(texts)
        return RawDoc(
            event=event,
            doc_kind="verbatim",
            source_version="ir_pdf",
            raw_text=raw_text,
            payload={"event": event.__dict__, "raw_text": raw_text},
        )
