"""Taiyo Yuden (TSE 6976) IR adapter.

English "Summary of Q&A" PDFs live in the IR Library under https://www.yuden.co.jp/
(https://www.yuden.co.jp/eu/ir/ -> IR Library). The Q&A summaries are IR-edited
paraphrases, so doc_kind = 'summary' and downstream analysis treats dropped
content as weak evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

from tde.config import CompanyConfig
from tde.sources.base import EventRef, RawDoc
from tde.sources.ir_base import PoliteFetcher, extract_pdf_text, find_pdf_links

IR_LIBRARY_URLS = [
    "https://www.yuden.co.jp/eu/ir/library/",
    "https://www.yuden.co.jp/ut/ir/library/",
]

QUARTER_RE = re.compile(r"FY\s*(\d{4})[^\d]{0,20}([1-4])Q|([1-4])Q\s*FY\s*(\d{4})", re.IGNORECASE)
QA_HINTS = ("summary of q&a", "q&a summary", "summary of questions")


class TaiyoIRSource:
    name = "ir_taiyo"

    def __init__(self, raw_dir: Path = Path("data/raw/6976")):
        self.fetcher = PoliteFetcher(raw_dir)

    def list_events(self, company: CompanyConfig) -> list[EventRef]:
        events: dict[str, EventRef] = {}
        for library_url in IR_LIBRARY_URLS:
            try:
                html = self.fetcher.get(library_url).decode("utf-8", errors="replace")
            except Exception:
                continue
            for url, text in find_pdf_links(html, library_url):
                lowered = text.lower()
                if not any(h in lowered for h in QA_HINTS):
                    continue
                m = QUARTER_RE.search(text) or QUARTER_RE.search(url)
                if not m:
                    continue
                fy = m.group(1) or m.group(4)
                q = m.group(2) or m.group(3)
                event_id = f"taiyo_fy{fy}_q{q}"
                # March fiscal year end, same cadence as Murata.
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
        # Q&A summaries are paraphrased pairs; coarse Q:/A: turns are acceptable.
        raw_text = re.sub(r"^\s*Q\s*[\.:）\)]", "Questioner: ", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"^\s*A\s*[\.:）\)]", "Answer: ", raw_text, flags=re.MULTILINE)
        return RawDoc(
            event=event,
            doc_kind="summary",
            source_version="ir_pdf",
            raw_text=raw_text,
            payload={"event": event.__dict__, "raw_text": raw_text},
        )
