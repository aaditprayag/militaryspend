"""Shared plumbing for polite IR-site scrapers (Phase 4)."""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": "tde-research/0.1 (personal research; contact: see repo)",
    "Accept-Language": "en",
}
REQUEST_DELAY_S = 2.0


class PoliteFetcher:
    """GET with polite headers, a fixed delay between requests, and file caching."""

    def __init__(self, cache_dir: Path, delay_s: float = REQUEST_DELAY_S):
        self.cache_dir = cache_dir
        self.delay_s = delay_s
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = self.delay_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, url: str) -> bytes:
        self._throttle()
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.content

    def download(self, url: str, filename: str) -> Path:
        """Download to cache_dir, skipping if the file already exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dest = self.cache_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.write_bytes(self.get(url))
        return dest


def extract_pdf_text(path: Path) -> str:
    """pdfplumber extraction; logs and skips pages that fail rather than crashing."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:  # layout quirks vary by year; keep going
                print(f"warn: {path.name} page {i + 1} failed extraction: {e}")
    return "\n".join(pages)


def find_pdf_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """(absolute_url, link_text) for every PDF link in the page."""
    from urllib.parse import urljoin

    out = []
    for m in re.finditer(
        r'<a[^>]+href="([^"]+\.pdf[^"]*)"[^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL
    ):
        text = re.sub(r"<[^>]+>", " ", m.group(2))
        text = re.sub(r"\s+", " ", text).strip()
        out.append((urljoin(base_url, m.group(1)), text))
    return out
