"""kFinance (Kensho/S&P CapIQ) transcript source.

Verified API surface (kensho-kfinance introspection, see scripts/smoke_kfinance.py):
  Client(refresh_token=...) | Client(client_id=..., private_key=...)
  client.ticker(identifier, exchange_code=None) -> Ticker
  ticker.company -> Company (company_id = CIQ id)
  company.all_earnings -> list[Earnings(name, datetime, key_dev_id)]
  earnings.transcript -> Transcript (sequence of TranscriptComponent:
      person_name, text, component_type)
Component types observed: 'Presentation Operator Message', 'Presenter Speech',
'Question', 'Answer', 'Question and Answer Operator Message'.

The current kfinance client exposes a single transcript per key_dev_id (no
multi-version endpoint), so source_version is recorded as 'latest'.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from tde.config import CompanyConfig
from tde.sources.base import EventRef, RawDoc

if TYPE_CHECKING:
    from kfinance.client.kfinance import Client


def make_client() -> Client:
    """Build an authenticated kFinance client from environment variables."""
    from kfinance.client.kfinance import Client

    refresh_token = os.environ.get("KFINANCE_REFRESH_TOKEN") or None
    client_id = os.environ.get("KFINANCE_CLIENT_ID") or None
    private_key = os.environ.get("KFINANCE_PRIVATE_KEY") or None
    if refresh_token:
        return Client(refresh_token=refresh_token)
    if client_id and private_key:
        return Client(client_id=client_id, private_key=private_key)
    raise RuntimeError(
        "No kFinance credentials: set KFINANCE_REFRESH_TOKEN or "
        "KFINANCE_CLIENT_ID + KFINANCE_PRIVATE_KEY in .env"
    )


class KFinanceSource:
    name = "kfinance"

    def __init__(self, client: Any | None = None):
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client()
        return self._client

    def _ticker(self, company: CompanyConfig) -> Any:
        # TSE names need the exchange-qualified lookup; kfinance uses exchange_code.
        # CIQ's code for Tokyo is 'TSE'; pass through whatever universe.yaml says.
        return self.client.ticker(company.ticker, exchange_code=company.exchange)

    def resolve_identifier(self, company: CompanyConfig) -> str:
        """CIQ company id, persisted to companies.source_identifier."""
        return str(self._ticker(company).company.company_id)

    def list_events(self, company: CompanyConfig) -> list[EventRef]:
        earnings = self._ticker(company).company.all_earnings
        past = [e for e in earnings if e.datetime is not None]
        past.sort(key=lambda e: e.datetime)
        return [
            EventRef(
                source_event_id=str(e.key_dev_id),
                call_date=e.datetime.date().isoformat(),
                name=e.name,
                fiscal_label=_fiscal_label_from_name(e.name),
            )
            for e in past
        ]

    def fetch(self, company: CompanyConfig, event: EventRef) -> RawDoc:
        transcript = self.client.transcript(int(event.source_event_id))
        components = [
            {
                "person_name": c.person_name,
                "text": c.text,
                "component_type": c.component_type,
            }
            for c in transcript
        ]
        return RawDoc(
            event=event,
            doc_kind="verbatim",
            source_version="latest",
            components=components,
            payload={"event": event.__dict__ | {"extra": event.extra}, "components": components},
        )


def _fiscal_label_from_name(name: str) -> str | None:
    """Extract e.g. 'Q3 2025' from CIQ event names like
    'Qualcomm Incorporated, Q3 2025 Earnings Call, Jul 30, 2025'."""
    import re

    m = re.search(r"\b(Q[1-4])\s*(?:FY)?\s*(\d{4})\b", name)
    if m:
        return f"{m.group(1)} FY{m.group(2)}"
    return None
