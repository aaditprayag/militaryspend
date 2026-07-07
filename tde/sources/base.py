"""Source adapter protocol and shared data shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from tde.config import CompanyConfig


@dataclass
class EventRef:
    """A reference to one earnings event at a source."""

    source_event_id: str
    call_date: str  # ISO date
    name: str = ""
    fiscal_label: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawDoc:
    """An untouched source payload plus minimal envelope metadata.

    Exactly one of `components` (structured speaker turns) or `raw_text` should
    be populated. `payload` is what gets written verbatim to data/raw/.
    """

    event: EventRef
    doc_kind: str = "verbatim"  # 'verbatim' | 'summary'
    source_version: str | None = None
    components: list[dict[str, str]] | None = None  # {person_name, text, component_type}
    raw_text: str | None = None
    payload: Any = None


class SourceAdapter(Protocol):
    name: str

    def list_events(self, company: CompanyConfig) -> list[EventRef]:
        """All available earnings events, most recent last."""
        ...

    def fetch(self, company: CompanyConfig, event: EventRef) -> RawDoc:
        """Fetch one event's transcript payload."""
        ...
