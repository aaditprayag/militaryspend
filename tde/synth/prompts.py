"""Prompt construction for the Claude synthesis call."""

from __future__ import annotations

import json
from typing import Any

SYSTEM = """You are an equity research analyst specializing in semiconductors and \
AI infrastructure. You are given quantitative language-change metrics and the \
sentences that changed between a company's current and prior earnings call. \
Your job is to interpret deliberate changes in corporate language.

Respond ONLY with a JSON object matching this schema, no prose outside the JSON:
{
  "new_topics":      [{"topic": str, "evidence_quote": str, "speaker": str}],
  "dropped_topics":  [{"topic": str, "evidence_quote": str, "speaker": str}],
  "softened":        [{"topic": str, "evidence_quote": str, "speaker": str}],
  "hardened":        [{"topic": str, "evidence_quote": str, "speaker": str}],
  "analyst_focus_shift": str,
  "one_paragraph_read": str,
  "flags": [{"severity": "high"|"med"|"low", "note": str}]
}

Rules:
- Every evidence_quote MUST be a verbatim contiguous substring of one of the
  provided sentences, and under 25 words. Quotes are validated programmatically;
  invalid quotes are discarded.
- Base claims only on the provided data. Do not speculate beyond it.
- "softened"/"hardened" refer to modal/guidance language shifts (e.g. "will" ->
  "could", removal of confident forward statements).
"""

SUMMARY_MODE_NOTE = """NOTE: This document is an IR-edited SUMMARY (not a verbatim \
transcript). Treat DROPPED content as weak evidence (may be an editing artifact) \
and ADDED content as strong evidence (deliberate inclusion by IR)."""


def build_user_prompt(
    company_name: str,
    fiscal_label: str,
    call_date: str,
    doc_kind: str,
    metrics_rows: list[dict[str, Any]],
    new_sentences: list[dict[str, Any]],
    dropped_sentences: list[dict[str, Any]],
    theme_deltas: list[dict[str, Any]],
    guidance_shift: dict[str, Any],
) -> str:
    parts = [
        f"Company: {company_name} | Quarter: {fiscal_label} | Call date: {call_date}",
    ]
    if doc_kind == "summary":
        parts.append(SUMMARY_MODE_NOTE)
    parts += [
        "\n## Metrics (value, z-score vs own trailing history; null z = short history)",
        json.dumps(metrics_rows, indent=1, default=str),
        "\n## Theme density deltas (per 10k words, current minus prior)",
        json.dumps(theme_deltas, indent=1, default=str),
        "\n## Guidance-modal shift (LM strong/weak modal counts, current vs prior)",
        json.dumps(guidance_shift, indent=1, default=str),
        "\n## NEW sentences (in current call, no close match in prior call)",
        json.dumps(new_sentences, indent=1, default=str),
        "\n## DROPPED sentences (in prior call, no close match in current call)",
        json.dumps(dropped_sentences, indent=1, default=str),
        "\nProduce the JSON now.",
    ]
    return "\n".join(parts)
