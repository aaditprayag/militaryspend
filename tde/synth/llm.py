"""Claude synthesis: deterministic JSON-out call plus programmatic validation."""

from __future__ import annotations

import json
import os
from typing import Any

from tde.synth.prompts import SYSTEM, build_user_prompt

EMPTY_SYNTHESIS: dict[str, Any] = {
    "new_topics": [],
    "dropped_topics": [],
    "softened": [],
    "hardened": [],
    "analyst_focus_shift": "",
    "one_paragraph_read": "",
    "flags": [],
}

QUOTE_KEYS = ("new_topics", "dropped_topics", "softened", "hardened")


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the model output as JSON, tolerating fenced code blocks."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def validate_quotes(result: dict[str, Any], source_sentences: list[str]) -> dict[str, Any]:
    """Drop any item whose evidence_quote is not a verbatim substring of a
    provided sentence or is 25+ words. Mutates a copy; returns it."""
    out = dict(result)
    for key in QUOTE_KEYS:
        items = out.get(key) or []
        kept = []
        for item in items:
            quote = str(item.get("evidence_quote", ""))
            if not quote or len(quote.split()) >= 25:
                continue
            if any(quote in s for s in source_sentences):
                kept.append(item)
        out[key] = kept
    for req_key, default in EMPTY_SYNTHESIS.items():
        out.setdefault(req_key, default)
    return out


def synthesize(
    *,
    model: str,
    max_tokens: int,
    company_name: str,
    fiscal_label: str,
    call_date: str,
    doc_kind: str,
    metrics_rows: list[dict[str, Any]],
    new_sentences: list[dict[str, Any]],
    dropped_sentences: list[dict[str, Any]],
    theme_deltas: list[dict[str, Any]],
    guidance_shift: dict[str, Any],
) -> dict[str, Any]:
    """One deterministic Claude call per transcript; retries once on malformed JSON."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
    client = anthropic.Anthropic(api_key=api_key)

    prompt = build_user_prompt(
        company_name,
        fiscal_label,
        call_date,
        doc_kind,
        metrics_rows,
        new_sentences,
        dropped_sentences,
        theme_deltas,
        guidance_shift,
    )
    messages = [{"role": "user", "content": prompt}]
    last_err: Exception | None = None
    for _attempt in range(2):  # retry once on malformed JSON
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=SYSTEM,
            messages=messages,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        try:
            result = _extract_json(text)
            break
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            messages = messages[:1] + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "That was not valid JSON. Respond again with ONLY the JSON object.",
                },
            ]
    else:
        raise RuntimeError(f"LLM returned malformed JSON twice: {last_err}")

    source = [str(s["text"]) for s in new_sentences + dropped_sentences]
    return validate_quotes(result, source)
