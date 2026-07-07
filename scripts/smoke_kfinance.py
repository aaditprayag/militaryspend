"""Phase 0 entitlement smoke test for kFinance transcript access.

Run: python scripts/smoke_kfinance.py
Requires KFINANCE_REFRESH_TOKEN or KFINANCE_CLIENT_ID + KFINANCE_PRIVATE_KEY in .env.
"""

from __future__ import annotations

import inspect
import sys
from datetime import UTC

from dotenv import load_dotenv


def introspect_client_surface() -> None:
    """Print the transcript-related API surface of the installed kfinance package."""
    from kfinance.client import kfinance as kf

    print("== kfinance transcript-related surface ==")
    for cls_name in ("Client", "Company", "Earnings", "Transcript"):
        cls = getattr(kf, cls_name, None)
        if cls is None:
            continue
        members = [
            n
            for n, _ in inspect.getmembers(cls)
            if not n.startswith("_") and ("earning" in n.lower() or "transcript" in n.lower())
        ]
        print(f"  {cls_name}: {members}")
    from kfinance.domains.earnings import earning_models

    print(f"  TranscriptComponent fields: {list(earning_models.TranscriptComponent.model_fields)}")
    print()


def fetch_and_report(client, ticker: str, exchange_code: str | None, label: str) -> bool:
    print(f"== {label} ==")
    try:
        company = client.ticker(ticker, exchange_code=exchange_code).company
        earnings = company.all_earnings
        from datetime import datetime

        past = [e for e in earnings if e.datetime < datetime.now(UTC)]
        if not past:
            print("  no past earnings events found")
            return False
        latest = max(past, key=lambda e: e.datetime)
        print(f"  event: {latest.name}")
        print(f"  date:  {latest.datetime.isoformat()}  key_dev_id: {latest.key_dev_id}")
        transcript = latest.transcript
        raw = transcript.raw
        components = list(transcript)
        types = sorted({c.component_type for c in components})
        print(f"  characters: {len(raw)}")
        print(f"  components: {len(components)}; component_types: {types}")
        has_structure = len(types) > 1 or any("question" in t.lower() for t in types)
        print(f"  speaker/section structure present: {has_structure}")
        print(f"  first 500 chars:\n---\n{raw[:500]}\n---")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return False


def main() -> int:
    load_dotenv()
    introspect_client_surface()

    import os

    if not (
        os.environ.get("KFINANCE_REFRESH_TOKEN")
        or (os.environ.get("KFINANCE_CLIENT_ID") and os.environ.get("KFINANCE_PRIVATE_KEY"))
    ):
        print(
            "STOP: no kFinance credentials configured.\n"
            "Set KFINANCE_REFRESH_TOKEN (or KFINANCE_CLIENT_ID + KFINANCE_PRIVATE_KEY) in .env\n"
            "and re-run. Live entitlement cannot be verified without credentials."
        )
        return 2

    from tde.sources.kfinance_source import make_client

    client = make_client()

    qcom_ok = fetch_and_report(client, "QCOM", None, "QCOM latest earnings transcript")
    tdk_ok = fetch_and_report(client, "6762", "TSE", "TDK (6762.TSE) latest earnings transcript")

    print("== summary ==")
    print(f"  QCOM transcript access: {'OK' if qcom_ok else 'FAILED'}")
    print(f"  TDK international coverage: {'OK' if tdk_ok else 'FAILED (falls back to Phase 4)'}")
    if not qcom_ok:
        print(
            "STOP: QCOM transcript retrieval failed. Per spec, do not proceed past Phase 0.\n"
            "Check entitlement (TranscriptsPermission) with Kensho support."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
