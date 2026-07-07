import numpy as np

from tde import db


def _conn():
    conn = db.get_conn(":memory:")
    db.init_db(conn)
    return conn


def test_company_and_transcript_upsert_idempotent():
    conn = _conn()
    cid1 = db.upsert_company(conn, "QCOM", "Qualcomm", "kfinance")
    cid2 = db.upsert_company(conn, "QCOM", "Qualcomm Inc", "kfinance", source_identifier="123")
    assert cid1 == cid2
    row = db.get_company(conn, "QCOM")
    assert row["name"] == "Qualcomm Inc" and row["source_identifier"] == "123"

    tid1, created1 = db.upsert_transcript(conn, cid1, "evt1", call_date="2025-01-01")
    tid2, created2 = db.upsert_transcript(conn, cid1, "evt1", call_date="2025-01-02")
    assert tid1 == tid2 and created1 and not created2
    assert len(db.transcripts_for_company(conn, cid1)) == 1


def test_embedding_roundtrip():
    vec = np.array([0.25, -1.5, 3.0], dtype=np.float32)
    assert np.array_equal(db.unpack_embedding(db.pack_embedding(vec)), vec)


def test_metric_upsert_and_history():
    conn = _conn()
    cid = db.upsert_company(conn, "X", "X", "kfinance")
    tids = []
    for i, date in enumerate(["2024-01-01", "2024-04-01", "2024-07-01"]):
        tid, _ = db.upsert_transcript(conn, cid, f"e{i}", call_date=date)
        db.write_metric(conn, tid, "all", "novelty_pct", 0.1 * (i + 1), None)
        tids.append(tid)
    # upsert overwrites, no duplicate rows
    db.write_metric(conn, tids[0], "all", "novelty_pct", 0.99, 1.0)
    rows = db.metrics_for_transcript(conn, tids[0])
    assert len(rows) == 1 and rows[0]["value"] == 0.99

    hist = db.metric_history(conn, cid, "all", "novelty_pct", "2024-07-01", window=8)
    assert hist == [0.99, 0.2]  # chronological, excludes the 2024-07-01 row
