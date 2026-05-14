import json

from src.pipeline.sqlite_logging import init_prediction_db, log_prediction_row


def test_sqlite_logging_roundtrip(tmp_path):
    db = tmp_path / "t.sqlite"
    init_prediction_db(db)
    row = {"PatientenID": "p1", "klasse": 1, "evidence_snippets": "[]"}
    log_prediction_row(db, row)
    log_prediction_row(db, {"PatientenID": "p2", "klasse": 0})

    import sqlite3

    conn = sqlite3.connect(str(db))
    try:
        cur = conn.execute("SELECT PatientenID, payload FROM prediction_rows ORDER BY id")
        rows = cur.fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert rows[0][0] == "p1"
    assert json.loads(rows[0][1])["klasse"] == 1
