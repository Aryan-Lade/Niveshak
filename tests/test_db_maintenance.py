import sqlite3
from pathlib import Path
import pytest

from stock_engine.ops.db_maintenance import (
    ensure_database_indexes,
    optimize_sqlite_db,
    run_full_maintenance,
)


def test_optimize_sqlite_db_enables_wal(tmp_path):
    db_file = tmp_path / "test.db"
    res = optimize_sqlite_db(db_file)
    assert res["journal_mode"] == "WAL"

    # Verify directly via PRAGMA
    with sqlite3.connect(db_file) as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(mode).upper() == "WAL"


def test_ensure_database_indexes_creates_index(tmp_path):
    db_file = tmp_path / "trading_test.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, date TEXT, ticker TEXT, score REAL);")
        conn.commit()

    created = ensure_database_indexes(db_file)
    assert "idx_signals_date_ticker" in created

    # Verify index exists
    with sqlite3.connect(db_file) as conn:
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index';").fetchall()
        idx_names = [r[0] for r in indexes]
        assert "idx_signals_date_ticker" in idx_names


def test_run_full_maintenance_batch(tmp_path):
    db1 = tmp_path / "db1.db"
    db2 = tmp_path / "db2.db"

    res = run_full_maintenance([db1, db2])
    assert str(db1) in res
    assert str(db2) in res
    assert res[str(db1)]["success"] is True
    assert res[str(db2)]["success"] is True
