import sqlite3
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.usefixtures("stub_databento_module")
def test_init_schema_and_tables(tmp_db_path: Path):
    from src.data.fetch_data_sqlite import get_conn, init_schema

    conn = get_conn(tmp_db_path)
    try:
        init_schema(conn)
        # spot-check a few tables
        for table in ("opra_1h", "equity_1h", "processed_merged_1h", "atm_slices_1h"):
            res = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            assert res is not None, f"Missing table {table}"
    finally:
        conn.close()


@pytest.mark.usefixtures("stub_databento_module")
def test__iso_utc_and__upsert(tmp_db_path: Path):
    from src.data.fetch_data_sqlite import get_conn, init_schema, _iso_utc, _upsert

    idx = pd.date_range("2024-01-01", periods=3, freq="H", tz="UTC")
    s = _iso_utc(pd.Series(idx))
    assert s.str.endswith("Z").all()

    conn = get_conn(tmp_db_path)
    try:
        init_schema(conn)
        # create a simple target table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS t (a INTEGER PRIMARY KEY, b REAL)"
        )
        df = pd.DataFrame({"a": [1, 2, 2], "b": [0.1, 0.2, 0.3]})
        _upsert(conn, df, "t")
        rows = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert rows == 2
    finally:
        conn.close()


@pytest.mark.usefixtures("stub_databento_module")
def test_check_data_exists(tmp_db_path: Path):
    from src.data.fetch_data_sqlite import get_conn, init_schema, check_data_exists

    conn = get_conn(tmp_db_path)
    try:
        init_schema(conn)
        # Insert a minimal valid row into the already-created atm_slices_1h table
        ts = pd.Timestamp("2024-01-01 14:00", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        conn.execute(
            """
            INSERT INTO atm_slices_1h (
                ticker, ts_event, expiry_date, opt_symbol, stock_symbol,
                opt_close, stock_close, opt_volume, stock_volume,
                option_type, strike_price, time_to_expiry, moneyness
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("AAPL", ts, ts, "240101C00100000", "AAPL", 1.0, 100.0, 1, 10, "C", 100.0, 1.0/365, 0.0),
        )
        conn.commit()

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        assert check_data_exists(conn, "AAPL", start, end, timeframe="1h") is True
        assert check_data_exists(conn, "MSFT", start, end, timeframe="1h") is False
    finally:
        conn.close()


@pytest.mark.usefixtures("stub_databento_module")
def test__populate_atm_inserts_one_row(tmp_db_path: Path):
    from src.data.fetch_data_sqlite import get_conn, init_schema
    # Access internal function via import after module load
    import importlib
    mod = importlib.import_module("src.data.fetch_data_sqlite")

    conn = get_conn(tmp_db_path)
    try:
        init_schema(conn)
        # Insert into existing processed_merged_1h with full column list
        ts = pd.Timestamp("2024-01-01 14:00", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        ex = pd.Timestamp("2024-02-01", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        conn.executemany(
            """
            INSERT INTO processed_merged_1h (
                ticker, ts_event, opt_symbol, stock_symbol,
                opt_close, stock_close, opt_volume, stock_volume,
                expiry_date, option_type, strike_price, time_to_expiry, moneyness
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("AAPL", ts, "240201C00100000", "AAPL", 2.0, 100.0, 10, 1000, ex, "C", 100.0, 30/365, 0.0),
                ("AAPL", ts, "240201C00090000", "AAPL", 2.1, 100.0, 8, 900, ex, "C", 90.0, 30/365, -0.1),
            ],
        )
        mod._populate_atm(conn, "AAPL", timeframe="1h")
        n = conn.execute("SELECT COUNT(*) FROM atm_slices_1h").fetchone()[0]
        assert n == 1
    finally:
        conn.close()
