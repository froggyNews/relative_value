import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from src.data import data_loader_coordinator as dlc


def test__safe_table_exists_and_data_exists(tmp_db_path: Path):
    conn = sqlite3.connect(str(tmp_db_path))
    try:
        conn.execute(
            "CREATE TABLE atm_slices_1m (ticker TEXT, ts_event TEXT, expiry_date TEXT)"
        )
        assert dlc._table_exists(conn, "atm_slices_1m") is True

        assert dlc._table_exists(conn, "nope_table") is False
    finally:
        conn.close()


def test__calculate_iv_edge_cases():
    # Non-finite inputs should return NaN
    assert np.isnan(dlc._calculate_iv(np.nan, 100, 100, 1, "C", 0.01))
    # Non-positive inputs should return NaN
    assert np.isnan(dlc._calculate_iv(1, 0, 100, 1, "P", 0.01))
    # Price near/below intrinsic should return a small epsilon IV
    intrinsic = max(100 - 90, 0.0)
    iv = dlc._calculate_iv(intrinsic, 100, 90, 1, "C", 0.01)
    assert iv == pytest.approx(1e-6)


def _make_min_schema(conn: sqlite3.Connection, table: str):
    conn.execute(
        f"""
        CREATE TABLE {table} (
            ticker TEXT, ts_event TEXT, expiry_date TEXT,
            opt_symbol TEXT, stock_symbol TEXT,
            opt_close REAL, stock_close REAL,
            opt_volume REAL, stock_volume REAL,
            option_type TEXT, strike_price REAL,
            time_to_expiry REAL, moneyness REAL
        )
        """
    )


def test_load_ticker_core_prefers_atm_when_available(tmp_db_path: Path, monkeypatch):
    # Monkeypatch IV calculation to a constant to avoid numeric brittleness
    monkeypatch.setattr(dlc, "_calculate_iv", lambda *a, **k: 0.2)

    with sqlite3.connect(str(tmp_db_path)) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        _make_min_schema(conn, "atm_slices_1m")
        rows = [
            (
                "XYZ",
                pd.Timestamp("2024-01-01 14:30", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                pd.Timestamp("2024-02-01", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "240201C00100000",
                "XYZ",
                2.5,
                100.0,
                10,
                1000,
                "C",
                100.0,
                30 / 365,
                0.0,
            )
        ]
        conn.executemany(
            "INSERT INTO atm_slices_1m VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()

    df = dlc.load_ticker_core(
        ticker="XYZ",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-02", tz="UTC"),
        db_path=tmp_db_path,
        atm_only=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"ts_event", "expiry_date", "iv", "symbol"}.issubset(df.columns)
    assert (df["symbol"] == "XYZ").all()
    # iv_clip should be present and clipped non-negative
    assert "iv_clip" in df.columns
    assert (df["iv_clip"] >= 1e-6).all()


def test_load_ticker_core_uses_processed_when_atm_empty(tmp_db_path: Path, monkeypatch):
    monkeypatch.setattr(dlc, "_calculate_iv", lambda *a, **k: 0.3)
    with sqlite3.connect(str(tmp_db_path)) as conn:
        _make_min_schema(conn, "processed_merged_1m")
        # Two strikes so the ATM selection logic (ROW_NUMBER) has work to do
        base_ts = pd.Timestamp("2024-01-01 14:30", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        exp_ts = pd.Timestamp("2024-02-01", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        conn.executemany(
            "INSERT INTO processed_merged_1m VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                ("XYZ", base_ts, "240201C00100000", "XYZ", 2.5, 100.0, 10, 1000, exp_ts, "C", 100.0, 30 / 365, 0.0),
                ("XYZ", base_ts, "240201C00110000", "XYZ", 2.0, 101.0, 5, 900, exp_ts, "C", 101.0, 30 / 365, 0.01),
            ],
        )
        conn.commit()

    df = dlc.load_ticker_core(
        ticker="XYZ",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-02", tz="UTC"),
        db_path=tmp_db_path,
        atm_only=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "iv" in df.columns

