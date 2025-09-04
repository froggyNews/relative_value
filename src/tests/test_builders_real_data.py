# tests/test_builders_db.py
import sqlite3
from pathlib import Path
import pandas as pd
import pytest

import src.feature_engineering as m

# If you prefer to use your coordinator’s loader directly:
try:
    # adjust import path/name to your project
    from src.data.data_loader_coordinator import load_cores_with_auto_fetch
    
    print ("Imported load_cores_with_auto_fetch from data_loader_coordinator")
    HAVE_COORDINATOR = True
except Exception:
    HAVE_COORDINATOR = False


# ---------------------------
# DB discovery helpers
# ---------------------------

def _find_table_with_data(db_path: Path) -> tuple[str, list[str]]:
    if not db_path.exists():
        return "", []
    candidates = [
        "atm_slices_1h", "processed_merged_1h",
        "atm_slices_1m", "processed_merged_1m",
        "processed_merged", "merged_1m",
    ]
    with sqlite3.connect(str(db_path)) as conn:
        for table in candidates:
            try:
                cur = conn.execute(
                    f"""
                    SELECT ticker, COUNT(1) cnt
                    FROM {table}
                    GROUP BY ticker
                    HAVING cnt > 100
                    ORDER BY cnt DESC
                    LIMIT 8
                    """
                )
                rows = cur.fetchall()
            except Exception:
                continue
            if rows:
                return table, [r[0] for r in rows]
    return "", []

def _find_recent_window(db_path: Path, table: str, tickers: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    with sqlite3.connect(str(db_path)) as conn:
        q = f"SELECT MIN(ts_event), MAX(ts_event) FROM {table} WHERE ticker IN ({','.join(['?']*len(tickers))})"
        mn, mx = conn.execute(q, tickers).fetchone()
    if mn is None or mx is None:
        return pd.NaT, pd.NaT
    mx_ts = pd.to_datetime(mx, utc=True, errors="coerce")
    start = (mx_ts - pd.Timedelta(days=2)).normalize()   # give a bit more room
    end = mx_ts
    return start, end


# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture(scope="module")
def db_path():
    # default hour DB path in your module; change if needed
    path = getattr(m, "DEFAULT_DB_PATH", Path("data/iv_data_1h.db"))
    return Path(path)

@pytest.fixture(scope="module")
def universe(db_path):
    table, tickers = _find_table_with_data(db_path)
    if not table or len(tickers) < 2:
        pytest.skip("No suitable table/tickers found in local DB for integration tests")
    tickers = tickers[:4]
    start, end = _find_recent_window(db_path, table, tickers)
    if not pd.notna(start) or not pd.notna(end) or start >= end:
        pytest.skip("Could not determine a recent window with data")
    return {"table": table, "tickers": tickers, "start": start, "end": end}


# ---------------------------
# Integration tests (DB-backed)
# ---------------------------

@pytest.mark.integration
def test_build_pooled_from_db(universe, db_path):
    if not HAVE_COORDINATOR:
        pytest.skip("data_loader_coordinator.load_cores_with_auto_fetch not available")
    cores = load_cores_with_auto_fetch(
        universe["tickers"],
        universe["start"],
        universe["end"],
        Path(db_path),
        auto_fetch=False,
        atm_only=False,   # keep broad coverage; tune if you store ATM-only
    )
    assert cores, "Expected non-empty cores from DB"
    pooled = m.build_pooled_iv_return_dataset_time_safe(
        tickers=universe["tickers"],
        start=universe["start"],
        end=universe["end"],
        r=0.045,
        forward_steps=1,
        tolerance="15s",
        db_path=db_path,
        cores=cores,
        debug=False,
    )
    assert isinstance(pooled, pd.DataFrame)
    assert not pooled.empty
    assert "iv_ret_fwd" in pooled.columns
    onehots = [f"sym_{t}" for t in universe["tickers"]]
    for c in onehots:
        assert c in pooled.columns
    # Normalization attrs
    assert "norm_means" in pooled.attrs and "norm_stds" in pooled.attrs

@pytest.mark.integration
def test_build_per_ticker_from_db(universe, db_path):
    if not HAVE_COORDINATOR:
        pytest.skip("data_loader_coordinator.load_cores_with_auto_fetch not available")
    cores = load_cores_with_auto_fetch(
        universe["tickers"],
        universe["start"],
        universe["end"],
        Path(db_path),
        auto_fetch=False,
        atm_only=False,
    )
    datasets = m.build_iv_return_dataset_time_safe(
        tickers=universe["tickers"],
        start=universe["start"],
        end=universe["end"],
        r=0.045,
        forward_steps=3,
        tolerance="15s",
        db_path=db_path,
        cores=cores,
        debug=False,
    )
    assert set(datasets.keys()) == set(universe["tickers"])
    for t, df in datasets.items():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "iv_ret_fwd" in df.columns
        # leak columns removed in per-ticker finalize
        assert all(not c.startswith("IV_") for c in df.columns if c != "ts_event")
        assert all(not c.startswith("IVRET_") for c in df.columns if c != "ts_event")

@pytest.mark.integration
def test_build_target_peer_from_db(universe, db_path):
    if not HAVE_COORDINATOR:
        pytest.skip("data_loader_coordinator.load_cores_with_auto_fetch not available")
    target = universe["tickers"][0]
    cores = load_cores_with_auto_fetch(
        universe["tickers"],
        universe["start"],
        universe["end"],
        Path(db_path),
        auto_fetch=False,
        atm_only=False,
    )
    ds = m.build_target_peer_dataset(
        target=target,
        tickers=universe["tickers"],
        start=universe["start"],
        end=universe["end"],
        r=0.045,
        forward_steps=1,
        tolerance="15s",
        db_path=db_path,
        cores=cores,
        target_kind="iv_ret",
        debug=False,
    )
    assert isinstance(ds, pd.DataFrame)
    assert not ds.empty
    assert "y" in ds.columns
    assert "symbol" not in ds.columns
    # Per-target finalize removes peer IV columns
    assert all(not c.startswith("IV_") for c in ds.columns if c != "ts_event")
    assert all(not c.startswith("IVRET_") for c in ds.columns if c != "ts_event")
