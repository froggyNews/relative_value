import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import relative_value as rv


# ---------------------------
# DB discovery helpers
# ---------------------------

def _find_table_with_data(db_path: Path) -> tuple[str, list[str]]:
    if not db_path.exists():
        return "", []
    candidates = [
        "atm_slices_1m",
        "processed_merged_1m",
        "atm_slices_1h",
        "processed_merged",
        "merged_1m",
    ]
    with sqlite3.connect(str(db_path)) as conn:
        for table in candidates:
            try:
                cur = conn.execute(
                    f"SELECT ticker, COUNT(1) cnt FROM {table} GROUP BY ticker "
                    "HAVING cnt > 100 ORDER BY cnt DESC LIMIT 8"
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
    start = (mx_ts - pd.Timedelta(days=1)).normalize()
    end = mx_ts
    return start, end


# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture(scope="module")
def db_path() -> Path:
    return Path("data/iv_data_1m.db")


@pytest.fixture(scope="module")
def universe(db_path):
    table, tickers = _find_table_with_data(db_path)
    if not table or len(tickers) < 2:
        pytest.skip("No suitable table/tickers found in local DB")
    tickers = tickers[:4]
    start, end = _find_recent_window(db_path, table, tickers)
    if not pd.notna(start) or not pd.notna(end) or start >= end:
        pytest.skip("Could not determine a recent window with data")
    return {"table": table, "tickers": tickers, "start": start, "end": end}


# ---------------------------
# Core integration tests
# ---------------------------

@pytest.mark.integration
def test_build_pooled_dataset_time_safe_from_db(db_path, universe):
    cores = rv.load_cores_with_auto_fetch(
        universe["tickers"],
        universe["start"],
        universe["end"],
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
    )
    assert cores, "Expected non-empty cores from DB"
    pooled = rv.build_pooled_iv_return_dataset_time_safe(
        tickers=universe["tickers"],
        start=universe["start"],
        end=universe["end"],
        r=0.045,
        forward_steps=1,
        tolerance="15s",
        db_path=db_path,
        cores=cores,
    )
    assert isinstance(pooled, pd.DataFrame)
    assert not pooled.empty
    assert "iv_ret_fwd" in pooled.columns
    # Symbol one-hots should exist for requested tickers (at least some)
    sym_cols = {f"sym_{t}" for t in universe["tickers"]}
    assert len(sym_cols.intersection(pooled.columns)) >= 1


@pytest.mark.integration
def test_time_respecting_split_no_leakage(db_path, universe):
    cfg = rv.BacktestConfig(
        tickers=universe["tickers"],
        start=universe["start"].isoformat(),
        end=universe["end"].isoformat(),
        test_frac=0.2,
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
    )
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, *_ = rv.build_and_split_pooled(cfg)
    assert len(X_tr) > 0 and len(X_te) > 0
    # No index overlap
    assert set(X_tr.index).isdisjoint(set(X_te.index))
    # If timestamps exist on both sides, train max < test min
    if ts_tr.notna().any() and ts_te.notna().any():
        assert pd.to_datetime(ts_tr).max() < pd.to_datetime(ts_te).min()


@pytest.mark.integration
def test_prediction_frame_shape_and_symbol_edges(db_path, universe):
    cfg = rv.BacktestConfig(
        tickers=universe["tickers"],
        start=universe["start"].isoformat(),
        end=universe["end"].isoformat(),
        test_frac=0.2,
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
    )
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, strike_tr, strike_te, expiry_tr, expiry_te = rv.build_and_split_pooled(cfg)
    model = rv.train_pooled_model_on_split(X_tr, y_tr)
    preds = rv.make_prediction_frame(model, X_te, y_te, ts_te, strike_te, expiry_te)
    assert {"ts_event", "y_true", "y_pred"}.issubset(preds.columns)
    assert "symbol" in preds.columns
    # Edges are optional; if present, they should be numeric and same length
    for c in ("edge_1m", "edge_15m", "edge_60m"):
        if c in preds.columns:
            assert len(preds[c]) == len(preds)


@pytest.mark.integration
def test_make_trades_basic_and_threshold(db_path, universe):
    cfg = rv.BacktestConfig(
        tickers=universe["tickers"],
        start=universe["start"].isoformat(),
        end=universe["end"].isoformat(),
        test_frac=0.2,
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
        top_k=1,
        threshold=0.0,
    )
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, strike_tr, strike_te, expiry_tr, expiry_te = rv.build_and_split_pooled(cfg)
    model = rv.train_pooled_model_on_split(X_tr, y_tr)
    preds = rv.make_prediction_frame(model, X_te, y_te, ts_te, strike_te, expiry_te)

    trades0 = rv.make_relative_value_trades(preds, top_k=1, threshold=0.0, pair_only=False, group_freq="1min")
    assert isinstance(trades0, pd.DataFrame)
    if not trades0.empty:
        required = {"ts_event", "long_symbols", "short_symbols", "pred_spread", "realized_spread"}
        assert required.issubset(trades0.columns)
        # threshold should reduce trade count
        trades_hi = rv.make_relative_value_trades(preds, top_k=1, threshold=abs(trades0["pred_spread"]).quantile(0.75), pair_only=False, group_freq="1min")
        assert len(trades_hi) <= len(trades0)


@pytest.mark.integration
def test_pair_only_enforces_two_names(db_path, universe):
    cfg = rv.BacktestConfig(
        tickers=universe["tickers"],
        start=universe["start"].isoformat(),
        end=universe["end"].isoformat(),
        test_frac=0.2,
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
    )
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, strike_tr, strike_te, expiry_tr, expiry_te = rv.build_and_split_pooled(cfg)
    model = rv.train_pooled_model_on_split(X_tr, y_tr)
    preds = rv.make_prediction_frame(model, X_te, y_te, ts_te, strike_te, expiry_te)
    trades = rv.make_relative_value_trades(preds, top_k=3, pair_only=True, threshold=0.0, group_freq="1min")
    if not trades.empty:
        for _, row in trades.iterrows():
            longs = [s for s in str(row["long_symbols"]).split(",") if s]
            shorts = [s for s in str(row["short_symbols"]).split(",") if s]
            assert len(longs) == 1 and len(shorts) == 1
            assert set(longs).isdisjoint(set(shorts))


@pytest.mark.integration
def test_group_freq_bucketting_changes_trade_timestamps(db_path, universe):
    cfg = rv.BacktestConfig(
        tickers=universe["tickers"],
        start=universe["start"].isoformat(),
        end=universe["end"].isoformat(),
        test_frac=0.2,
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
    )
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, strike_tr, strike_te, expiry_tr, expiry_te = rv.build_and_split_pooled(cfg)
    model = rv.train_pooled_model_on_split(X_tr, y_tr)
    preds = rv.make_prediction_frame(model, X_te, y_te, ts_te, strike_te, expiry_te)

    t1 = rv.make_relative_value_trades(preds, top_k=1, threshold=0.0, group_freq="1min")
    t5 = rv.make_relative_value_trades(preds, top_k=1, threshold=0.0, group_freq="5min")
    if not t1.empty and not t5.empty:
        # Coarser buckets should yield fewer or equal trade rows
        assert len(t5) <= len(t1)


@pytest.mark.integration
def test_liquidity_filter_effect(db_path, universe):
    cfg = rv.BacktestConfig(
        tickers=universe["tickers"],
        start=universe["start"].isoformat(),
        end=universe["end"].isoformat(),
        test_frac=0.2,
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
    )
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, *_ = rv.build_and_split_pooled(cfg)
    model = rv.train_pooled_model_on_split(X_tr, y_tr)
    preds = rv.make_prediction_frame(model, X_te, y_te, ts_te, None, None)

    # Baseline (no liquidity filter)
    base = rv.make_relative_value_trades(preds, top_k=1, threshold=0.0, min_roll_trades=0.0, roll_by_rows=True, roll_bars=10, group_freq="1min")
    # Aggressive liquidity filter should remove equal/more rows
    filt = rv.make_relative_value_trades(preds, top_k=1, threshold=0.0, min_roll_trades=1e9, roll_by_rows=True, roll_bars=10, group_freq="1min")
    assert len(filt) <= len(base)


@pytest.mark.integration
def test_contract_filters_if_available(db_path, universe):
    cfg = rv.BacktestConfig(
        tickers=universe["tickers"],
        start=universe["start"].isoformat(),
        end=universe["end"].isoformat(),
        test_frac=0.2,
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
    )
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, strike_tr, strike_te, expiry_tr, expiry_te = rv.build_and_split_pooled(cfg)
    model = rv.train_pooled_model_on_split(X_tr, y_tr)
    preds = rv.make_prediction_frame(model, X_te, y_te, ts_te, strike_te, expiry_te)

    # Identify a (strike, expiry) that actually exists in preds
    if "strike" not in preds.columns or "expiry" not in preds.columns:
        pytest.skip("No strike/expiry columns present")
    df = preds.dropna(subset=["strike", "expiry"])
    if df.empty:
        pytest.skip("No non-null strike/expiry to test")
    sample = df.iloc[0]
    strike = float(sample["strike"])
    expiry = pd.to_datetime(sample["expiry"]).date().isoformat()

    t_all = rv.make_relative_value_trades(preds, top_k=1, threshold=0.0, group_freq="1min")
    t_filt = rv.make_relative_value_trades(preds, top_k=1, threshold=0.0, strike=strike, expiry=expiry, strike_tol=0.0, group_freq="1min")
    # Filtered set should be <= unfiltered
    assert len(t_filt) <= len(t_all)
    # If non-empty, all rows should reflect the filtered date (strike uniqueness is per-bucket best-effort)
    if not t_filt.empty and "expiry" in t_filt.columns:
        # expiry is recorded as iso string when unique in the bucket
        exps = [e for e in t_filt["expiry"].tolist() if e]
        if exps:
            assert all(e == expiry for e in exps)


@pytest.mark.integration
def test_summarize_trades_consistency(db_path, universe):
    cfg = rv.BacktestConfig(
        tickers=universe["tickers"],
        start=universe["start"].isoformat(),
        end=universe["end"].isoformat(),
        test_frac=0.2,
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
        top_k=1,
    )
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, *_ = rv.build_and_split_pooled(cfg)
    model = rv.train_pooled_model_on_split(X_tr, y_tr)
    preds = rv.make_prediction_frame(model, X_te, y_te, ts_te, None, None)
    trades = rv.make_relative_value_trades(preds, top_k=1, threshold=0.0, group_freq="1min")
    summary = rv.summarize_trades(trades)
    assert "n_trades" in summary
    if summary["n_trades"] > 0:
        for k in ["avg_edge_1m", "avg_edge_15m", "avg_edge_60m", "hit_rate_1m", "hit_rate_15m", "hit_rate_60m"]:
            assert k in summary

@pytest.mark.integration
def test_summarize_trades_consistency_comprehensive(db_path, universe):
    cfg = rv.BacktestConfig(
        tickers=universe["tickers"],
        start=universe["start"].isoformat(),
        end=universe["end"].isoformat(),
        test_frac=0.2,
        db_path=db_path,
        auto_fetch=False,
        atm_only=True,
        top_k=1,
    )
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, *_ = rv.build_and_split_pooled(cfg)
    model = rv.train_pooled_model_on_split(X_tr, y_tr)
    preds = rv.make_prediction_frame(model, X_te, y_te, ts_te, None, None)
    trades = rv.make_relative_value_trades(preds, top_k=1, threshold=0.0, group_freq="1min")
    summary = rv.summarize_trades(trades)

    # --- Assertions ---
    assert "n_trades" in summary
    if summary["n_trades"] > 0:
        for k in ["avg_edge_1m", "avg_edge_15m", "avg_edge_60m",
                  "hit_rate_1m", "hit_rate_15m", "hit_rate_60m"]:
            assert k in summary

    # --- Debug/summary output ---
    print("\n==== Backtest Summary ====")
    print(f"Number of trades: {summary['n_trades']}")
    if summary["n_trades"] > 0:
        print(f"Avg Edge 1m:   {summary['avg_edge_1m']:.6f}")
        print(f"Avg Edge 15m:  {summary['avg_edge_15m']:.6f}")
        print(f"Avg Edge 60m:  {summary['avg_edge_60m']:.6f}")
        print(f"Hit Rate 1m:   {summary['hit_rate_1m']:.3f}")
        print(f"Hit Rate 15m:  {summary['hit_rate_15m']:.3f}")
        print(f"Hit Rate 60m:  {summary['hit_rate_60m']:.3f}")
