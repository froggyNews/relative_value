from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import relative_value as rv


def _choose_universe_and_window(db_path: Path) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    """Pick a reasonable universe and static window; fall back if empty."""
    # Prefer your common universe; adjust as needed
    candidates = ["QBTS", "IONQ", "RGTI", "QUBT", "ASTS"]
    # Use a fixed 1-day window known to exist in typical DB snapshots
    start = pd.Timestamp("2025-08-04", tz="UTC")
    end = pd.Timestamp("2025-08-05 19:59", tz="UTC")
    # Filter to tickers that actually load cores in this window
    cores = rv.load_cores_with_auto_fetch(candidates, start, end, db_path=db_path, auto_fetch=False, atm_only=True)
    if not cores:
        return [], pd.NaT, pd.NaT
    tickers = [t for t, df in cores.items() if df is not None and not df.empty][:4]
    return tickers, start, end


@pytest.mark.integration
def test_build_pooled_dataset_time_safe_from_db():
    db_path = Path("data/iv_data_1m.db")
    tickers, start, end = _choose_universe_and_window(db_path)
    if len(tickers) < 2 or not pd.notna(start) or not pd.notna(end) or start >= end:
        pytest.skip("Could not determine a recent window with data")

    cores = rv.load_cores_with_auto_fetch(tickers, start, end, db_path=db_path, auto_fetch=False, atm_only=True)
    if not cores:
        pytest.skip("Cores could not be loaded from DB; skipping integration test")

    pooled = rv.build_pooled_iv_return_dataset_time_safe(
        tickers=tickers, start=start, end=end, r=0.045, forward_steps=1, tolerance="15s", db_path=db_path, cores=cores
    )
    assert isinstance(pooled, pd.DataFrame)
    assert not pooled.empty
    assert "iv_ret_fwd" in pooled.columns
    # Should have symbol one-hots for requested tickers
    sym_cols = [f"sym_{t}" for t in tickers]
    assert any(c in pooled.columns for c in sym_cols)


@pytest.mark.integration
def test_run_backtest_end_to_end_real_data():
    db_path = Path("data/iv_data_1m.db")
    tickers, start, end = _choose_universe_and_window(db_path)
    if len(tickers) < 2 or not pd.notna(start) or not pd.notna(end) or start >= end:
        pytest.skip("Could not determine a recent window with data")

    cfg = rv.BacktestConfig(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        forward_steps=1,
        test_frac=0.2,
        tolerance="15s",
        r=0.045,
        db_path=db_path,
        top_k=1,
        threshold=0.0,  # allow trades even with constant preds if xgboost stubbed
        auto_fetch=False,  # do not hit network
        atm_only=True,
        min_roll_trades=0.0,
        roll_window="30min",
        roll_by_rows=True,
        roll_bars=10,
        pair_only=False,
    )

    results = rv.run_backtest(cfg)
    assert isinstance(results, dict)
    assert "summary" in results and "trades" in results and "preds_head" in results

    trades = results["trades"]
    assert isinstance(trades, pd.DataFrame)
    # If trades exist, validate structure and logic constraints
    if not trades.empty:
        required_cols = {"ts_event", "long_symbols", "short_symbols", "pred_spread", "realized_spread"}
        assert required_cols.issubset(trades.columns)
        # Ensure non-overlapping legs and correct counts for top_k=1 when pair_only is False
        for _, row in trades.iterrows():
            longs = [s for s in str(row["long_symbols"]).split(",") if s]
            shorts = [s for s in str(row["short_symbols"]).split(",") if s]
            assert set(longs).isdisjoint(set(shorts))
            assert len(longs) == 1 and len(shorts) == 1

    # Summary always present
    summary = results["summary"]
    assert "n_trades" in summary
    # When trades exist, additional metrics should be present
    if summary.get("n_trades", 0) > 0:
        for k in [
            "avg_edge_1m",
            "avg_edge_15m",
            "avg_edge_60m",
            "hit_rate_1m",
            "hit_rate_15m",
            "hit_rate_60m",
        ]:
            assert k in summary
