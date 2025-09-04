from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.trades import make_relative_value_trades, summarize_trades


def _synthetic_preds(n: int = 20) -> pd.DataFrame:
    # Build two-symbol panel over time with known patterns
    ts = pd.date_range("2024-01-01 09:30", periods=n, freq="1min", tz="UTC")
    rows = []
    for i, t in enumerate(ts):
        for sym in ("AAA", "BBB"):
            # Make even timestamps favor AAA over BBB; odds are neutral
            if i % 2 == 0:
                y_pred = 1.0 if sym == "AAA" else -1.0
                y_true = 0.8 if sym == "AAA" else -0.8
            else:
                y_pred = 0.0
                y_true = 0.0

            dte = 30.0  # in range
            if i in (3, 7):  # some rows out of DTE range
                dte = 10.0

            lm = 0.0
            if i in (5,):  # some rows out of moneyness range
                lm = 1.0

            vol = 100.0
            if i < 3:  # low early volume to fail liquidity roll
                vol = 0.0

            rows.append({
                "ts_event": t,
                "symbol": sym,
                "y_pred": y_pred,
                "y_true": y_true,
                "opt_volume": vol,
                "dte": dte,
                "log_moneyness": lm,
            })
    return pd.DataFrame(rows)


def test_make_relative_value_trades_debug_summary():
    preds = _synthetic_preds(30)
    top_k = 1
    threshold = 0.0
    group_freq = "1min"
    dte_range = (20.0, 45.0)
    moneyness_range = (-0.5, 0.5)
    roll_by_rows = True
    roll_bars = 3
    min_roll_trades = 50.0

    # Baseline input stats
    print("[DEBUG] input_rows:", len(preds))
    print("[DEBUG] symbols:", sorted(preds["symbol"].unique().tolist()))
    print("[DEBUG] n_ts:", preds["ts_event"].nunique())

    # DTE/moneyness filter preview
    dte_ok = preds["dte"].between(dte_range[0], dte_range[1], inclusive="both")
    mon_ok = preds["log_moneyness"].between(moneyness_range[0], moneyness_range[1], inclusive="both")
    print("[DEBUG] dte_out_of_range:", int((~dte_ok).sum()))
    print("[DEBUG] moneyness_out_of_range:", int((~mon_ok).sum()))

    # Liquidity roll preview (row-based)
    df_roll = preds.sort_values(["symbol", "ts_event"]).copy()
    df_roll["roll_trades"] = (
        df_roll.groupby("symbol", group_keys=False)["opt_volume"]
               .rolling(int(roll_bars), min_periods=1)
               .sum()
               .reset_index(level=0, drop=True)
    )
    liq_fail = int((df_roll["roll_trades"] < min_roll_trades).sum())
    print("[DEBUG] rows_below_liquidity_threshold:", liq_fail)

    # Run trade construction
    trades = make_relative_value_trades(
        preds,
        top_k=top_k,
        threshold=threshold,
        min_roll_trades=min_roll_trades,
        roll_by_rows=roll_by_rows,
        roll_bars=roll_bars,
        group_freq=group_freq,
        granularity="slice",
        weighting_scheme="opt_volume",
        dte_range=dte_range,
        moneyness_range=moneyness_range,
    )

    summary = summarize_trades(trades)
    print("[DEBUG] trades_formed:", summary.get("n_trades", 0))
    print("[DEBUG] avg_edge_1m:", summary.get("avg_edge_1m"))
    print("[DEBUG] hit_rate_1m:", summary.get("hit_rate_1m"))

    # Sanity assertions (not strict values to keep the test robust)
    assert "n_trades" in summary
    # With our synthetic data, we should form at least one trade after filters
    assert summary["n_trades"] >= 1

