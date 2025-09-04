
from __future__ import annotations

from typing import Optional, Dict, List
import numpy as np
import pandas as pd

def make_relative_value_trades(
    preds: pd.DataFrame,
    top_k: int = 1,
    threshold: float = 0.0,
    min_roll_trades: float = 0.0,
    roll_window: str = "30min",
    roll_by_rows: bool = True,
    roll_bars: Optional[int] = 30,
    pair_only: bool = False,
    strike: Optional[float] = None,
    expiry: Optional[str] = None,
    strike_tol: float = 0.0,
    group_freq: str = "1min",
) -> pd.DataFrame:
    """Form long/short trades per timestamp using cross-sectional predictions."""
    # Ensure ts_event column exists
    if "ts_event" not in preds.columns:
        preds = preds.copy()
        preds["ts_event"] = pd.NaT
    df = preds.dropna(subset=["ts_event", "y_pred", "y_true"]).copy()
    if "symbol" not in df.columns:
        df["symbol"] = None

    # Compute rolling net trades/volume per symbol
    try:
        if "opt_volume" not in df.columns:
            df["opt_volume"] = 0.0
        if roll_by_rows and (roll_bars is not None and roll_bars > 0):
            df = df.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
            df["roll_trades"] = (
                df.groupby("symbol", group_keys=False)["opt_volume"]
                  .rolling(int(roll_bars), min_periods=1).sum()
                  .reset_index(level=0, drop=True)
            )
        else:
            if "ts_event" in df.columns:
                df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
                df = df.dropna(subset=["ts_event"]).copy()
                df = df.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
                def _roll(g: pd.DataFrame) -> pd.Series:
                    g = g.set_index("ts_event")
                    return g["opt_volume"].rolling(roll_window).sum()
                df["roll_trades"] = (
                    df.groupby("symbol", group_keys=False).apply(_roll).reset_index(level=0, drop=True)
                )
            else:
                df["roll_trades"] = np.nan
    except Exception:
        df["roll_trades"] = np.nan

    # Liquidity filter
    if min_roll_trades and min_roll_trades > 0:
        df = df[df["roll_trades"] >= float(min_roll_trades)]

    # Optional contract filters
    if strike is not None and "strike" in df.columns:
        tol = abs(float(strike_tol)) if strike_tol is not None else 0.0
        if tol > 0:
            df = df[np.abs(df["strike"].astype(float) - float(strike)) <= tol]
        else:
            df = df[df["strike"].astype(float) == float(strike)]
    if expiry is not None and "expiry" in df.columns:
        try:
            exp_ts = pd.to_datetime(expiry, utc=True).normalize()
            df_exp = pd.to_datetime(df["expiry"], utc=True, errors="coerce").dt.normalize()
            df = df[df_exp == exp_ts]
        except Exception:
            pass

    # Cross-sectional grouping key: bucket timestamps
    if "ts_event" in df.columns:
        ts_dt = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df["ts_bucket"] = ts_dt.dt.floor(group_freq)
        key_col = "ts_bucket"
    else:
        key_col = "ts_event"

    trades = []
    for ts, grp in df.groupby(key_col):
        grp = grp.dropna(subset=["y_pred", "y_true"]).copy()
        # Collapse to one row per symbol
        if not grp.empty and "symbol" in grp.columns:
            agg_cols = {"y_pred": "mean", "y_true": "mean"}
            for c in ("edge_1m", "edge_15m", "edge_60m"):
                if c in grp.columns:
                    agg_cols[c] = "mean"
            grp_sym = grp.groupby("symbol", as_index=False).agg(agg_cols)
        else:
            grp_sym = grp.copy()

        if pair_only:
            if grp_sym["symbol"].nunique() < 2:
                continue
            two = grp_sym.sort_values("y_pred", ascending=False)
            long_leg = two.head(1)
            short_leg = two.tail(1)
        else:
            if len(grp_sym) < max(2, 2 * top_k):
                continue
            grp_sym = grp_sym.sort_values("y_pred", ascending=False)
            long_leg = grp_sym.head(top_k)
            short_leg = grp_sym.tail(top_k)

        pred_spread = float(long_leg["y_pred"].mean() - short_leg["y_pred"].mean())
        if abs(pred_spread) < threshold:
            continue

        def _spread(col: str) -> float:
            if col not in grp.columns:
                return float("nan")
            return float(long_leg[col].mean() - short_leg[col].mean())

        realized_spread = float(long_leg["y_true"].mean() - short_leg["y_true"].mean())
        edge_1m = _spread("edge_1m")
        edge_15m = _spread("edge_15m")
        edge_60m = _spread("edge_60m")

        exp_out = ""
        if "expiry" in grp.columns:
            euniq = pd.to_datetime(grp["expiry"], errors="coerce").dropna().dt.normalize().unique()
            if len(euniq) == 1:
                exp_out = pd.to_datetime(euniq[0]).date().isoformat()

        strike_out = ""
        if "strike" in grp.columns:
            sunique = pd.to_numeric(grp["strike"], errors="coerce").dropna().unique()
            if len(sunique) == 1:
                strike_out = float(sunique[0])

        trades.append({
            "ts_event": ts,
            "long_symbols": ",".join([str(s) for s in long_leg["symbol"].tolist()]),
            "short_symbols": ",".join([str(s) for s in short_leg["symbol"].tolist()]),
            "strike": strike_out,
            "expiry": exp_out,
            "pred_spread": pred_spread,
            "realized_spread": realized_spread,
            "edge_1m": edge_1m,
            "edge_15m": edge_15m,
            "edge_60m": edge_60m,
            "correct": int(np.sign(pred_spread) == np.sign(realized_spread) and realized_spread != 0.0),
            "n_universe": int(len(grp)),
        })

    result = pd.DataFrame(trades)
    if result.empty:
        return result
    if "ts_event" in result.columns:
        result = result.sort_values("ts_event").reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)
    return result

def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades is None or trades.empty:
        return {"n_trades": 0}
    def _avg(col: str) -> float:
        return float(pd.to_numeric(trades[col], errors="coerce").mean()) if col in trades.columns else float("nan")
    def _hit(col: str) -> float:
        if col not in trades.columns:
            return float("nan")
        s = pd.to_numeric(trades[col], errors="coerce")
        ps = pd.to_numeric(trades["pred_spread"], errors="coerce")
        m = s.notna() & ps.notna() & (ps != 0)
        if not m.any():
            return float("nan")
        return float((np.sign(s[m]) == np.sign(ps[m])).mean())
    avg_1 = _avg("edge_1m")
    avg_15 = _avg("edge_15m")
    avg_60 = _avg("edge_60m")
    hr_1 = _hit("edge_1m")
    hr_15 = _hit("edge_15m")
    hr_60 = _hit("edge_60m")
    return {
        "n_trades": int(len(trades)),
        "avg_edge_1m": avg_1,
        "avg_edge_15m": avg_15,
        "avg_edge_60m": avg_60,
        "hit_rate_1m": hr_1,
        "hit_rate_15m": hr_15,
        "hit_rate_60m": hr_60,
    }
