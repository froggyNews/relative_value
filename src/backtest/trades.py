from __future__ import annotations

from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd


# ===========================
# Public API
# ===========================
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
    granularity: str = "slice",                 # "contract" | "slice" | "surface"
    weighting_scheme: str = "opt_volume",       # "opt_volume" | "vega" | "vega_x_liquidity" | "equal"
    dte_range: Optional[Tuple[float, float]] = None,
    moneyness_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Form long/short trades per timestamp using cross-sectional predictions.
    Orchestrates: preprocess -> liquidity -> contract filters -> per-bucket slice/aggregate -> pick legs -> emit trades.
    """
    df = _preprocess_preds(preds, group_freq)
    df = _compute_rolling_trades(df, roll_by_rows, roll_bars, roll_window)

    if min_roll_trades and min_roll_trades > 0:
        df = df[df["roll_trades"] >= float(min_roll_trades)]

    df = _apply_contract_filters(df, strike, strike_tol, expiry)

    trades: List[dict] = []
    for ts, grp in df.groupby("ts_bucket"):
        grp = grp.dropna(subset=["y_pred", "y_true"]).copy()
        if grp.empty:
            continue

        # build the “universe” we rank within, based on granularity
        universe = _build_universe_for_ranking(
            grp,
            granularity=granularity,
            weighting_scheme=weighting_scheme,
            dte_range=dte_range,
            moneyness_range=moneyness_range,
        )
        if universe is None or len(universe) < max(2, 2 * top_k):
            continue

        long_leg, short_leg = _pick_legs(universe, top_k=top_k, pair_only=pair_only)
        if long_leg is None or short_leg is None:
            continue

        pred_edge = float(long_leg["y_pred"].mean() - short_leg["y_pred"].mean())
        if abs(pred_edge) < threshold:
            continue

        realized_edge = float(long_leg["y_true"].mean() - short_leg["y_true"].mean())
        edge_1m, edge_15m, edge_60m = _compute_edges(grp, long_leg, short_leg)

        strike_out, exp_out = _resolve_contract_identity(grp)

        trades.append(_build_trade_row(
            ts=ts,
            long_leg=long_leg,
            short_leg=short_leg,
            pred_edge=pred_edge,
            realized_edge=realized_edge,
            edge_1m=edge_1m,
            edge_15m=edge_15m,
            edge_60m=edge_60m,
            strike_out=strike_out,
            exp_out=exp_out,
            n_universe=int(len(grp)),
        ))

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
        ps = pd.to_numeric(trades["pred_edge"], errors="coerce")
        m = s.notna() & ps.notna() & (ps != 0)
        if not m.any():
            return float("nan")
        return float((np.sign(s[m]) == np.sign(ps[m])).mean())

    return {
        "n_trades": int(len(trades)),
        "avg_edge_1m": _avg("edge_1m"),
        "avg_edge_15m": _avg("edge_15m"),
        "avg_edge_60m": _avg("edge_60m"),
        "hit_rate_1m": _hit("edge_1m"),
        "hit_rate_15m": _hit("edge_15m"),
        "hit_rate_60m": _hit("edge_60m"),
    }


# ===========================
# Helpers — preprocessing
# ===========================
def _preprocess_preds(preds: pd.DataFrame, group_freq: str) -> pd.DataFrame:
    """Ensure required columns exist, clean types, and add ts_bucket."""
    if "ts_event" not in preds.columns:
        preds = preds.copy()
        preds["ts_event"] = pd.NaT
    if "symbol" not in preds.columns:
        preds = preds.copy()
        preds["symbol"] = None

    df = preds.dropna(subset=["ts_event", "y_pred", "y_true"]).copy()

    # opt_volume present? ensure numeric for later rolling/liquidity
    if "opt_volume" not in df.columns:
        df["opt_volume"] = 0.0
    else:
        df["opt_volume"] = pd.to_numeric(df["opt_volume"], errors="coerce").fillna(0.0)

    # standardize datetime + bucket
    ts_dt = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.loc[ts_dt.notna()].copy()
    df["ts_event"] = ts_dt.loc[ts_dt.notna()]
    df["ts_bucket"] = df["ts_event"].dt.floor(group_freq)
    return df


def _compute_rolling_trades(
    df: pd.DataFrame,
    roll_by_rows: bool,
    roll_bars: Optional[int],
    roll_window: str
) -> pd.DataFrame:
    """Compute rolling net trades/volume per symbol."""
    df = df.sort_values(["symbol", "ts_event"]).reset_index(drop=True)

    try:
        if roll_by_rows and (roll_bars is not None and roll_bars > 0):
            df["roll_trades"] = (
                df.groupby("symbol", group_keys=False)["opt_volume"]
                  .rolling(int(roll_bars), min_periods=1).sum()
                  .reset_index(level=0, drop=True)
            )
        else:
            def _roll(g: pd.DataFrame) -> pd.Series:
                g2 = g.set_index("ts_event")
                return g2["opt_volume"].rolling(roll_window).sum()
            df["roll_trades"] = (
                df.groupby("symbol", group_keys=False).apply(_roll).reset_index(level=0, drop=True)
            )
    except Exception:
        df["roll_trades"] = np.nan
    return df


def _apply_contract_filters(
    df: pd.DataFrame,
    strike: Optional[float],
    strike_tol: float,
    expiry: Optional[str],
) -> pd.DataFrame:
    """Apply optional strike and expiry filters."""
    out = df
    if strike is not None and "strike_price" in out.columns:
        tol = abs(float(strike_tol)) if strike_tol is not None else 0.0
        sp = pd.to_numeric(out["strike_price"], errors="coerce")
        if tol > 0:
            out = out[np.abs(sp - float(strike)) <= tol]
        else:
            out = out[sp == float(strike)]

    if expiry is not None and "expiry" in out.columns:
        try:
            exp_ts = pd.to_datetime(expiry, utc=True).normalize()
            df_exp = pd.to_datetime(out["expiry"], utc=True, errors="coerce").dt.normalize()
            out = out[df_exp == exp_ts]
        except Exception:
            pass
    return out


# ===========================
# Helpers — weighting & universe
# ===========================
def _choose_weights(df: pd.DataFrame, weighting_scheme: str) -> pd.Series:
    if weighting_scheme == "equal":
        return pd.Series(1.0, index=df.index)
    if weighting_scheme == "vega" and "vega" in df.columns:
        return pd.to_numeric(df["vega"], errors="coerce").fillna(0.0)
    if weighting_scheme == "vega_x_liquidity" and "vega" in df.columns and "opt_volume" in df.columns:
        vega = pd.to_numeric(df["vega"], errors="coerce").fillna(0.0)
        liq  = pd.to_numeric(df["opt_volume"], errors="coerce").fillna(0.0)
        return vega * np.sqrt(liq + 1.0)
    # default: opt_volume
    return pd.to_numeric(df.get("opt_volume", 0.0), errors="coerce").fillna(0.0)


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    tot = w.sum()
    return float((v * w).sum() / tot) if tot > 0 else float(v.mean())


def _apply_slice_filters(
    grp: pd.DataFrame,
    dte_range: Optional[Tuple[float, float]],
    moneyness_range: Optional[Tuple[float, float]],
) -> pd.DataFrame:
    """Apply optional DTE and moneyness filters (for 'slice' and 'surface')."""
    out = grp
    if dte_range and "dte" in out.columns:
        lo, hi = map(float, dte_range)
        out = out[(out["dte"] >= lo) & (out["dte"] <= hi)]
    if moneyness_range and "log_moneyness" in out.columns:
        lo, hi = map(float, moneyness_range)
        lm = pd.to_numeric(out["log_moneyness"], errors="coerce")
        out = out[(lm >= lo) & (lm <= hi)]
    return out


def _aggregate_by_symbol_with_weights(grp: pd.DataFrame, weighting_scheme: str) -> pd.DataFrame:
    """Collapse to one row per symbol with weighted means."""
    if grp.empty:
        return grp
    w = _choose_weights(grp, weighting_scheme=weighting_scheme)
    grp = grp.assign(_w=w.values)

    def _wm(series: pd.Series) -> float:
        return _wmean(series, grp.loc[series.index, "_w"])

    agg: Dict[str, object] = {"y_pred": _wm, "y_true": _wm}
    for c in ("edge_1m", "edge_15m", "edge_60m"):
        if c in grp.columns:
            agg[c] = (lambda s, col=c: _wmean(grp[col].loc[s.index], grp.loc[s.index, "_w"]))
    if "expiry" in grp.columns:
        agg["expiry"] = "first"
    if "strike_price" in grp.columns:
        agg["strike_price"] = "median"

    return grp.groupby("symbol", as_index=False).agg(agg)


def _build_universe_for_ranking(
    grp: pd.DataFrame,
    granularity: str,
    weighting_scheme: str,
    dte_range: Optional[Tuple[float, float]],
    moneyness_range: Optional[Tuple[float, float]],
) -> Optional[pd.DataFrame]:
    """Return the per-timestamp 'universe' to rank (contract | slice | surface)."""
    if granularity not in ("contract", "slice", "surface"):
        # fallback: behave like slice
        granularity = "slice"

    # Apply slice filters to both "slice" and "surface" if provided
    g = grp.copy()
    if granularity in ("slice", "surface"):
        g = _apply_slice_filters(g, dte_range=dte_range, moneyness_range=moneyness_range)
        if g.empty:
            return None

    if granularity == "contract":
        return g  # no aggregation

    # slice/surface: aggregate to one row per symbol
    g_sym = _aggregate_by_symbol_with_weights(g, weighting_scheme=weighting_scheme)
    return g_sym if not g_sym.empty else None


# ===========================
# Helpers — legs, edges, rows
# ===========================
def _pick_legs(universe: pd.DataFrame, top_k: int, pair_only: bool) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[None, None]:
    """Pick long/short legs given a universe already filtered/aggregated."""
    if universe is None or universe.empty:
        return None, None

    U = universe.sort_values("y_pred", ascending=False)
    if pair_only:
        if U["symbol"].nunique() < 2:
            return None, None
        long_leg = U.head(1)
        short_leg = U.tail(1)
        return long_leg, short_leg

    if len(U) < max(2, 2 * top_k):
        return None, None
    long_leg = U.head(top_k)
    short_leg = U.tail(top_k)
    return long_leg, short_leg


def _compute_edges(grp: pd.DataFrame, long_leg: pd.DataFrame, short_leg: pd.DataFrame) -> Tuple[float, float, float]:
    """Compute edge edges for available horizons."""
    def _edge(col: str) -> float:
        if col not in grp.columns:
            return float("nan")
        return float(pd.to_numeric(long_leg[col], errors="coerce").mean() -
                     pd.to_numeric(short_leg[col], errors="coerce").mean())

    edge_1m = _edge("edge_1m")
    edge_15m = _edge("edge_15m")
    edge_60m = _edge("edge_60m")
    return edge_1m, edge_15m, edge_60m


def _resolve_contract_identity(grp: pd.DataFrame) -> Tuple[str, str]:
    """Resolve a single strike/expiry identity for logging if unique within the group."""
    exp_out = ""
    if "expiry" in grp.columns:
        euniq = pd.to_datetime(grp["expiry"], errors="coerce").dropna().dt.normalize().unique()
        if len(euniq) == 1:
            exp_out = pd.to_datetime(euniq[0]).date().isoformat()

    strike_out = ""
    if "strike_price" in grp.columns:
        sunique = pd.to_numeric(grp["strike_price"], errors="coerce").dropna().unique()
        if len(sunique) == 1:
            strike_out = float(sunique[0])
    return strike_out, exp_out


def _build_trade_row(
    ts,
    long_leg: pd.DataFrame,
    short_leg: pd.DataFrame,
    pred_edge: float,
    realized_edge: float,
    edge_1m: float,
    edge_15m: float,
    edge_60m: float,
    strike_out,
    exp_out,
    n_universe: int,
) -> dict:
    return {
        "ts_event": ts,
        "long_symbols": ",".join(map(str, long_leg["symbol"].tolist())),
        "short_symbols": ",".join(map(str, short_leg["symbol"].tolist())),
        "strike_price": strike_out,
        "expiry": exp_out,
        "pred_edge": pred_edge,
        "realized_edge": realized_edge,
        "edge_1m": edge_1m,
        "edge_15m": edge_15m,
        "edge_60m": edge_60m,
        "correct": int(np.sign(pred_edge) == np.sign(realized_edge) and realized_edge != 0.0),
        "n_universe": n_universe,
    }
