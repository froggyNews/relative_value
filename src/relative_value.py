"""
Relative Value Backtest using pooled IV-returns prediction model

Workflow
- Build pooled dataset with peer features for a set of tickers
- Train pooled XGB model on early portion of data (time-respecting split)
- Predict on the held-out test portion
- For each timestamp, form long/short portfolios using cross-sectional y_pred
- Compute realized spread using forward IV returns (y_true)
- Report hit-rate, average spread, Sharpe, and cumulative PnL

CLI
python src/relative_value.py --config configs/test1.json
python -m src.relative_value \
  --tickers QBTS IONQ RGTI QUBT \
  --start 2025-06-02 --end 2025-08-06 \
  --forward-steps 1 --test-frac 0.2 --top-k 1 --threshold 0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Optional

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering import build_pooled_iv_return_dataset_time_safe
from data.data_loader_coordinator import load_cores_with_auto_fetch
from train_iv_returns import train_xgb_iv_returns_time_safe_pooled


# -----------------------------
# Helpers
# -----------------------------

def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    non_num = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_num:
        df = df.drop(columns=non_num)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(float)
    return df


def _expected_feature_names(model: xgb.XGBRegressor) -> List[str]:
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else []


def _looks_like_generic_xgb_names(names: Sequence[str]) -> bool:
    names = list(names)
    return bool(names) and all(n.startswith("f") and n[1:].isdigit() for n in names)


def _align_columns_to_model(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    expected = _expected_feature_names(model)
    if not expected:
        return X
    if _looks_like_generic_xgb_names(expected):
        # Avoid reindexing when feature names are generic
        return X
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X.reindex(columns=list(expected))


def _extract_symbol_from_dummies(row: pd.Series) -> Optional[str]:
    sym_cols = [c for c in row.index if c.startswith("sym_")]
    if not sym_cols:
        return None
    if len(sym_cols) == 1:
        return sym_cols[0][4:]
    # Choose the dummy with max activation
    vals = row[sym_cols]
    if vals.isna().all():
        return None
    chosen = vals.idxmax()
    if pd.isna(vals[chosen]) or float(vals[chosen]) <= 0:
        return None
    return chosen[4:]


# -----------------------------
# Core logic
# -----------------------------

@dataclass
class BacktestConfig:
    tickers: Sequence[str]
    start: str
    end: str
    forward_steps: int = 1
    test_frac: float = 0.2
    tolerance: str = "15s"
    r: float = 0.045
    db_path: Path | str = Path("data/iv_data_1m.db")
    top_k: int = 1
    threshold: float = 0.0  # min predicted spread to take a trade
    save_trades_csv: Optional[Path] = None
    auto_fetch: bool = True
    atm_only: bool = True
    # Liquidity filter (rolling net trades/volume over window)
    min_roll_trades: float = 0.0
    roll_window: str = "30min"
    roll_by_rows: bool = True
    roll_bars: Optional[int] = 30
    volume_col: Optional[str] = None
    group_freq: str = "1min"  # cross-sectional grouping bucket
    # Pair-only and contract constraints
    pair_only: bool = False  # if True, trade exactly two different tickers (top vs bottom)
    strike: Optional[float] = None
    expiry: Optional[str] = None  # YYYY-MM-DD
    strike_tol: float = 0.0


def build_and_split_pooled(cfg: BacktestConfig) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series
]:
    # Load cores the same way as rolling_surface_eval (ensures identical fetch path/timeframe detection)
    cores = load_cores_with_auto_fetch(
        list(cfg.tickers),
        pd.Timestamp(cfg.start, tz="UTC"),
        pd.Timestamp(cfg.end, tz="UTC"),
        Path(cfg.db_path),
        auto_fetch=cfg.auto_fetch,
        atm_only=cfg.atm_only,
    )

    pooled = build_pooled_iv_return_dataset_time_safe(
        tickers=list(cfg.tickers),
        start=pd.Timestamp(cfg.start, tz="UTC"),
        end=pd.Timestamp(cfg.end, tz="UTC"),
        r=cfg.r,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        db_path=cfg.db_path,
        cores=cores,
    )
    # Ensure clean, global time order and a leakage-free split
    if "ts_event" not in pooled.columns:
        raise KeyError("ts_event missing from pooled dataset")

    pooled["ts_event"] = pd.to_datetime(pooled["ts_event"], utc=True, errors="coerce")
    pooled = pooled.dropna(subset=["ts_event"]).sort_values(["ts_event"]).reset_index(drop=True)

    # Choose a split boundary on unique timestamps (time-respecting)
    unique_ts = pooled["ts_event"].unique()
    if len(unique_ts) < 2:
        raise ValueError("Insufficient distinct timestamps for a time-respecting split")

    # Use the requested test_frac to pick a boundary in time (not rows)
    ts_cut_idx = max(1, int(len(unique_ts) * (1 - cfg.test_frac)))
    ts_cut = unique_ts[ts_cut_idx - 1]

    # All rows <= ts_cut → train; > ts_cut → test (strict separation)
    train_mask = pooled["ts_event"] <= ts_cut
    test_mask  = pooled["ts_event"] >  ts_cut

    if train_mask.sum() < 10 or test_mask.sum() < 10:
        raise ValueError(
            f"Insufficient data after time split. "
            f"train={train_mask.sum()} test={test_mask.sum()} (cut={ts_cut})"
        )

    pooled_tr = pooled.loc[train_mask].copy()
    pooled_te = pooled.loc[test_mask].copy()

    if pooled is None or pooled.empty:
        raise ValueError("Pooled dataset came back empty.")

    # Target and features
    if "iv_ret_fwd" not in pooled.columns:
        raise KeyError("iv_ret_fwd missing from pooled dataset")

    # Preserve auxiliary columns before dropping non-numeric
    ts = pd.to_datetime(pooled.get("ts_event"), utc=True, errors="coerce") if "ts_event" in pooled.columns else pd.Series(pd.NaT, index=pooled.index)
    strike = pd.to_numeric(pooled.get("core_strike_price"), errors="coerce") if "core_strike_price" in pooled.columns else pd.Series(np.nan, index=pooled.index)
    expiry = (
        pd.to_datetime(pooled.get("core_expiry_date"), utc=True, errors="coerce")
        if "core_expiry_date" in pooled.columns
        else (pd.to_datetime(pooled.get("expiry_date"), utc=True, errors="coerce") if "expiry_date" in pooled.columns else pd.Series(pd.NaT, index=pooled.index))
    )

    y = pooled["iv_ret_fwd"].astype(float)
    X = pooled.drop(columns=[c for c in ["iv_ret_fwd", "iv_ret_fwd_abs", "core_iv_ret_fwd_abs"] if c in pooled.columns])
    X = _ensure_numeric(X)

    # Enforce strict time ordering and boundary-based split to avoid leakage
    ts_ord = pd.to_datetime(ts, utc=True, errors="coerce")
    order_idx = ts_ord.sort_values(kind="mergesort").index
    X = X.loc[order_idx].reset_index(drop=True)
    y = y.loc[order_idx].reset_index(drop=True)
    ts = ts.loc[order_idx].reset_index(drop=True)
    strike = strike.loc[order_idx].reset_index(drop=True)
    expiry = expiry.loc[order_idx].reset_index(drop=True)

    n = len(X)
    split = int(n * (1 - cfg.test_frac))
    if split < 10 or (n - split) < 10:
        raise ValueError(f"Insufficient data for split. n={n}, split={split}")

    # Define a time boundary and split on timestamps (train <= bound, test > bound)
    t_bound = pd.to_datetime(ts.iloc[max(split - 1, 0)], utc=True, errors="coerce")
    tser = pd.to_datetime(ts, utc=True, errors="coerce")
    train_mask = tser <= t_bound
    test_mask = tser > t_bound

    X_tr, X_te = X.loc[train_mask].copy(), X.loc[test_mask].copy()
    y_tr, y_te = y.loc[train_mask].copy(), y.loc[test_mask].copy()
    ts_tr, ts_te = ts.loc[train_mask].copy(), ts.loc[test_mask].copy()
    strike_tr, strike_te = strike.loc[train_mask].copy(), strike.loc[test_mask].copy()
    expiry_tr, expiry_te = expiry.loc[train_mask].copy(), expiry.loc[test_mask].copy()
    return X_tr, X_te, y_tr, y_te, ts_tr, ts_te, strike_tr, strike_te, expiry_tr, expiry_te


def train_pooled_model_on_split(X_tr: pd.DataFrame, y_tr: pd.Series) -> xgb.XGBRegressor:
    # Use same defaults as train_xgb_iv_returns_time_safe_pooled
    params = dict(objective="reg:squarederror",
                  n_estimators=350, learning_rate=0.05,
                  max_depth=6, subsample=0.9, colsample_bytree=0.9,
                  random_state=42)
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    return model


def make_prediction_frame(
    model: xgb.XGBRegressor,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    ts_te: Optional[pd.Series] = None,
    strike_te: Optional[pd.Series] = None,
    expiry_te: Optional[pd.Series] = None,
) -> pd.DataFrame:
    X_aligned = _align_columns_to_model(model, X_te.copy())
    y_pred = model.predict(X_aligned)
    out = pd.DataFrame({
        "ts_event": (pd.to_datetime(ts_te, utc=True, errors="coerce").values if ts_te is not None else (X_te.get("ts_event").values if "ts_event" in X_te.columns else pd.NaT)),
        "y_true": y_te.values,
        "y_pred": y_pred,
    }, index=X_te.index)

    # Recover symbol from one-hot dummies
    sym_cols = [c for c in X_te.columns if c.startswith("sym_")]
    if sym_cols:
        sym_series = X_te[sym_cols].apply(_extract_symbol_from_dummies, axis=1)
        out["symbol"] = sym_series.values
    else:
        out["symbol"] = None

    # Attach strike and expiry if provided
    if strike_te is not None:
        out["strike"] = pd.to_numeric(strike_te, errors="coerce").values
    else:
        out["strike"] = np.nan
    if expiry_te is not None:
        out["expiry"] = pd.to_datetime(expiry_te, utc=True, errors="coerce").values
    else:
        out["expiry"] = pd.NaT

    # Carry through per-row option volume if available (used for rolling trades filter)
    vol_col = None
    for c in ("core_opt_volume", "opt_volume"):
        if c in X_te.columns:
            vol_col = c
            break
    if vol_col is not None:
        out["opt_volume"] = pd.to_numeric(X_te[vol_col], errors="coerce").fillna(0.0)
    else:
        out["opt_volume"] = 0.0

    # Compute theoretical edges at multiple horizons from panel IV levels
    # Requires panel_IV_{ticker} columns in X_te and symbol per row
    try:
        out["iv_level"] = np.nan
        if "symbol" in out.columns:
            for s in pd.Series(out["symbol"].unique()).dropna().astype(str):
                col = f"panel_IV_{s}"
                if col in X_te.columns:
                    mask = out["symbol"].astype(str) == s
                    out.loc[mask, "iv_level"] = pd.to_numeric(X_te.loc[mask, col], errors="coerce").values
        # Group by symbol to compute forward log returns at 1, 15, 60 steps (minutes)
        def _fwd_edges(v: pd.Series, h: int) -> pd.Series:
            vv = pd.to_numeric(v, errors="coerce").astype(float)
            # Avoid log of non-positive values
            vv = vv.where(vv > 0, np.nan)
            return np.log(vv.shift(-h)) - np.log(vv)
        out = out.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
        out["edge_1m"] = out.groupby("symbol", group_keys=False)["iv_level"].apply(lambda s: _fwd_edges(s, 1))
        out["edge_15m"] = out.groupby("symbol", group_keys=False)["iv_level"].apply(lambda s: _fwd_edges(s, 15))
        out["edge_60m"] = out.groupby("symbol", group_keys=False)["iv_level"].apply(lambda s: _fwd_edges(s, 60))
    except Exception:
        out["edge_1m"] = np.nan
        out["edge_15m"] = np.nan
        out["edge_60m"] = np.nan

    return out.reset_index(drop=True)


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
    """Form long/short trades per timestamp using cross-sectional predictions.

    For each timestamp, go long the top_k highest y_pred tickers and short the
    top_k lowest y_pred tickers. Realized spread uses y_true (forward return).
    """
    # Ensure clean input
    # Ensure ts_event column exists to avoid KeyError in sorting/rolling
    if "ts_event" not in preds.columns:
        preds = preds.copy()
        preds["ts_event"] = pd.NaT
    df = preds.dropna(subset=["ts_event", "y_pred", "y_true"]).copy()
    if "symbol" not in df.columns:
        df["symbol"] = None

    # Compute rolling net trades/volume per symbol over the specified window
    try:
        if "opt_volume" not in df.columns:
            df["opt_volume"] = 0.0
        if roll_by_rows and (roll_bars is not None and roll_bars > 0):
            # Row-based rolling: last N rows per symbol
            df = df.sort_values(["symbol", "ts_event"]).reset_index(drop=True)
            df["roll_trades"] = (
                df.groupby("symbol", group_keys=False)["opt_volume"]
                  .rolling(int(roll_bars), min_periods=1).sum()
                  .reset_index(level=0, drop=True)
            )
        else:
            # Time-based rolling (requires ts_event)
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

    # Apply min rolling trades/volume filter if requested
    if min_roll_trades and min_roll_trades > 0:
        df = df[df["roll_trades"] >= float(min_roll_trades)]

    # No explicit pair filter when pair_only is boolean; we will enforce two-name trades later

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

    # Cross-sectional grouping key: bucket timestamps to a common bar
    if "ts_event" in df.columns:
        ts_dt = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df["ts_bucket"] = ts_dt.dt.floor(group_freq)
        key_col = "ts_bucket"
    else:
        key_col = "ts_event"

    trades = []
    for ts, grp in df.groupby(key_col):
        grp = grp.dropna(subset=["y_pred", "y_true"]).copy()
        # Collapse to one row per symbol to avoid duplicate symbols in legs
        if not grp.empty and "symbol" in grp.columns:
            agg_cols = {"y_pred": "mean", "y_true": "mean"}
            for c in ("edge_1m", "edge_15m", "edge_60m"):
                if c in grp.columns:
                    agg_cols[c] = "mean"
            grp_sym = grp.groupby("symbol", as_index=False).agg(agg_cols)
        else:
            grp_sym = grp.copy()

        if pair_only:
            # Enforce a two-name trade: top vs bottom
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
        # Theoretical edges at multiple horizons using IV levels
        def _spread(col: str) -> float:
            if col not in grp.columns:
                return float("nan")
            return float(long_leg[col].mean() - short_leg[col].mean())
        realized_spread = float(long_leg["y_true"].mean() - short_leg["y_true"].mean())
        edge_1m = _spread("edge_1m")
        edge_15m = _spread("edge_15m")
        edge_60m = _spread("edge_60m")
        # Resolve a single expiry if the group has a unique non-null expiry; else blank
        exp_out = ""
        if "expiry" in grp.columns:
            euniq = pd.to_datetime(grp["expiry"], errors="coerce").dropna().dt.normalize().unique()
            if len(euniq) == 1:
                exp_out = pd.to_datetime(euniq[0]).date().isoformat()

        # Resolve strike if unique; else blank
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
    # Theoretical edge metrics
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


def run_backtest(cfg: BacktestConfig) -> Dict[str, object]:
    # Build + split
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, strike_tr, strike_te, expiry_tr, expiry_te = build_and_split_pooled(cfg)

    # Train pooled model (time-respecting)
    model = train_pooled_model_on_split(X_tr, y_tr)

    # Predictions
    preds = make_prediction_frame(model, X_te, y_te, ts_te, strike_te, expiry_te)

    # Trades
    trades = make_relative_value_trades(
        preds,
        top_k=cfg.top_k,
        threshold=cfg.threshold,
        min_roll_trades=cfg.min_roll_trades,
        roll_window=cfg.roll_window,
        roll_by_rows=cfg.roll_by_rows,
        roll_bars=cfg.roll_bars,
        pair_only=cfg.pair_only,
        strike=cfg.strike,
        expiry=cfg.expiry,
        strike_tol=cfg.strike_tol,
        group_freq=cfg.group_freq,
    )
    summary = summarize_trades(trades)

    # Optional save
    if cfg.save_trades_csv:
        out_path = Path(cfg.save_trades_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out_path, index=False)

    return {
        "summary": summary,
        "trades": trades,
        "preds_head": preds.head(5),
    }


# -----------------------------
# Config I/O and CLI
# -----------------------------

def _load_json_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        import json
        return json.load(f)


def _config_to_backtest(cfg: dict) -> BacktestConfig:
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    liq = cfg.get("liquidity", {})
    trading = cfg.get("trading", {})
    output = cfg.get("output", {})

    return BacktestConfig(
        tickers=data.get("tickers", ["QBTS", "IONQ", "RGTI", "QUBT"]),
        start=data.get("start", "2025-06-02"),
        end=data.get("end", "2025-06-16"),
        db_path=Path(data.get("db_path", "data/iv_data_1m.db")),
        auto_fetch=bool(data.get("auto_fetch", True)),
        atm_only=(str(data.get("surface_mode", "atm")).lower() == "atm"),

        forward_steps=int(model.get("forward_steps", 1)),
        test_frac=float(model.get("test_frac", 0.2)),
        tolerance=str(model.get("tolerance", "15s")),
        r=float(model.get("r", 0.045)),

        min_roll_trades=float(liq.get("min_roll_trades", 0.0)),
        roll_by_rows=bool(liq.get("roll_by_rows", True)),
        roll_bars=int(liq.get("roll_bars", 30)) if liq.get("roll_bars", 30) is not None else None,
        roll_window=str(liq.get("roll_window", "30min")),
        volume_col=liq.get("volume_col", None),

        top_k=int(trading.get("top_k", 1)),
        threshold=float(trading.get("threshold", 0.0)),
        pair_only=bool(trading.get("pair_only", False)),
        strike=trading.get("strike", None),
        expiry=trading.get("expiry", None),
        strike_tol=float(trading.get("strike_tol", 0.0)),
        group_freq=str(trading.get("group_freq", "1min")),

        save_trades_csv=Path(output.get("save_trades_csv")) if output.get("save_trades_csv") else None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relative value backtest using pooled IV-returns model.")
    # Config-based usage
    p.add_argument("--config", type=str, default="", help="Path to JSON config file.")
    p.add_argument("--write-config-template", type=str, default="", help="Write a config template JSON to this path and exit.")
    # Legacy/explicit flags (kept for compatibility and tests)
    p.add_argument("--tickers", nargs="+", default=["QBTS", "IONQ", "RGTI", "QUBT"], help="Universe tickers")
    p.add_argument("--start", required=False, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=False, help="End date YYYY-MM-DD")
    p.add_argument("--forward-steps", type=int, default=1, help="Forward horizon in steps")
    p.add_argument("--test-frac", type=float, default=0.2, help="Test fraction for time split")
    p.add_argument("--tolerance", default="15s", help="As-of merge tolerance for panel")
    p.add_argument("--r", type=float, default=0.045, help="Risk-free rate for Greeks/IV calc")
    p.add_argument("--db", type=str, default="data/iv_data_1m.db", help="SQLite DB path")
    p.add_argument("--top-k", type=int, default=1, help="Number of names in each leg")
    p.add_argument("--threshold", type=float, default=0.0, help="Min predicted spread to trade")
    p.add_argument("--save-trades", type=str, default="", help="Optional CSV path to save trades")
    p.add_argument("--no-fetch", action="store_true", help="Disable auto-fetch; require data to already exist")
    p.add_argument("--surface-mode", choices=["atm", "full"], default="atm", help="Use ATM-only or full surface cores")
    # Liquidity filter options
    p.add_argument("--min-roll-trades", type=float, default=0.0, help="Minimum rolling net trades/volume over window")
    p.add_argument("--roll-window", type=str, default="30min", help="Time-based rolling window (e.g., '30min')")
    p.add_argument("--time-rolling", action="store_true", help="Use time-based rolling (default is row-based)")
    p.add_argument("--roll-bars", type=int, default=30, help="Row-based rolling bars")
    p.add_argument("--volume-col", type=str, default=None, help="Preferred volume column to use if available")
    # Pair/contract constraints
    p.add_argument("--pair-only", action="store_true", help="Only trade two different tickers (top vs bottom)")
    p.add_argument("--strike", type=float, default=None, help="Restrict to a specific strike")
    p.add_argument("--expiry", type=str, default=None, help="Restrict to a specific expiry YYYY-MM-DD")
    p.add_argument("--strike-tol", type=float, default=0.0, help="Tolerance for strike matching (abs)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Write template and exit
    if args.write_config_template:
        from pathlib import Path as _P
        outp = _P(args.write_config_template)
        outp.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        template = {
            "data": {
                "tickers": ["QBTS", "IONQ", "RGTI", "QUBT"],
                "start": "2025-06-02",
                "end": "2025-06-16",
                "db_path": "data/iv_data_1m.db",
                "auto_fetch": True,
                "surface_mode": "atm"
            },
            "model": {
                "forward_steps": 1,
                "test_frac": 0.2,
                "tolerance": "15s",
                "r": 0.045
            },
            "liquidity": {
                "min_roll_trades": 0.0,
                "roll_by_rows": True,
                "roll_bars": 30,
                "roll_window": "30min",
                "volume_col": None
            },
            "trading": {
                "top_k": 1,
                "threshold": 0.0,
                "pair_only": False,
                "strike": None,
                "expiry": None,
                "strike_tol": 0.0,
                "group_freq": "1min"
            },
            "output": {
                "save_trades_csv": "outputs/relative_value_trades.csv"
            }
        }
        with open(outp, "w", encoding="utf-8") as f:
            _json.dump(template, f, indent=2)
        print(f"[TEMPLATE] Wrote config template to {outp}")
        raise SystemExit(0)

    # Load config or fall back to minimal overrides
    if args.config:
        cfg_dict = _load_json_config(Path(args.config))
        cfg = _config_to_backtest(cfg_dict)
        if args.start:
            cfg.start = args.start
        if args.end:
            cfg.end = args.end
        if args.save_trades:
            cfg.save_trades_csv = Path(args.save_trades)
    else:
        # Minimal inline default if no config is provided
        cfg = BacktestConfig(
            tickers=["QBTS", "IONQ", "RGTI", "QUBT"],
            start=args.start or "2025-06-02",
            end=args.end or "2025-06-16",
        )
        if args.save_trades:
            cfg.save_trades_csv = Path(args.save_trades)

    results = run_backtest(cfg)
    print("Summary:")
    print(results["summary"]) 
