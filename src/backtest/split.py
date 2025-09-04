
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from src.modeling.schemas import BacktestConfig
from src.feature_engineering import build_pooled_iv_return_dataset_time_safe
from src.data.data_loader_coordinator import load_cores_with_auto_fetch

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
    # enforce numeric-only
    non_num = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_num:
        X = X.drop(columns=non_num)
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(float)

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
