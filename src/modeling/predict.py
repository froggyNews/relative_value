
from __future__ import annotations

from typing import Optional, Sequence, List
import numpy as np
import pandas as pd
import xgboost as xgb

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

def make_prediction_frame(
    model: xgb.XGBRegressor,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    ts_te: Optional[pd.Series] = None,
    strike_te: Optional[pd.Series] = None,
    expiry_te: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return a tidy inference DataFrame (ts_event, y_true, y_pred, symbol, strike, expiry, opt_volume, edge_*)."""
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
