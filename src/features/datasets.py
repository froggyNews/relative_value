import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Sequence, Optional
from .finalize import finalize_dataset, _normalize_numeric_features
from .engineering import add_all_features
from .constants import HIDE_COLUMNS, DEFAULT_DB_PATH



def build_iv_panel(
    cores: Dict[str, pd.DataFrame],
    tolerance: str = "15s",
    agg: str = "median",
) -> pd.DataFrame:
    """Build a per-timestamp IV panel for all tickers.

    For each ticker, aggregate IV across the available surface observations at
    each timestamp (median), then compute a composite IV return series. This
    ensures downstream rolling correlations operate over the whole surface,
    not just a single ATM slice.
    """
    tol = pd.Timedelta(tolerance)
    iv_wide = None

    for ticker, df in cores.items():
        if df is None or df.empty or not {"ts_event", "iv_clip"}.issubset(df.columns):
            continue

        # Robust per-timestamp composite IV across the surface
        tmp = df[["ts_event", "iv_clip"]].copy()
        tmp["ts_event"] = pd.to_datetime(tmp["ts_event"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["ts_event", "iv_clip"]) \
                 .groupby("ts_event", as_index=False)["iv_clip"].agg(
                     "median" if str(agg).lower() == "median" else "mean"
                 ) \
                 .rename(columns={"iv_clip": f"IV_{ticker}"}) \
                 .sort_values("ts_event")

        # Composite IV returns for the ticker
        tmp[f"IVRET_{ticker}"] = (
            np.log(tmp[f"IV_{ticker}"]) - np.log(tmp[f"IV_{ticker}"].shift(1))
        )
        tmp = tmp[["ts_event", f"IV_{ticker}", f"IVRET_{ticker}"]]

        if iv_wide is None:
            iv_wide = tmp
        else:
            iv_wide = pd.merge_asof(
                iv_wide.sort_values("ts_event"),
                tmp.sort_values("ts_event"),
                on="ts_event",
                direction="backward",
                tolerance=tol,
            )

    return iv_wide if iv_wide is not None else pd.DataFrame(columns=["ts_event"])



def build_pooled_iv_return_dataset_time_safe(
    tickers: Sequence[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "15s",
    db_path: Path | str | None = None,
    cores: Optional[Dict[str, pd.DataFrame]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Build pooled dataset for forecasting forward IV return, keeping peer IV/IVRET columns."""
    
    if debug:
        print(f"DEBUG: Building pooled dataset for {len(tickers)} tickers")
        print(f"DEBUG: Parameters - forward_steps: {forward_steps}, tolerance: {tolerance}")
    
    # If cores not provided, need to load them (fallback for backward compatibility)
    if cores is None:
        from data.data_loader_coordinator import load_cores_with_auto_fetch
        cores = load_cores_with_auto_fetch(tickers, start, end, Path(db_path) if db_path is not None else Path("data/iv_data_1h.db"))
    
    if debug:
        print(f"DEBUG: Cores loaded for tickers: {list(cores.keys())}")
        for ticker, core in cores.items():
            print(f"DEBUG: {ticker} core shape: {core.shape if core is not None else 'None'}")
    
    # Build panel (contains IV and IVRET columns for all tickers)
    panel = build_iv_panel(cores, tolerance=tolerance)
    if panel is not None and not panel.empty:
        panel = panel.rename(
            columns={c: (f"panel_{c}" if c != "ts_event" else c) for c in panel.columns}
        )
    
    if debug:
        print(f"DEBUG: IV panel shape: {panel.shape}")
        if not panel.empty:
            debug_dir = Path("debug_snapshots")
            debug_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            panel_path = debug_dir / f"iv_panel_{timestamp}.csv"
            panel.to_csv(panel_path, index=False)
            print(f"DEBUG: Saved IV panel to {panel_path}")
    
    frames = []
    for ticker in tickers:
        if ticker not in cores:
            if debug:
                print(f"DEBUG: Skipping {ticker} - not in cores")
            continue
            
        if debug:
            print(f"DEBUG: Processing {ticker}")
            
        # Add features and prefix with source label
        feats = add_all_features(cores[ticker], forward_steps=forward_steps, r=r)
        rename_map = {
            c: f"core_{c}" for c in feats.columns
            if c not in {"ts_event", "iv_ret_fwd", "iv_ret_fwd_abs", "symbol", "iv_clip"}
        }
        feats = feats.rename(columns=rename_map)
        if "iv_ret_fwd_abs" in feats.columns:
            feats = feats.rename(columns={"iv_ret_fwd_abs": "core_iv_ret_fwd_abs"})

        if debug:
            print(f"DEBUG: {ticker} after add_all_features: {feats.shape}")

        # Merge with panel (keeps all IV/IVRET columns for all tickers)
        feats = pd.merge_asof(
            feats.sort_values("ts_event"), panel.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
        
        if debug:
            print(f"DEBUG: {ticker} after panel merge: {feats.shape}")
        
        # Finalize (keeps symbol column for pooled analysis, but does NOT drop IV/IVRET columns)
        # So we need to call finalize_dataset but prevent it from dropping IV/IVRET columns for peers.
        # We'll drop only the target's own raw columns, not the panel columns.
        # To do this, we temporarily patch HIDE_COLUMNS and leak_cols logic.
        # Instead, we just drop the target's own 'stock_close' and 'iv_clip' columns, but keep IV_/IVRET_.
        # We'll copy finalize_dataset logic here with this tweak.
        out = feats.copy()
        for c in out.columns:
            if c not in {"ts_event", "symbol"}:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        # Drop missing targets
        out = out.dropna(subset=["iv_ret_fwd"])
        # Hide leaky columns (only those in HIDE_COLUMNS for this target)
        for col in HIDE_COLUMNS.get("iv_ret_fwd", []):
            if col in out.columns:
                out = out.drop(columns=col)
        # Remove only raw price columns from core data, keep iv_clip and panel
        # features so that an IV-level model can be trained later.
        for c in ["stock_close", "core_stock_close"]:
            if c in out.columns:
                out = out.drop(columns=c)
        # Keep symbol for pooled analysis
        out = out.reset_index(drop=True)
        out = _normalize_numeric_features(out, target_col="iv_ret_fwd")
        if debug:
            print(f"DEBUG: {ticker} after finalization: {out.shape}")
        frames.append(out)
    
    if not frames:
        if debug:
            print("DEBUG: No frames to concatenate - returning empty DataFrame")
        return pd.DataFrame()
        
    pooled = pd.concat(frames, ignore_index=True)
    if pooled.empty:
        if debug:
            print("DEBUG: Pooled dataset is empty after concatenation")
        return pooled
    
    if debug:
        print(f"DEBUG: Pooled dataset after concatenation: {pooled.shape}")
    
    # One-hot encode symbol (preserves original logic)
    pooled = pd.get_dummies(pooled, columns=["symbol"], prefix="sym", dtype=float)
    
    # Ensure all ticker columns exist
    for ticker in tickers:
        col = f"sym_{ticker}"
        if col not in pooled.columns:
            pooled[col] = 0.0
    
    # Column ordering (preserves original order)
    front = ["iv_ret_fwd"]
    if "core_iv_ret_fwd_abs" in pooled.columns:
        front.append("core_iv_ret_fwd_abs")
    if "iv_clip" in pooled.columns:
        front.append("iv_clip")
    onehots = [f"sym_{t}" for t in tickers]
    # Keep all IV_/IVRET_ columns (from panel) in the "other" section
    other = [c for c in pooled.columns if c not in front + onehots]
    
    final_pooled = pooled[front + other + onehots]

    # Normalize numeric features (preserve label and one-hots) and attach attrs
    try:
        final_pooled = _normalize_numeric_features(final_pooled, target_col="iv_ret_fwd")
    except Exception:
        pass
    
    if debug:
        print(f"DEBUG: Final pooled dataset shape: {final_pooled.shape}")
        debug_dir = Path("debug_snapshots")
        debug_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pooled_path = debug_dir / f"pooled_final_{timestamp}.csv"
        final_pooled.to_csv(pooled_path, index=False)
        print(f"DEBUG: Saved final pooled dataset to {pooled_path}")
        final_pooled = pooled[front + other + onehots]

        # normalize once on the pooled set, so attrs are attached to the returned DF
        final_pooled = _normalize_numeric_features(final_pooled, target_col="iv_ret_fwd")

    return final_pooled




def build_iv_return_dataset_time_safe(
    tickers: Sequence[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    r: float = 0.045,
    forward_steps: int = 15,
    tolerance: str = "15s",
    db_path: Path | str | None = None,
    cores: Optional[Dict[str, pd.DataFrame]] = None,
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Build per‑ticker datasets for forecasting forward IV return."""
    
    if debug:
        print(f"DEBUG: Building per-ticker datasets for {len(tickers)} tickers")
    
    # If cores not provided, need to load them (fallback for backward compatibility)
    if cores is None:
        from data.data_loader_coordinator import load_cores_with_auto_fetch
        cores = load_cores_with_auto_fetch(tickers, start, end, db_path)
    
    # Build panel
    panel = build_iv_panel(cores, tolerance=tolerance)
    
    datasets = {}
    for ticker in tickers:
        if ticker not in cores:
            if debug:
                print(f"DEBUG: Skipping {ticker} - not in cores")
            continue
            
        if debug:
            print(f"DEBUG: Processing per-ticker dataset for {ticker}")
            
        # Add features
        feats = add_all_features(cores[ticker], forward_steps=forward_steps, r=r)
        
        # Merge with panel
        feats = pd.merge_asof(
            feats.sort_values("ts_event"), panel.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
        
        # Finalize (removes symbol column for per-ticker analysis)
        datasets[ticker] = finalize_dataset(feats, "iv_ret_fwd", drop_symbol=False, debug=debug)
        
    if debug:
        print(f"DEBUG: Built {len(datasets)} per-ticker datasets")
        
    return datasets




def build_target_peer_dataset(
    target: str,
    tickers: Sequence[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "15s",
    db_path: Path | str | None = None,
    target_kind: str = "iv_ret",
    cores: Optional[Dict[str, pd.DataFrame]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Build dataset for single target vs peers."""
    
    if debug:
        print(f"DEBUG: Building target-peer dataset for {target} vs {len(tickers)} tickers")
        print(f"DEBUG: target_kind: {target_kind}")
    
    if target not in tickers:
        raise AssertionError("target must be included in tickers")
    
    # If cores not provided, need to load them (fallback for backward compatibility)
    if cores is None:
        from data.data_loader_coordinator import load_cores_with_auto_fetch
        cores = load_cores_with_auto_fetch(tickers, start, end, db_path)
    
    if target not in cores:
        raise ValueError(f"Target {target} produced no valid core")
    
    # Build panel and add features
    panel = build_iv_panel(cores, tolerance=tolerance)
    feats = add_all_features(cores[target], forward_steps=forward_steps, r=r)
    
    # Merge with panel
    feats = pd.merge_asof(
        feats.sort_values("ts_event"), panel.sort_values("ts_event"),
        on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
    )
    
    # Set target column based on requested target_kind
    if target_kind in ("iv_ret", "iv_ret_fwd"):
        target_col = "iv_ret_fwd"
    elif target_kind == "iv_ret_fwd_abs":
        target_col = "iv_ret_fwd_abs"
    elif target_kind == "iv":
        target_col = "iv_clip"
    else:
        raise ValueError(
            "target_kind must be one of 'iv_ret', 'iv_ret_fwd', 'iv_ret_fwd_abs', or 'iv'"
        )

    if debug:
        print(f"DEBUG: Using target column: {target_col}")
        print(f"DEBUG: Dataset shape before finalization: {feats.shape}")

    # Finalize dataset using the specific target column then rename to 'y'
    feats = finalize_dataset(feats, target_col, drop_symbol=True, debug=debug)
    final_dataset = feats.rename(columns={target_col: "y"})
    
    if debug:
        print(f"DEBUG: Final target-peer dataset shape: {final_dataset.shape}")
        debug_dir = Path("debug_snapshots")
        debug_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_peer_path = debug_dir / f"target_peer_{target}_{target_kind}_{timestamp}.csv"
        final_dataset.to_csv(target_peer_path, index=False)
        print(f"DEBUG: Saved target-peer dataset to {target_peer_path}")
    
    return final_dataset

from pandas.api.types import is_numeric_dtype

