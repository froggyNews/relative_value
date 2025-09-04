import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .constants import HIDE_COLUMNS


def finalize_dataset(df: pd.DataFrame, target_col: str, drop_symbol: bool = True, debug: bool = False) -> pd.DataFrame:
    """Centralized dataset finalization (preserves original cleanup logic)."""
    # Convert to numeric (preserves original approach)
    out = df.copy()
    
    # Save raw data snapshot if debug mode
    if debug:
        debug_dir = Path("debug_snapshots")
        debug_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_snapshot_path = debug_dir / f"raw_data_{target_col}_{timestamp}.csv"
        out.to_csv(raw_snapshot_path, index=False)
        print(f"DEBUG: Saved raw data snapshot to {raw_snapshot_path}")
    
    for c in out.columns:
        if c != "ts_event":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    
    # Drop missing targets
    initial_rows = len(out)
    out = out.dropna(subset=[target_col])
    dropped_target = initial_rows - len(out)
    
    if debug and dropped_target > 0:
        print(f"DEBUG: Dropped {dropped_target} rows with missing {target_col}")
    
    # Hide leaky columns (preserves original hide logic)
    hidden_cols = []
    for col in HIDE_COLUMNS.get(target_col, []):
        if col in out.columns:
            out = out.drop(columns=col)
            hidden_cols.append(col)
    
    if debug and hidden_cols:
        print(f"DEBUG: Hidden leaky columns for {target_col}: {hidden_cols}")

    # Remove raw stock price information and peer IV/IVRET columns. Retain
    # 'iv_clip' when it is the modeling target so that level models can be
    # trained.
    leak_cols = [c for c in out.columns if c == "stock_close" or c.startswith("IV_") or c.startswith("IVRET_")]
    if target_col != "iv_clip" and "iv_clip" in out.columns:
        leak_cols.append("iv_clip")
    if leak_cols:
        out = out.drop(columns=leak_cols)
        if debug:
            print(f"DEBUG: Removed leak columns: {leak_cols}")
    
    # Drop symbol if requested (for per-ticker datasets)
    if drop_symbol and "symbol" in out.columns:
        out = out.drop(columns=["symbol"])
        if debug:
            print(f"DEBUG: Dropped symbol column")
    
    out = out.reset_index(drop=True)
    out = _normalize_numeric_features(out, target_col=target_col)

    if debug:
        print(f"DEBUG: Final dataset shape: {out.shape}")
        print(f"DEBUG: Final columns: {list(out.columns)}")
        
        # Save final processed data snapshot
        final_snapshot_path = debug_dir / f"final_data_{target_col}_{timestamp}.csv"
        out.to_csv(final_snapshot_path, index=False)
        print(f"DEBUG: Saved final data snapshot to {final_snapshot_path}")
        
        # Save column info
        info_path = debug_dir / f"column_info_{target_col}_{timestamp}.json"
        column_info = {
            "target_column": target_col,
            "final_columns": list(out.columns),
            "hidden_columns": hidden_cols,
            "leak_columns": leak_cols,
            "initial_rows": initial_rows,
            "final_rows": len(out),
            "dropped_rows": dropped_target,
        }
        with open(info_path, 'w') as f:
            json.dump(column_info, f, indent=2, default=str)
        print(f"DEBUG: Saved column info to {info_path}")
    
    # log final set of columns for inspection
    logging.getLogger(__name__).info("Final dataset columns: %s", list(out.columns))

    return out



# ------------------------------------------------------------
# Keep: Main dataset building functions (required by other modules)
# ------------------------------------------------------------



def _normalize_numeric_features(out: pd.DataFrame, target_col: str) -> pd.DataFrame:
    keep = {"ts_event"}            # never normalize timestamps
    skip_prefixes = ("sym_",)      # keep one-hots as 0/1
    skip_exact = {target_col}      # don't normalize the label

    cols = []
    for c in out.columns:
        if c in keep or c in skip_exact: 
            continue
        if any(c.startswith(p) for p in skip_prefixes):
            continue
        if is_numeric_dtype(out[c]):
            cols.append(c)

    if cols:
        mu = out[cols].mean()
        sd = out[cols].std().replace(0.0, 1.0).fillna(1.0)
        out[cols] = (out[cols] - mu) / sd
        out.attrs["norm_means"] = {k: float(mu[k]) for k in mu.index}
        out.attrs["norm_stds"]  = {k: float(sd[k]) for k in sd.index}
    return out

# Export original names for backward compatibility
__all__ = [
    "build_pooled_iv_return_dataset_time_safe",
    "build_iv_return_dataset_time_safe", 
    "build_target_peer_dataset",
    "add_all_features",
    "build_iv_panel",
    "finalize_dataset"
]



