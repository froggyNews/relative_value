import os
import sqlite3
import json
import logging
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm
from scipy.optimize import brentq

logging.basicConfig(level=logging.INFO)

@contextmanager
def suppress_runtime_warnings():
    """Context manager to suppress specific runtime warnings during SABR calculations."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero encountered.*")
        yield

# Config
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1h.db"))  # Updated for 1-hour data
ANNUAL_HOURS = 252 * 6.5  # 252 trading days * 6.5 hours per day
ANNUAL_MINUTES = 252 * 390  # Legacy for backward compatibility

# What to hide when predicting each target (preserved from original)
HIDE_COLUMNS = {
    "iv_ret_fwd": ["iv_ret_fwd_abs", "core_iv_ret_fwd_abs"],
    "iv_ret_fwd_abs": ["iv_ret_fwd"],
    "iv_clip": ["iv_ret_fwd", "iv_ret_fwd_abs", "core_iv_ret_fwd_abs"],
}

# Core features (preserved from original)
CORE_FEATURE_COLS = [
    # Basic option characteristics
    "opt_volume", "time_to_expiry", "days_to_expiry", "strike_price", "option_type_enc",
    # Greeks
    "delta", "gamma", "vega",
    # Time features
    "hour", "minute", "day_of_week",
    # SABR features
    "sabr_alpha", "sabr_rho", "sabr_nu", "moneyness", "log_moneyness",
    # Volatility features (updated for 1-hour timeframe)
    "rv_30h", "iv_ret_1h", "iv_ret_3h", "iv_ret_6h", "iv_sma_3h", "iv_sma_6h", "iv_std_6h", "iv_rsi_6h", "iv_zscore_6h",
    # Volume features (updated for 1-hour timeframe)
    "opt_vol_change_1h", "opt_vol_roll_3h", "opt_vol_roll_6h", "opt_vol_roll_24h", "opt_vol_zscore_6h"
]


# ------------------------------------------------------------
# Keep: Core feature engineering functions (other modules depend on these)
# ------------------------------------------------------------


def _hagan_implied_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """Approximate Black implied volatility under the SABR model."""
    with suppress_runtime_warnings():
        if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
            return np.nan
        
        # Check for invalid parameters
        if abs(rho) >= 1 or nu < 0 or not (0 <= beta <= 1):
            return np.nan

        if np.isclose(F, K):
            term1 = alpha / (F ** (1 - beta))
            term2 = 1 + (
                ((1 - beta) ** 2 / 24) * (alpha ** 2 / (F ** (2 - 2 * beta)))
                + (rho * beta * nu * alpha / (4 * F ** (1 - beta)))
                + ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
            ) * T
            return term1 * term2

        FK_beta = (F * K) ** ((1 - beta) / 2)
        logFK = np.log(F / K)
        z = (nu / alpha) * FK_beta * logFK
        
        # Handle the z/x_z term carefully
        if np.isclose(z, 0, atol=1e-7):
            # When z ≈ 0, limit of z/x_z = 1
            term2 = 1.0
        else:
            # Check if the log argument is valid
            sqrt_term = np.sqrt(1 - 2 * rho * z + z * z)
            log_arg = (sqrt_term + z - rho) / (1 - rho)
            
            if log_arg <= 0 or (1 - rho) == 0:
                return np.nan
                
            x_z = np.log(log_arg)
            
            if np.isclose(x_z, 0, atol=1e-12):
                # Avoid division by zero
                term2 = 1.0
            else:
                term2 = z / x_z
        
        term1 = alpha / (FK_beta * (1 + ((1 - beta) ** 2 / 24) * (logFK ** 2) + ((1 - beta) ** 4 / 1920) * (logFK ** 4)))
        term3 = 1 + (
            ((1 - beta) ** 2 / 24) * (alpha ** 2 / (FK_beta ** 2))
            + (rho * beta * nu * alpha / (4 * FK_beta))
            + ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
        ) * T
        
        result = term1 * term2 * term3
        
        # Final sanity check
        if not np.isfinite(result) or result <= 0:
            return np.nan
            
        return result


def _solve_sabr_alpha(sigma: float, F: float, K: float, T: float, beta: float, rho: float, nu: float) -> float:
    """Calibrate alpha for a single observation using Hagan's formula."""
    if np.any(np.isnan([sigma, F, K, T])) or sigma <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan
    
    # Check for invalid parameter ranges
    if abs(rho) >= 1 or nu < 0 or not (0 <= beta <= 1):
        return np.nan

    def objective(a: float) -> float:
        if a <= 0:
            return float('inf')  # Force positive alpha
        vol = _hagan_implied_vol(F, K, T, a, beta, rho, nu)
        if np.isnan(vol):
            return float('inf')
        return vol - sigma

    try:
        # Check if the objective function is well-behaved at the boundaries
        obj_low = objective(1e-6)
        obj_high = objective(5.0)
        
        if not (np.isfinite(obj_low) and np.isfinite(obj_high)):
            return np.nan
            
        # Only proceed if we can bracket the root
        if obj_low * obj_high > 0:
            return np.nan
            
        return brentq(objective, 1e-6, 5.0, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


def _add_sabr_features(df: pd.DataFrame, beta: float = 0.5) -> pd.DataFrame:
    """Compute simple SABR parameter features and drop raw price/IV columns."""
    F_series = df.get("stock_close")
    K_series = df.get("strike_price")
    T_series = df.get("time_to_expiry")
    sigma_series = df.get("iv_clip")
    if F_series is None or K_series is None or T_series is None or sigma_series is None:
        return df

    try:
        F = F_series.astype(float).to_numpy()
        K = K_series.astype(float).to_numpy()
        T = np.maximum(T_series.astype(float).to_numpy(), 1e-9)
        sigma = sigma_series.astype(float).to_numpy()

        # More robust heuristic estimates for rho and nu
        moneyness = np.clip((K / F) - 1.0, -2.0, 2.0)  # Clip extreme moneyness
        rho = np.tanh(moneyness * 3.0)  # Reduced sensitivity
        
        # Improved nu estimation with better fallback (adjusted for 1-hour data)
        nu_series = (
            df["iv_clip"].astype(float).rolling(30, min_periods=5).std() * np.sqrt(ANNUAL_HOURS / 30)
        ).shift(1)
        nu = nu_series.fillna(0.3).to_numpy()  # Fallback to reasonable default
        nu = np.clip(nu, 0.01, 3.0)  # Clip to reasonable range

        # Vectorized SABR alpha calculation with better error handling
        alpha = np.full(len(sigma), np.nan)
        
        # Only calculate for valid data points
        valid_mask = (
            np.isfinite(sigma) & np.isfinite(F) & np.isfinite(K) & np.isfinite(T) &
            np.isfinite(rho) & np.isfinite(nu) &
            (sigma > 0) & (F > 0) & (K > 0) & (T > 0) & (nu > 0) &
            (np.abs(rho) < 0.99)  # Avoid extreme correlations
        )
        
        if np.any(valid_mask):
            for i in np.where(valid_mask)[0]:
                try:
                    alpha[i] = _solve_sabr_alpha(sigma[i], F[i], K[i], T[i], beta, rho[i], nu[i])
                except:
                    alpha[i] = np.nan
        
        # Fallback for failed SABR calculations
        failed_mask = ~np.isfinite(alpha)
        if np.any(failed_mask):
            # Simple fallback: use ATM IV scaled by moneyness effect
            fallback_alpha = sigma * (F ** (1 - beta))
            alpha[failed_mask] = fallback_alpha[failed_mask]

        df["sabr_alpha"] = alpha
        df["sabr_beta"] = beta
        df["sabr_rho"] = rho
        df["sabr_nu"] = nu
        
        # Add SABR-based features
        df["moneyness"] = moneyness
        df["log_moneyness"] = np.log(K / F)
        
    except Exception as e:
        print(f"Warning: SABR feature calculation failed: {e}")
        # Create dummy SABR features
        n_rows = len(df)
        df["sabr_alpha"] = np.full(n_rows, 0.2)
        df["sabr_beta"] = beta
        df["sabr_rho"] = np.zeros(n_rows)
        df["sabr_nu"] = np.full(n_rows, 0.3)
        df["moneyness"] = np.zeros(n_rows)
        df["log_moneyness"] = np.zeros(n_rows)

    # Remove raw stock price information but keep iv_clip so it can be used as
    # a modeling target later in the pipeline.
    if "stock_close" in df.columns:
        df = df.drop(columns=["stock_close"])
    return df

def _validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean input data before feature engineering."""
    required_cols = ["ts_event", "iv_clip", "strike_price", "time_to_expiry", "option_type"]
    
    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Basic data validation
    initial_rows = len(df)
    
    # Remove rows with invalid core data
    df = df.dropna(subset=["iv_clip", "strike_price", "time_to_expiry"])
    df = df[df["iv_clip"] > 0]  # IV must be positive
    df = df[df["strike_price"] > 0]  # Strike must be positive  
    df = df[df["time_to_expiry"] > 0]  # Time to expiry must be positive
    
    cleaned_rows = len(df)
    if cleaned_rows < initial_rows:
        print(f"Data validation: Removed {initial_rows - cleaned_rows} invalid rows ({cleaned_rows} remaining)")
    
    return df
def _calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    """Calculate RSI for a given series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_all_features(df: pd.DataFrame, forward_steps: int = 1, r: float = 0.045, validate: bool = True) -> pd.DataFrame:
    """Centralized feature engineering (preserves all original feature logic)."""
    df = df.copy()
    
    # Optional input validation
    if validate:
        df = _validate_input_data(df)
    
    # Forward returns (preserves original single log transform approach)
    log_col = np.log(df["iv_clip"].astype(float))
    fwd = log_col.shift(-forward_steps) - log_col
    df["iv_ret_fwd"] = fwd
    df["iv_ret_fwd_abs"] = fwd.abs()
    
    # Vectorized Greeks (preserves original implementation)
    S = df["stock_close"].astype(float).to_numpy()
    K = df["strike_price"].astype(float).to_numpy()
    T = np.maximum(df["time_to_expiry"].astype(float).to_numpy(), 1e-9)
    sig = df["iv_clip"].astype(float).to_numpy()
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    is_call = df["option_type"].astype(str).str.upper().str[0].eq("C").to_numpy()
    df["delta"] = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    df["gamma"] = pdf / (S * sig * sqrtT)
    df["vega"] = S * pdf * sqrtT
    
    # Time features (fixed implementation)
    if "ts_event" in df.columns:
        # Ensure ts_event is datetime with timezone info
        if not pd.api.types.is_datetime64_any_dtype(df["ts_event"]):
            df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        
        # Extract time features safely
        try:
            df["hour"] = df["ts_event"].dt.hour.astype("int16")
            df["minute"] = df["ts_event"].dt.minute.astype("int16") 
            df["day_of_week"] = df["ts_event"].dt.dayofweek.astype("int16")
        except Exception as e:
            print(f"Warning: Could not extract time features: {e}")
            # Fallback: create dummy time features
            df["hour"] = 15  # Default to mid-trading day
            df["minute"] = 30
            df["day_of_week"] = 1  # Default to Tuesday
    
    df["days_to_expiry"] = (df["time_to_expiry"] * 365.0).astype("float32")
    df["option_type_enc"] = (df["option_type"].astype(str).str.upper().str[0]
                            .map({"P": 0, "C": 1}).astype("float32"))
    
    # Equity context (adjusted for 1-hour data)
    if "stock_close" in df.columns:
        logS = np.log(df["stock_close"].astype(float))
        ret_1h = logS.diff()  # 1-hour returns
        rv = ret_1h.rolling(30).std()  # 30-hour rolling volatility
        df["rv_30h"] = (rv * np.sqrt(ANNUAL_HOURS / 30)).shift(1)  # Annualized realized vol
    
    # Option flow (adjusted for 1-hour data)
    if "opt_volume" in df.columns:
        pct_change = df["opt_volume"].pct_change()
        df["opt_vol_change_1h"] = (pct_change.replace([np.inf, -np.inf], np.nan).fillna(0.0))
        df["opt_vol_roll_6h"] = df["opt_volume"].rolling(6).mean().shift(1)  # 6-hour average
        
        # Enhanced volume features (adjusted for 1-hour timeframe)
        df["opt_vol_roll_3h"] = df["opt_volume"].rolling(3, min_periods=1).mean().shift(1)
        df["opt_vol_roll_24h"] = df["opt_volume"].rolling(24, min_periods=6).mean().shift(1)  # Daily average
        df["opt_vol_zscore_6h"] = (
            (df["opt_volume"] - df["opt_vol_roll_6h"]) / 
            (df["opt_volume"].rolling(6, min_periods=3).std().shift(1) + 1e-8)
        )
    
    # Enhanced volatility features (adjusted for 1-hour data)
    if "iv_clip" in df.columns:
        iv_log = np.log(df["iv_clip"])
        
        # IV momentum features (1-hour based)
        df["iv_ret_1h"] = iv_log.diff()
        df["iv_ret_3h"] = iv_log.diff(3)
        df["iv_ret_6h"] = iv_log.diff(6)
        
        # IV rolling statistics (adjusted for 1-hour timeframe)
        df["iv_sma_3h"] = df["iv_clip"].rolling(3, min_periods=2).mean().shift(1)
        df["iv_sma_6h"] = df["iv_clip"].rolling(6, min_periods=3).mean().shift(1)
        df["iv_std_6h"] = df["iv_clip"].rolling(6, min_periods=3).std().shift(1)
        
        # IV relative position
        df["iv_rsi_6h"] = _calculate_rsi(df["iv_clip"], 6)
        
        # IV z-score
        df["iv_zscore_6h"] = (
            (df["iv_clip"] - df["iv_sma_6h"]) / (df["iv_std_6h"] + 1e-8)
        )

    # SABR parameters and hide raw price/IV data
    df = _add_sabr_features(df)

    return df


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


