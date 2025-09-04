# -*- coding: utf-8 -*-
"""
Refactored feature engineering with granular helper functions.

Public API preserved:
- _add_sabr_features(df, beta=0.5)
- _validate_input_data(df)
- _calculate_rsi(series, window)
- add_all_features(df, forward_steps=1, r=0.045, validate=True)

"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .constants import ANNUAL_HOURS
from .sabr import _hagan_implied_vol, _solve_sabr_alpha

logger = logging.getLogger(__name__)


# -------------------------
# Basic validators & casting
# -------------------------
def _validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean input data before feature engineering.

    Ensures required columns exist and filters invalid rows for
    iv_clip, strike_price, and time_to_expiry (must be > 0).
    """
    required_cols = ["ts_event", "iv_clip", "strike_price", "time_to_expiry", "option_type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_rows = len(df)
    df = df.dropna(subset=["iv_clip", "strike_price", "time_to_expiry"])
    df = df[df["iv_clip"] > 0]
    df = df[df["strike_price"] > 0]
    df = df[df["time_to_expiry"] > 0]

    removed = initial_rows - len(df)
    if removed > 0:
        logger.info("Data validation removed %d invalid rows (%d remaining)", removed, len(df))
    return df


def _ensure_ts_event_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df['ts_event'] is timezone-aware datetime if present."""
    if "ts_event" not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df["ts_event"]):
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    return df


# -------------------------
# Core target construction
# -------------------------
def _compute_forward_iv_returns(df: pd.DataFrame, forward_steps: int) -> pd.DataFrame:
    """Compute forward log returns on iv_clip and attach absolute value."""
    iv = pd.to_numeric(df["iv_clip"], errors="coerce")
    log_iv = np.log(iv)
    fwd = log_iv.shift(-forward_steps) - log_iv
    df["iv_ret_fwd"] = fwd
    df["iv_ret_fwd_abs"] = fwd.abs()
    return df


# -------------------------
# Greeks (Black–Scholes style)
# -------------------------
def _compute_black_scholes_greeks(df: pd.DataFrame, r: float) -> pd.DataFrame:
    """Compute (delta, gamma, vega). Requires stock_close if available.

    Skips if 'stock_close' is missing. All operations are vectorized.
    """
    if "stock_close" not in df.columns:
        return df

    S = pd.to_numeric(df["stock_close"], errors="coerce").to_numpy()
    K = pd.to_numeric(df["strike_price"], errors="coerce").to_numpy()
    T = np.maximum(pd.to_numeric(df["time_to_expiry"], errors="coerce").to_numpy(), 1e-9)
    sig = pd.to_numeric(df["iv_clip"], errors="coerce").to_numpy()

    sqrtT = np.sqrt(T)
    # Avoid division by zero
    denom = np.clip(sig * sqrtT, 1e-12, np.inf)
    d1 = (np.log(np.clip(S / K, 1e-12, np.inf)) + (r + 0.5 * sig * sig) * T) / denom
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)

    is_call = df["option_type"].astype(str).str.upper().str[0].eq("C").to_numpy()
    df["delta"] = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    # gamma ~ pdf / (S * sigma * sqrt(T))
    df["gamma"] = pdf / np.clip(S * sig * sqrtT, 1e-12, np.inf)
    df["vega"] = S * pdf * sqrtT
    return df


# -------------------------
# Calendar/time attributes
# -------------------------
def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, minute, day_of_week from 'ts_event' if present; otherwise use robust defaults."""
    if "ts_event" not in df.columns:
        return df

    try:
        s = df["ts_event"].dt  # requires datetime64 dtype; ensured earlier
        df["hour"] = s.hour.astype("int16")
        df["minute"] = s.minute.astype("int16")
        df["day_of_week"] = s.dayofweek.astype("int16")
    except Exception as e:
        logger.warning("Could not extract time features cleanly: %s", e)
        # Minimal, stable defaults (mid-session)
        df["hour"] = 15
        df["minute"] = 30
        df["day_of_week"] = 1
    return df


def _add_misc_option_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add days_to_expiry and option_type_enc features."""
    df["days_to_expiry"] = pd.to_numeric(df["time_to_expiry"], errors="coerce") * 365.0
    df["days_to_expiry"] = df["days_to_expiry"].astype("float32")

    df["option_type_enc"] = (
        df["option_type"].astype(str).str.upper().str[0].map({"P": 0.0, "C": 1.0}).astype("float32")
    )
    return df


# -------------------------
# Equity context
# -------------------------
def _add_equity_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute equity realized volatility context using hourly bars.

    rv_30h is annualized with ANNUAL_HOURS and shifted by 1 to prevent leakage.
    """
    if "stock_close" not in df.columns:
        return df

    logS = np.log(pd.to_numeric(df["stock_close"], errors="coerce"))
    ret_1h = logS.diff()
    rv = ret_1h.rolling(30).std()
    df["rv_30h"] = (rv * np.sqrt(ANNUAL_HOURS / 30.0)).shift(1)
    return df


# -------------------------
# Option flow
# -------------------------
def _add_option_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add option volume–based features; robust to zeros/NaNs."""
    if "opt_volume" not in df.columns:
        return df

    vol = pd.to_numeric(df["opt_volume"], errors="coerce")
    pct_change = vol.pct_change()
    df["opt_vol_change_1h"] = pct_change.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["opt_vol_roll_6h"] = vol.rolling(6).mean().shift(1)
    df["opt_vol_roll_3h"] = vol.rolling(3, min_periods=1).mean().shift(1)
    df["opt_vol_roll_24h"] = vol.rolling(24, min_periods=6).mean().shift(1)
    df["opt_vol_zscore_6h"] = (vol - df["opt_vol_roll_6h"]) / (vol.rolling(6, min_periods=3).std().shift(1) + 1e-8)
    return df


# -------------------------
# IV momentum & stats
# -------------------------
def _calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index (RSI) for a series (no warm-up trimming)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _add_iv_momentum_and_stats(df: pd.DataFrame) -> pd.DataFrame:
    """IV momentum, SMA/STD bands, RSI, and z-scores over hourly bars."""
    if "iv_clip" not in df.columns:
        return df

    iv = pd.to_numeric(df["iv_clip"], errors="coerce")
    iv_log = np.log(iv.replace(0, np.nan))

    # momentum
    df["iv_ret_1h"] = iv_log.diff()
    df["iv_ret_3h"] = iv_log.diff(3)
    df["iv_ret_6h"] = iv_log.diff(6)

    # rolling stats
    df["iv_sma_3h"] = iv.rolling(3, min_periods=2).mean().shift(1)
    df["iv_sma_6h"] = iv.rolling(6, min_periods=3).mean().shift(1)
    df["iv_std_6h"] = iv.rolling(6, min_periods=3).std().shift(1)

    # RSI and z-score
    df["iv_rsi_6h"] = _calculate_rsi(iv, 6)
    df["iv_zscore_6h"] = (iv - df["iv_sma_6h"]) / (df["iv_std_6h"] + 1e-8)
    return df


# -------------------------
# SABR helpers
# -------------------------
def _compute_moneyness(F: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (moneyness, log_moneyness) with clipping and log-protect."""
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.clip((K / np.clip(F, 1e-12, np.inf)) - 1.0, -2.0, 2.0)
        lm = np.log(np.clip(K, 1e-12, np.inf) / np.clip(F, 1e-12, np.inf))
    return m, lm


def _estimate_sabr_rho(moneyness: np.ndarray) -> np.ndarray:
    """Heuristic rho from moneyness (tanh for stability)."""
    return np.tanh(moneyness * 3.0)


def _estimate_sabr_nu(iv_clip: pd.Series, window: int = 30) -> pd.Series:
    """Heuristic nu from rolling IV std, annualized for hourly data."""
    rolling_std = pd.to_numeric(iv_clip, errors="coerce").rolling(window, min_periods=5).std()
    # sqrt(ANNUAL_HOURS / window) normalizes from window-hours to annual
    return (rolling_std * np.sqrt(ANNUAL_HOURS / float(window))).shift(1)


def _solve_sabr_alpha_vectorized(
    sigma: np.ndarray, F: np.ndarray, K: np.ndarray, T: np.ndarray,
    beta: float, rho: np.ndarray, nu: np.ndarray
) -> np.ndarray:
    """Solve SABR alpha per-row using _solve_sabr_alpha with robust masking."""
    n = len(sigma)
    alpha = np.full(n, np.nan, dtype=float)

    valid = (
        np.isfinite(sigma) & np.isfinite(F) & np.isfinite(K) & np.isfinite(T) &
        np.isfinite(rho) & np.isfinite(nu) &
        (sigma > 0) & (F > 0) & (K > 0) & (T > 0) & (nu > 0) &
        (np.abs(rho) < 0.99)
    )
    idx = np.where(valid)[0]
    for i in idx:
        try:
            alpha[i] = _solve_sabr_alpha(sigma[i], F[i], K[i], T[i], beta, rho[i], nu[i])
        except Exception:
            # leave as NaN; will be filled by fallback
            pass
    return alpha


def _fallback_alpha_from_sigma_beta(sigma: np.ndarray, F: np.ndarray, beta: float) -> np.ndarray:
    """Simple alpha fallback using ATM IV scaled by F^(1-beta)."""
    return sigma * np.power(np.clip(F, 1e-12, np.inf), 1.0 - beta)


def _add_sabr_features(df: pd.DataFrame, beta: float = 0.5) -> pd.DataFrame:
    """Compute SABR feature set and drop raw price column.

    Adds: sabr_alpha, sabr_beta, sabr_rho, sabr_nu, moneyness, log_moneyness
    Drops: stock_close (to avoid target leakage via spot)
    """
    F_series = df.get("stock_close")
    K_series = df.get("strike_price")
    T_series = df.get("time_to_expiry")
    sigma_series = df.get("iv_clip")
    if F_series is None or K_series is None or T_series is None or sigma_series is None:
        return df

    try:
        F = pd.to_numeric(F_series, errors="coerce").to_numpy()
        K = pd.to_numeric(K_series, errors="coerce").to_numpy()
        T = np.maximum(pd.to_numeric(T_series, errors="coerce").to_numpy(), 1e-9)
        sigma = pd.to_numeric(sigma_series, errors="coerce").to_numpy()

        # moneyness & its log
        moneyness, log_moneyness = _compute_moneyness(F, K)
        rho = _estimate_sabr_rho(moneyness)

        nu_series = _estimate_sabr_nu(df["iv_clip"], window=30)
        nu = nu_series.fillna(0.3).to_numpy()
        nu = np.clip(nu, 0.01, 3.0)

        alpha = _solve_sabr_alpha_vectorized(sigma, F, K, T, beta, rho, nu)
        # fill NaNs with fallback
        need_fallback = ~np.isfinite(alpha)
        if np.any(need_fallback):
            alpha[need_fallback] = _fallback_alpha_from_sigma_beta(sigma, F, beta)[need_fallback]

        df["sabr_alpha"] = alpha
        df["sabr_beta"] = float(beta)
        df["sabr_rho"] = rho
        df["sabr_nu"] = nu
        df["moneyness"] = moneyness
        df["log_moneyness"] = log_moneyness

    except Exception as e:
        logger.warning("SABR feature calculation failed (%s). Falling back to constants.", e)
        n_rows = len(df)
        df["sabr_alpha"] = np.full(n_rows, 0.2, dtype=float)
        df["sabr_beta"] = float(beta)
        df["sabr_rho"] = np.zeros(n_rows, dtype=float)
        df["sabr_nu"] = np.full(n_rows, 0.3, dtype=float)
        df["moneyness"] = np.zeros(n_rows, dtype=float)
        df["log_moneyness"] = np.zeros(n_rows, dtype=float)

    # drop spot to avoid data leakage later in the pipeline
    if "stock_close" in df.columns:
        df = df.drop(columns=["stock_close"])
    return df


# -------------------------
# Orchestration (public API)
# -------------------------
def add_all_features(df: pd.DataFrame, forward_steps: int = 1, r: float = 0.045, validate: bool = True) -> pd.DataFrame:
    """Centralized feature engineering with robust, testable helpers.

    Steps (time-safe ordering):
      1) Optional validation & ts casting
      2) Forward IV returns (target)
      3) Greeks (if stock_close present)
      4) Time features + misc option features
      5) Equity context (rv_30h, shift(1))
      6) Option flow features
      7) IV momentum & stats
      8) SABR features (drops stock_close)

    Returns the modified DataFrame (copy of input).
    """
    df = df.copy()
    if validate:
        df = _validate_input_data(df)

    df = _ensure_ts_event_datetime(df)
    df = _compute_forward_iv_returns(df, forward_steps=forward_steps)
    df = _compute_black_scholes_greeks(df, r=r)
    df = _add_time_features(df)
    df = _add_misc_option_features(df)
    df = _add_equity_context_features(df)
    df = _add_option_flow_features(df)
    df = _add_iv_momentum_and_stats(df)
    df = _add_sabr_features(df, beta=0.5)
    return df


__all__ = [
    "_validate_input_data",
    "_ensure_ts_event_datetime",
    "_compute_forward_iv_returns",
    "_compute_black_scholes_greeks",
    "_add_time_features",
    "_add_misc_option_features",
    "_add_equity_context_features",
    "_add_option_flow_features",
    "_add_iv_momentum_and_stats",
    "_compute_moneyness",
    "_estimate_sabr_rho",
    "_estimate_sabr_nu",
    "_solve_sabr_alpha_vectorized",
    "_fallback_alpha_from_sigma_beta",
    "_add_sabr_features",
    "_calculate_rsi",
    "add_all_features",
]
