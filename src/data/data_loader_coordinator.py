"""
Data loader coordinator - helps orchestrate data loading across existing modules.

This module doesn't duplicate functionality but provides a clean interface
to coordinate between feature_engineering.py, fetch_data_sqlite.py, and 
train_peer_effects.py.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, Sequence, Optional, List, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# Import existing functions with safe error handling and path fallback
try:
    from src.data.fetch_data_sqlite import (
        fetch_and_save, get_conn, init_schema,
        auto_fetch_missing_data, ensure_data_availability,
        check_data_exists,
    )
    FETCH_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import fetch_data_sqlite functions: {e}")
    # Fallback: try loading the module from a known relative path
    try:
        import importlib.util as _importlib_util
        import sys as _sys
        from pathlib import Path as _Path

        repo_root = _Path(__file__).resolve().parents[2]
        candidate_paths = [
            repo_root / "data" / "fetch_data_sqlite.py",        # project-root/data
            repo_root / "src" / "data" / "fetch_data_sqlite.py", # src/data
        ]
        loaded = False
        for module_path in candidate_paths:
            if not module_path.exists():
                continue
            spec = _importlib_util.spec_from_file_location("_fetch_data_sqlite_local", str(module_path))
            if spec and spec.loader:
                _mod = _importlib_util.module_from_spec(spec)
                spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
                fetch_and_save = getattr(_mod, "fetch_and_save", None)
                get_conn = getattr(_mod, "get_conn", None)
                init_schema = getattr(_mod, "init_schema", None)
                auto_fetch_missing_data = getattr(_mod, "auto_fetch_missing_data", None)
                ensure_data_availability = getattr(_mod, "ensure_data_availability", None)
                check_data_exists = getattr(_mod, "check_data_exists", None)
                FETCH_FUNCTIONS_AVAILABLE = True
                loaded = True
                break
        if not loaded:
            raise ImportError(f"Module file not found in candidates: {candidate_paths}")
    except Exception as e2:
        print(f"Warning: Path-based import of fetch_data_sqlite failed: {e2}")
        fetch_and_save = None
        get_conn = None
        init_schema = None
        auto_fetch_missing_data = None
        ensure_data_availability = None
        check_data_exists = None
        FETCH_FUNCTIONS_AVAILABLE = False


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> list:
    """Get column names for a table."""
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]
    except Exception:
        return []


def _calculate_iv(price: float, S: float, K: float, T: float, cp: str, r: float) -> float:
    """IV calculation (moved here to avoid circular imports)."""
    if not np.isfinite([price, S, K, T, r]).all() or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    
    intrinsic = max(S - K, 0.0) if cp.upper().startswith('C') else max(K - S, 0.0)
    if price <= intrinsic + 1e-10:
        return 1e-6
        
    def bs_price(sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return intrinsic
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        if cp.upper().startswith('C'):
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    try:
        return brentq(lambda sig: bs_price(sig) - price, 1e-6, 5.0, maxiter=100, xtol=1e-8)
    except:
        return np.nan


def _safe_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Safely check if a table exists in the database."""
    try:
        result = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", 
            (table_name,)
        ).fetchone()
        return result is not None
    except Exception:
        return False


def _safe_data_exists(conn: sqlite3.Connection, table: str, ticker: str, start: str = None, end: str = None) -> bool:
    """Safely check if data exists for a ticker in the given time range."""
    try:
        where_clauses, params = ["ticker=?"], [ticker]
        if start:
            where_clauses.append("ts_event >= ?")
            params.append(pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if end:
            where_clauses.append("ts_event <= ?")
            params.append(pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            
        query = f"SELECT COUNT(*) FROM {table} WHERE {' AND '.join(where_clauses)}"
        result = conn.execute(query, params).fetchone()
        return result[0] > 0 if result else False
    except Exception:
        return False


def _populate_atm_slices(conn: sqlite3.Connection, ticker: str) -> None:
    """Populate ATM slices table from processed_merged_1m data."""
    try:
        # Get column schemas for both tables
        atm_cols = _get_table_columns(conn, "atm_slices_1m")
        processed_cols = _get_table_columns(conn, "processed_merged_1m")
        
        if not atm_cols or not processed_cols:
            print(f"  ✗ Could not get table schemas for {ticker}")
            return
        
        # Find common columns (excluding any extras)
        common_cols = [col for col in atm_cols if col in processed_cols]
        
        if len(common_cols) < 10:  # Sanity check
            print(f"  ✗ Too few common columns ({len(common_cols)}) for {ticker}")
            return
        
        # Build the query with exact column matching
        cols_str = ", ".join(common_cols)
        
        q = f"""
        INSERT OR REPLACE INTO atm_slices_1m ({cols_str})
        SELECT {cols_str}
        FROM (
            SELECT
                {cols_str},
                ROW_NUMBER() OVER (
                  PARTITION BY ticker, ts_event, expiry_date
                  ORDER BY ABS(strike_price - stock_close)
                ) rn
            FROM processed_merged_1m
            WHERE ticker = ?
        )
        WHERE rn = 1;
        """
        
        result = conn.execute(q, (ticker,))
        conn.commit()
        rows_affected = result.rowcount
        print(f"  ✓ Populated ATM slices for {ticker} ({rows_affected} rows)")
        
    except Exception as e:
        print(f"  ✗ Failed to populate ATM slices for {ticker}: {e}")


def load_ticker_core(ticker: str, start=None, end=None, r=0.045, db_path=None, atm_only: bool = True) -> pd.DataFrame:
    """Load ticker core data with IV calculation."""
    
    if db_path is None:
        db_path = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
    
    # Check if database file exists
    if not Path(db_path).exists():
        print(f"Database file does not exist: {db_path}")
        return pd.DataFrame()
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Debug: Check table schemas
            # print(f"DEBUG: atm_slices_1m columns: {_get_table_columns(conn, 'atm_slices_1m')}")
            # print(f"DEBUG: processed_merged_1m columns: {_get_table_columns(conn, 'processed_merged_1m')}")
            
            # Try tables depending on whether we want full surface or ATM-only
            table = None
            if atm_only:
                for candidate in ["atm_slices_1m", "processed_merged_1m", "processed_merged"]:
                    if _safe_table_exists(conn, candidate):
                        if _safe_data_exists(conn, candidate, ticker, start, end):
                            table = candidate
                            break
            else:
                # Prefer processed tables that contain the full surface; skip atm_slices
                for candidate in ["processed_merged_1m", "processed_merged", "merged_1m"]:
                    if _safe_table_exists(conn, candidate):
                        if _safe_data_exists(conn, candidate, ticker, start, end):
                            table = candidate
                            break
            
            if table is None:
                print(f"No data found for {ticker} in any table")
                return pd.DataFrame()
            
            # If ATM-only and we have processed_merged_1m but not atm_slices_1m, populate ATM slices
            if atm_only and table == "processed_merged_1m" and _safe_table_exists(conn, "atm_slices_1m"):
                # Only populate if ATM table is truly empty for this ticker
                if not _safe_data_exists(conn, "atm_slices_1m", ticker, start, end):
                    _populate_atm_slices(conn, ticker)
                    # Check if ATM data is now available
                    if _safe_data_exists(conn, "atm_slices_1m", ticker, start, end):
                        table = "atm_slices_1m"
            
            where_clauses, params = ["ticker=?"], [ticker]
            if start:
                start_ts = pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                where_clauses.append("ts_event >= ?")
                params.append(start_ts)
                print(f"DEBUG: Filtering {ticker} with start >= {start} (converted to {start_ts})")
            if end:
                end_ts = pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                where_clauses.append("ts_event <= ?")
                params.append(end_ts)
                print(f"DEBUG: Filtering {ticker} with end <= {end} (converted to {end_ts})")
            
            # Get required columns that actually exist in the table
            required_cols = ["ts_event", "expiry_date", "opt_symbol", "stock_symbol",
                           "opt_close", "stock_close", "opt_volume", "stock_volume",
                           "option_type", "strike_price", "time_to_expiry", "moneyness"]
            
            available_cols = _get_table_columns(conn, table)
            select_cols = [col for col in required_cols if col in available_cols]
            
            if len(select_cols) < 8:  # Minimum required columns
                print(f"Insufficient columns in {table} for {ticker}")
                return pd.DataFrame()
            
            cols_str = ", ".join(select_cols)
            
            # Build query based on table and ATM preference
            if table == "atm_slices_1m":
                query = f"""
                SELECT {cols_str}
                FROM {table} WHERE {' AND '.join(where_clauses)}
                ORDER BY ts_event
                """
            elif atm_only:
                # For processed_merged_1m, select ATM option per expiry and timestamp
                query = f"""
                SELECT {cols_str}
                FROM (
                    SELECT {cols_str},
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker, ts_event, expiry_date
                               ORDER BY ABS(strike_price - stock_close)
                           ) as rn
                    FROM {table} 
                    WHERE {' AND '.join(where_clauses)}
                ) ranked
                WHERE rn = 1
                ORDER BY ts_event
                """
            else:
                # Full surface: pull all processed rows (no ATM filtering)
                query = f"""
                SELECT {cols_str}
                FROM {table}
                WHERE {' AND '.join(where_clauses)}
                ORDER BY ts_event
                """
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["ts_event", "expiry_date"])
            
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df
        
    try:
        # IV calculation
        df["iv"] = df.apply(lambda row: _calculate_iv(
            row["opt_close"], row["stock_close"], row["strike_price"], 
            max(row["time_to_expiry"], 1e-6), row["option_type"], r
        ), axis=1)
        
        # Core cleanup (preserves original column selection and processing)
        available_keep = [col for col in ["ts_event", "expiry_date", "iv", "opt_volume", "stock_close", 
                                        "stock_volume", "time_to_expiry", "strike_price", "option_type"] 
                         if col in df.columns]
        df = df[available_keep].copy()
        df["symbol"] = ticker
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.dropna(subset=["iv"]).sort_values("ts_event").reset_index(drop=True)
        df["iv_clip"] = df["iv"].clip(lower=1e-6)
        
    except Exception as e:
        print(f"Error processing IV data for {ticker}: {e}")
        return pd.DataFrame()
    
    return df


class DataCoordinator:
    """Coordinates data loading and ensures consistency across modules."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
        self.api_key = os.getenv("DATABENTO_API_KEY")

    def _infer_timeframe(self) -> str:
        name = str(self.db_path).lower()
        return "1m" if ("1m" in name or name.endswith("_1m.db")) else "1h"
        
    def _safe_fetch_data(self, ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
        """Safely attempt to fetch data using fetch_data_sqlite functions."""
        if not self.api_key:
            print(f"    ✗ No API key available for fetching {ticker}")
            return False
            
        if not FETCH_FUNCTIONS_AVAILABLE:
            print(f"    ✗ fetch_data_sqlite functions not available for {ticker}")
            return False
            
        try:
            # Initialize database schema if needed
            if not self.db_path.exists():
                print(f"    Creating new database: {self.db_path}")
                if get_conn and init_schema:
                    conn = get_conn(self.db_path)
                    init_schema(conn)
                    conn.close()
                else:
                    print(f"    ✗ Cannot initialize database schema")
                    return False
            
            # Attempt to fetch data (force=True to repopulate ATM slices)
            timeframe = self._infer_timeframe()
            fetch_and_save(
                self.api_key,
                ticker,
                start_ts,
                end_ts,
                self.db_path,
                force=True,
                timeframe=timeframe,
            )
            return True
            
        except Exception as e:
            print(f"    ✗ Fetch failed for {ticker}: {e}")
            return False

    def ensure_all_data_available(self, tickers: List[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
        """Ensure all required data is available, using enhanced fetch functionality."""
        if not FETCH_FUNCTIONS_AVAILABLE:
            print("⚠️  Enhanced fetch functions not available - falling back to basic method")
            return self._fallback_data_check(tickers, start_ts, end_ts)
        
        try:
            timeframe = self._infer_timeframe()
            return ensure_data_availability(
                tickers, start_ts, end_ts, self.db_path, auto_fetch=True, timeframe=timeframe
            )
        except Exception as e:
            print(f"❌ Error in enhanced data availability check: {e}")
            return self._fallback_data_check(tickers, start_ts, end_ts)
    
    def _fallback_data_check(self, tickers: List[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
        """Fallback method for checking data availability."""
        missing_count = 0
        for ticker in tickers:
            core = load_ticker_core(ticker, start_ts, end_ts, db_path=self.db_path)
            if core.empty:
                missing_count += 1
                print(f"⚠️  {ticker}: No data available")
        
        if missing_count > 0:
            print(f"⚠️  {missing_count}/{len(tickers)} tickers have missing data")
            return False
        return True
    

    def load_cores_with_fetch(
        self,
        tickers: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        auto_fetch: bool = True,
        drop_zero_iv_ret: bool = True,
        atm_only: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load cores with optional auto-fetch and proper timezone handling."""
        
        # Enhanced timezone handling
        def normalize_timestamp(ts, label=""):
            """Convert any timestamp input to UTC timezone."""
            if isinstance(ts, pd.Timestamp):
                if ts.tz is None:
                    # Naive timestamp, assume UTC
                    return ts.tz_localize("UTC")
                else:
                    # Already has timezone, convert to UTC
                    return ts.tz_convert("UTC")
            elif isinstance(ts, str):
                # String input, parse as UTC
                return pd.Timestamp(ts, tz="UTC")
            else:
                # Try to convert to timestamp first
                try:
                    temp_ts = pd.Timestamp(ts)
                    return temp_ts.tz_localize("UTC")
                except Exception as e:
                    raise ValueError(f"Cannot convert {label} timestamp: {ts}, error: {e}")
        
        try:
            start_ts = normalize_timestamp(start, "start")
            end_ts = normalize_timestamp(end, "end")
        except Exception as e:
            print(f"❌ Timestamp conversion error: {e}")
            return {}
        
        print(f"📊 Loading cores for {len(tickers)} tickers from {start_ts.date()} to {end_ts.date()}")
        
        # Use enhanced data availability checking if auto_fetch is enabled
        if auto_fetch:
            print("🔍 Ensuring all required data is available...")
            if not self.ensure_all_data_available(tickers, start_ts, end_ts):
                print("⚠️  Some data could not be fetched, proceeding with available data...")
        
        # Load available cores
        available_cores = {}
        missing_tickers = []
        
        for ticker in tickers:
            try:
                core = load_ticker_core(ticker, start_ts, end_ts, db_path=self.db_path, atm_only=atm_only)
                if core is not None and not core.empty:
                    available_cores[ticker] = core
                    print(f"  ✅ {ticker}: {len(core):,} rows")
                else:
                    missing_tickers.append(ticker)
                    print(f"  ❌ {ticker}: No data available")
            except Exception as e:
                missing_tickers.append(ticker)
                print(f"  ❌ {ticker}: Error loading - {e}")
        
        # Final fallback: try individual fetching for still-missing tickers
        if auto_fetch and missing_tickers and FETCH_FUNCTIONS_AVAILABLE:
            print(f"🔄 Final attempt: individually fetching {len(missing_tickers)} remaining tickers...")
            
            for ticker in missing_tickers[:]:  # Use slice copy to avoid modification during iteration
                print(f"  Final fetch attempt for {ticker}...")
                if self._safe_fetch_data(ticker, start_ts, end_ts):
                    # Try to load the newly fetched data
                    try:
                        core = load_ticker_core(ticker, start_ts, end_ts, db_path=self.db_path, atm_only=atm_only)
                        if core is not None and not core.empty:
                            available_cores[ticker] = core
                            missing_tickers.remove(ticker)
                            print(f"  ✅ {ticker}: {len(core):,} rows (final fetch successful)")
                    except Exception as e:
                        print(f"  ❌ {ticker}: Still failed after final fetch - {e}")
        
        if not available_cores:
            print("⚠️  No data loaded for any ticker")
        else:
            print(f"✅ Successfully loaded data for {len(available_cores)} tickers")
            if missing_tickers:
                print(f"⚠️  Failed to load data for {len(missing_tickers)} tickers: {missing_tickers}")
        
        return available_cores
    
    def validate_cores_for_analysis(
        self,
        cores: Dict[str, pd.DataFrame],
        analysis_type: str = "general",
        drop_zero_iv_ret: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate and clean core data for analysis.
        
        Parameters
        ----------
        cores : Dict[str, pd.DataFrame]
            Dictionary mapping ticker to core dataframe
        analysis_type : str
            Type of analysis (affects validation criteria)
        drop_zero_iv_ret : bool
            Whether to drop rows with zero IV returns
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Validated cores dictionary
        """
        validated_cores = {}
        
        for ticker, df in cores.items():
            if df.empty:
                print(f"  ⚠️  {ticker}: Empty dataframe, skipping")
                continue
                
            # Basic validation
            required_cols = ["ts_event", "iv", "symbol"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  ❌ {ticker}: Missing required columns: {missing_cols}")
                continue
            
            # Make a copy for validation
            clean_df = df.copy()
            
            # Drop NaN IV values
            initial_rows = len(clean_df)
            clean_df = clean_df.dropna(subset=["iv"])
            if len(clean_df) < initial_rows:
                print(f"  🧹 {ticker}: Dropped {initial_rows - len(clean_df)} NaN IV rows")
            
            # Optional: drop zero IV returns
            if drop_zero_iv_ret and "iv_ret" in clean_df.columns:
                pre_filter = len(clean_df)
                clean_df = clean_df[clean_df["iv_ret"] != 0]
                if len(clean_df) < pre_filter:
                    print(f"  🧹 {ticker}: Dropped {pre_filter - len(clean_df)} zero IV return rows")
            
            # Ensure proper sorting
            if "ts_event" in clean_df.columns:
                clean_df = clean_df.sort_values("ts_event").reset_index(drop=True)
            
            if clean_df.empty:
                print(f"  ❌ {ticker}: No valid data after cleaning")
                continue
                
            validated_cores[ticker] = clean_df
            print(f"  ✅ {ticker}: {len(clean_df):,} valid rows")
        
        return validated_cores


def load_cores_with_auto_fetch(
    tickers: List[str],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    db_path: Path,
    auto_fetch: bool = True,
    drop_zero_iv_ret: bool = True,
    atm_only: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load core data with automatic fetching of missing data.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start : Union[str, pd.Timestamp]
        Start date/timestamp
    end : Union[str, pd.Timestamp]
        End date/timestamp  
    db_path : Path
        Path to database
    auto_fetch : bool
        Whether to auto-fetch missing data
    drop_zero_iv_ret : bool
        Whether to drop zero IV return rows
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping ticker to core dataframe
    """
    coordinator = DataCoordinator(db_path)
    return coordinator.load_cores_with_fetch(tickers, start, end, auto_fetch, drop_zero_iv_ret, atm_only)

def validate_cores(
    cores: Dict[str, pd.DataFrame],
    analysis_type: str = "general",
    drop_zero_iv_ret: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Convenience function for core validation."""
    coordinator = DataCoordinator()
    return coordinator.validate_cores_for_analysis(cores, analysis_type, drop_zero_iv_ret)
