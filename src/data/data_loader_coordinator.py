"""
Data loader coordinator - helps orchestrate data loading across existing modules.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, Sequence, Optional, List, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# Import existing functions with simple fallback
try:
    from src.data.fetch_data_sqlite import (
        fetch_and_save, get_conn, init_schema,
        ensure_data_availability, check_data_exists,
    )
    FETCH_FUNCTIONS_AVAILABLE = True
except ImportError:
    # Simple fallback - try one common path
    try:
        import sys
        sys.path.append(str(Path(__file__).resolve().parents[2] / "data"))
        from fetch_data_sqlite import (
            fetch_and_save, get_conn, init_schema,
            ensure_data_availability, check_data_exists,
        )
        FETCH_FUNCTIONS_AVAILABLE = True
    except ImportError:
        fetch_and_save = get_conn = init_schema = None
        ensure_data_availability = check_data_exists = None
        FETCH_FUNCTIONS_AVAILABLE = False

def _calculate_iv(price: float, S: float, K: float, T: float, cp: str, r: float) -> float:
    """IV calculation using Black-Scholes."""
    if not np.isfinite([price, S, K, T, r]).all() or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    
    intrinsic = max(S - K, 0.0) if cp.upper().startswith('C') else max(K - S, 0.0)
    if price <= intrinsic + 1e-10:
        return 1e-6
        
    def bs_price(sigma):
        if T <= 0 or sigma <= 0:
            return intrinsic
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        if cp.upper().startswith('C'):
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    try:
        return brentq(lambda sig: bs_price(sig) - price, 1e-6, 5.0, maxiter=100)
    except:
        return np.nan

def _normalize_timestamp(ts, label="") -> pd.Timestamp:
    """Convert any timestamp input to UTC timezone."""
    if isinstance(ts, pd.Timestamp):
        return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return pd.Timestamp(ts, tz="UTC")

def _safe_execute(conn: sqlite3.Connection, query: str, params=None) -> bool:
    """Safely execute a query, return success status."""
    try:
        conn.execute(query, params or [])
        return True
    except:
        return False

def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if table exists."""
    try:
        result = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", 
            (table_name,)
        ).fetchone()
        return result is not None
    except:
        return False

def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get column names for a table."""
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]
    except:
        return []

def _populate_atm_slices(conn: sqlite3.Connection, ticker: str) -> None:
    """Populate ATM slices table from processed_merged_1m data."""
    try:
        atm_cols = _get_table_columns(conn, "atm_slices_1m")
        processed_cols = _get_table_columns(conn, "processed_merged_1m")
        
        if not atm_cols or not processed_cols:
            return
        
        common_cols = [col for col in atm_cols if col in processed_cols]
        if len(common_cols) < 10:
            return
        
        cols_str = ", ".join(common_cols)
        query = f"""
        INSERT OR REPLACE INTO atm_slices_1m ({cols_str})
        SELECT {cols_str}
        FROM (
            SELECT {cols_str},
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, ts_event, expiry_date
                       ORDER BY ABS(strike_price - stock_close)
                   ) rn
            FROM processed_merged_1m WHERE ticker = ?
        ) WHERE rn = 1
        """
        
        conn.execute(query, (ticker,))
        conn.commit()
        
    except Exception as e:
        print(f"  ✗ Failed to populate ATM slices for {ticker}: {e}")

def load_ticker_core(ticker: str, start=None, end=None, r=0.045, db_path=None, atm_only: bool = True) -> pd.DataFrame:
    """Load ticker core data with IV calculation."""
    
    if db_path is None:
        db_path = Path(os.getenv("IV_DB_PATH", "IV_DB_PATH"))
    
    if not Path(db_path).exists():
        print(f"Database file does not exist: {db_path}")
        return pd.DataFrame()
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Simple table selection with priority order
            if atm_only:
                table_candidates = ["atm_slices_1m", "processed_merged_1m", "processed_merged"]
            else:
                table_candidates = ["processed_merged_1m", "processed_merged", "merged_1m"]
            
            table = None
            for candidate in table_candidates:
                if _table_exists(conn, candidate):
                    # Quick data check
                    count_query = f"SELECT COUNT(*) FROM {candidate} WHERE ticker=?"
                    try:
                        result = conn.execute(count_query, (ticker,)).fetchone()
                        if result and result[0] > 0:
                            table = candidate
                            break
                    except:
                        continue
            
            if table is None:
                return pd.DataFrame()
            
            # Handle ATM slice population if needed
            if (atm_only and table == "processed_merged_1m" and 
                _table_exists(conn, "atm_slices_1m")):
                _populate_atm_slices(conn, ticker)
                # Try to use ATM table if population succeeded
                try:
                    result = conn.execute("SELECT COUNT(*) FROM atm_slices_1m WHERE ticker=?", (ticker,)).fetchone()
                    if result and result[0] > 0:
                        table = "atm_slices_1m"
                except:
                    pass
            
            # Build query with time filters
            where_clauses, params = ["ticker=?"], [ticker]
            if start:
                where_clauses.append("ts_event >= ?")
                params.append(_normalize_timestamp(start).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            if end:
                where_clauses.append("ts_event <= ?")
                params.append(_normalize_timestamp(end).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            
            # Get available required columns
            required_cols = ["ts_event", "expiry_date", "opt_close", "stock_close", 
                           "opt_volume", "stock_volume", "option_type", "strike_price", 
                           "time_to_expiry"]
            available_cols = _get_table_columns(conn, table)
            select_cols = [col for col in required_cols if col in available_cols]
            
            if len(select_cols) < 8:
                return pd.DataFrame()
            
            cols_str = ", ".join(select_cols)
            where_str = " AND ".join(where_clauses)
            
            # Build appropriate query based on table and ATM preference
            if table == "atm_slices_1m" or not atm_only:
                query = f"SELECT {cols_str} FROM {table} WHERE {where_str} ORDER BY ts_event"
            else:
                # ATM filtering for full surface tables
                query = f"""
                SELECT {cols_str} FROM (
                    SELECT {cols_str},
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker, ts_event, expiry_date
                               ORDER BY ABS(strike_price - stock_close)
                           ) as rn
                    FROM {table} WHERE {where_str}
                ) ranked WHERE rn = 1 ORDER BY ts_event
                """
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["ts_event", "expiry_date"])
            
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df
        
    try:
        # Calculate IV and clean up
        df["iv"] = df.apply(lambda row: _calculate_iv(
            row["opt_close"], row["stock_close"], row["strike_price"], 
            max(row.get("time_to_expiry", 0), 1e-6), row["option_type"], r
        ), axis=1)
        
        # Select final columns
        keep_cols = [col for col in ["ts_event", "expiry_date", "iv", "opt_volume", 
                                   "stock_close", "stock_volume", "time_to_expiry", 
                                   "strike_price", "option_type"] if col in df.columns]
        df = df[keep_cols].copy()
        
        # Final processing
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
        self.db_path = db_path or Path(os.getenv("IV_DB_PATH", "IV_DB_PATH"))
        self.api_key = os.getenv("DATABENTO_API_KEY")

    def _infer_timeframe(self) -> str:
        """Infer timeframe from database path."""
        name = str(self.db_path).lower()
        return "1m" if ("1m" in name or name.endswith("_1m.db")) else "1h"
        
    def _try_fetch_data(self, ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
        """Try to fetch data for a ticker."""
        if not self.api_key or not FETCH_FUNCTIONS_AVAILABLE:
            return False
            
        try:
            # Initialize DB if needed
            if not self.db_path.exists() and get_conn and init_schema:
                conn = get_conn(self.db_path)
                init_schema(conn)
                conn.close()
            
            # Fetch data
            timeframe = self._infer_timeframe()
            fetch_and_save(self.api_key, ticker, start_ts, end_ts, self.db_path, 
                          force=True, timeframe=timeframe)
            return True
            
        except Exception as e:
            print(f"    ✗ Fetch failed for {ticker}: {e}")
            return False

    def load_cores_with_fetch(
        self,
        tickers: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        auto_fetch: bool = True,
        drop_zero_iv_ret: bool = True,
        atm_only: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load cores with optional auto-fetch."""
        
        try:
            start_ts = _normalize_timestamp(start)
            end_ts = _normalize_timestamp(end)
        except Exception as e:
            print(f"❌ Timestamp error: {e}")
            return {}
        
        print(f"📊 Loading cores for {len(tickers)} tickers from {start_ts.date()} to {end_ts.date()}")
        
        # Enhanced data check if available
        if auto_fetch and FETCH_FUNCTIONS_AVAILABLE:
            try:
                timeframe = self._infer_timeframe()
                ensure_data_availability(tickers, start_ts, end_ts, self.db_path, 
                                       auto_fetch=True, timeframe=timeframe)
            except Exception as e:
                print(f"⚠️  Enhanced fetch failed: {e}")
        
        # Load cores
        cores = {}
        failed_tickers = []
        
        for ticker in tickers:
            try:
                core = load_ticker_core(ticker, start_ts, end_ts, db_path=self.db_path, atm_only=atm_only)
                if core is not None and not core.empty:
                    cores[ticker] = core
                    print(f"  ✅ {ticker}: {len(core):,} rows")
                else:
                    failed_tickers.append(ticker)
                    print(f"  ❌ {ticker}: No data")
            except Exception as e:
                failed_tickers.append(ticker)
                print(f"  ❌ {ticker}: Error - {e}")
        
        # Final retry for failed tickers
        if auto_fetch and failed_tickers:
            print(f"🔄 Retrying {len(failed_tickers)} failed tickers...")
            for ticker in failed_tickers[:]:  # Copy to avoid modification during iteration
                if self._try_fetch_data(ticker, start_ts, end_ts):
                    try:
                        core = load_ticker_core(ticker, start_ts, end_ts, db_path=self.db_path, atm_only=atm_only)
                        if core is not None and not core.empty:
                            cores[ticker] = core
                            failed_tickers.remove(ticker)
                            print(f"  ✅ {ticker}: {len(core):,} rows (retry success)")
                    except:
                        pass
        
        if cores:
            print(f"✅ Loaded data for {len(cores)} tickers")
            if failed_tickers:
                print(f"⚠️  Failed: {failed_tickers}")
        else:
            print("⚠️  No data loaded")
        
        return cores

# Public convenience functions (preserve original API)
def load_cores_with_auto_fetch(
    tickers: List[str],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    db_path: Path,
    auto_fetch: bool = True,
    drop_zero_iv_ret: bool = True,
    atm_only: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load core data with automatic fetching of missing data."""
    coordinator = DataCoordinator(db_path)
    return coordinator.load_cores_with_fetch(tickers, start, end, auto_fetch, drop_zero_iv_ret, atm_only)

def validate_cores(
    cores: Dict[str, pd.DataFrame],
    analysis_type: str = "general",
    drop_zero_iv_ret: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Validate and clean core data for analysis."""
    validated_cores = {}
    
    for ticker, df in cores.items():
        if df.empty:
            continue
            
        # Basic validation
        required_cols = ["ts_event", "iv", "symbol"]
        if not all(col in df.columns for col in required_cols):
            print(f"  ❌ {ticker}: Missing required columns")
            continue
        
        # Clean data
        clean_df = df.dropna(subset=["iv"])
        
        if drop_zero_iv_ret and "iv_ret" in clean_df.columns:
            clean_df = clean_df[clean_df["iv_ret"] != 0]
        
        if "ts_event" in clean_df.columns:
            clean_df = clean_df.sort_values("ts_event").reset_index(drop=True)
        
        if not clean_df.empty:
            validated_cores[ticker] = clean_df
            print(f"  ✅ {ticker}: {len(clean_df):,} valid rows")
    
    return validated_cores