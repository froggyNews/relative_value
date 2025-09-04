# fetch_data_sqlite.py
from __future__ import annotations
import os
from pathlib import Path
import argparse
import sqlite3
from typing import Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Note: databento is imported lazily inside _fetch() to avoid a hard
# dependency when this module is imported for non-fetch operations (e.g., tests).

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _resolve_db_path(db_path: Path | str) -> Path:
    """Resolve database path robustly.

    Resolution order:
    - If `db_path` is provided and absolute, expanduser/vars and use it
    - If `db_path` is relative and exists from CWD, use it
    - If `IV_DB_PATH` env var points to an existing file/dir, use that
    - Otherwise, try relative to the repo root (two levels up from this file)
    - Finally, return the expanded provided path (creating parent dirs later)
    """
    p = Path(str(db_path)).expanduser()
    if p.is_absolute():
        return p
    # CWD relative
    if p.exists():
        return p
    # Environment override
    env_p = os.getenv("IV_DB_PATH")
    if env_p:
        env_path = Path(env_p).expanduser()
        if env_path.exists() or env_path.parent.exists():
            return env_path
    # Try repo root/data
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / p
    if candidate.exists() or candidate.parent.exists():
        return candidate
    return p


def get_conn(db_path: Path | str) -> sqlite3.Connection:
    db_path = _resolve_db_path(db_path)
    _ensure_dir(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    -- 1-hour options data (primary analysis timeframe)
    CREATE TABLE IF NOT EXISTS opra_1h (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL, symbol TEXT,
        PRIMARY KEY (ticker, ts_event, symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_opra_1h_ts ON opra_1h(ticker, ts_event);

    -- 1-hour equity data 
    CREATE TABLE IF NOT EXISTS equity_1h (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL, symbol TEXT,
        PRIMARY KEY (ticker, ts_event)
    );
    CREATE INDEX IF NOT EXISTS idx_equity_1h_ts ON equity_1h(ticker, ts_event);

    -- Daily equity data for longer-term context
    CREATE TABLE IF NOT EXISTS equity_1d (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL,
        PRIMARY KEY (ticker, ts_event)
    );
    CREATE INDEX IF NOT EXISTS idx_equity_1d_ts ON equity_1d(ticker, ts_event);

    -- Merged 1-hour options and equity data
    CREATE TABLE IF NOT EXISTS merged_1h (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        opt_symbol TEXT, stock_symbol TEXT,
        opt_close REAL, stock_close REAL,
        opt_volume REAL, stock_volume REAL,
        PRIMARY KEY (ticker, ts_event, opt_symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_merged_1h_ts ON merged_1h(ticker, ts_event);

    -- Processed merged data with option characteristics
    CREATE TABLE IF NOT EXISTS processed_merged_1h (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        opt_symbol TEXT, stock_symbol TEXT,
        opt_close REAL, stock_close REAL,
        opt_volume REAL, stock_volume REAL,
        expiry_date TEXT, option_type TEXT,
        strike_price REAL, time_to_expiry REAL, moneyness REAL,
        PRIMARY KEY (ticker, ts_event, opt_symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_processed_1h_ts  ON processed_merged_1h(ticker, ts_event);
    CREATE INDEX IF NOT EXISTS idx_processed_1h_exp ON processed_merged_1h(ticker, expiry_date);

    -- ATM slices for 1-hour analysis
    CREATE TABLE IF NOT EXISTS atm_slices_1h (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        expiry_date TEXT NOT NULL,
        opt_symbol TEXT, stock_symbol TEXT,
        opt_close REAL, stock_close REAL,
        opt_volume REAL, stock_volume REAL,
        option_type TEXT, strike_price REAL,
        time_to_expiry REAL, moneyness REAL,
        PRIMARY KEY (ticker, ts_event, expiry_date)
    );
    CREATE INDEX IF NOT EXISTS idx_atm_1h_ts  ON atm_slices_1h(ticker, ts_event);
    CREATE INDEX IF NOT EXISTS idx_atm_1h_exp ON atm_slices_1h(ticker, expiry_date);
    
    -- Legacy 1-minute tables (kept for backward compatibility)
    CREATE TABLE IF NOT EXISTS opra_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL, symbol TEXT,
        PRIMARY KEY (ticker, ts_event, symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_opra_1m_ts ON opra_1m(ticker, ts_event);

    CREATE TABLE IF NOT EXISTS equity_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL, symbol TEXT,
        PRIMARY KEY (ticker, ts_event)
    );
    CREATE INDEX IF NOT EXISTS idx_equity_1m_ts ON equity_1m(ticker, ts_event);

    CREATE TABLE IF NOT EXISTS merged_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        opt_symbol TEXT, stock_symbol TEXT,
        opt_close REAL, stock_close REAL,
        opt_volume REAL, stock_volume REAL,
        PRIMARY KEY (ticker, ts_event, opt_symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_merged_1m_ts ON merged_1m(ticker, ts_event);

    CREATE TABLE IF NOT EXISTS processed_merged_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        opt_symbol TEXT, stock_symbol TEXT,
        opt_close REAL, stock_close REAL,
        opt_volume REAL, stock_volume REAL,
        expiry_date TEXT, option_type TEXT,
        strike_price REAL, time_to_expiry REAL, moneyness REAL,
        PRIMARY KEY (ticker, ts_event, opt_symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_processed_1m_ts  ON processed_merged_1m(ticker, ts_event);
    CREATE INDEX IF NOT EXISTS idx_processed_1m_exp ON processed_merged_1m(ticker, expiry_date);

    CREATE TABLE IF NOT EXISTS atm_slices_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        expiry_date TEXT NOT NULL,
        opt_symbol TEXT, stock_symbol TEXT,
        opt_close REAL, stock_close REAL,
        opt_volume REAL, stock_volume REAL,
        option_type TEXT, strike_price REAL,
        time_to_expiry REAL, moneyness REAL,
        PRIMARY KEY (ticker, ts_event, expiry_date)
    );
    CREATE INDEX IF NOT EXISTS idx_atm_1m_ts  ON atm_slices_1m(ticker, ts_event);
    CREATE INDEX IF NOT EXISTS idx_atm_1m_exp ON atm_slices_1m(ticker, expiry_date);
    """)
    conn.commit()

def _iso_utc(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, utc=True, errors="coerce")
    return s.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def _upsert(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    if df.empty: return
    tmp = f"tmp_{table}"
    df.to_sql(tmp, conn, if_exists="replace", index=False)
    cols = ",".join(df.columns)
    conn.execute(f"INSERT OR IGNORE INTO {table} ({cols}) SELECT {cols} FROM {tmp};")
    conn.execute(f"DROP TABLE {tmp};")
    conn.commit()

def _calculate_sigma_realized(bars: pd.DataFrame, tz: str = "America/New_York") -> float:
    if bars is None or bars.empty: return np.nan
    df = bars[["ts_event","close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(inplace=True)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df.dropna(subset=["ts_event"], inplace=True)
    df["ts_local"] = df["ts_event"].dt.tz_convert(tz)
    delta = df["ts_event"].sort_values().diff().median()
    is_hourly = pd.notna(delta) and delta >= pd.Timedelta("30min")
    if is_hourly:
        mask = ((df["ts_local"].dt.minute == 0) &
                (df["ts_local"].dt.hour >= 10) &
                (df["ts_local"].dt.hour <= 15))
        min_obs = 3
    else:
        h, m = df["ts_local"].dt.hour, df["ts_local"].dt.minute
        mask = (((h > 9) | ((h == 9) & (m >= 30))) & (h < 16))
        min_obs = 100
    df = df.loc[mask].sort_values("ts_event")
    if df.empty: return np.nan
    df["trade_date"] = df["ts_local"].dt.date
    df["logp"] = np.log(df["close"])
    df["ret"] = df.groupby("trade_date")["logp"].diff()
    rv = df.groupby("trade_date")["ret"].apply(lambda x: np.nansum(x.values**2))
    obs = df.groupby("trade_date")["ret"].apply(lambda x: np.isfinite(x.values).sum())
    rv = rv.loc[obs[obs >= min_obs].index]
    if rv.empty or not np.isfinite(rv).any(): return np.nan
    sig = float(np.sqrt(rv.mean() * 252.0))
    return sig if np.isfinite(sig) else np.nan

def _fetch(API_KEY: str, start: pd.Timestamp, end: pd.Timestamp, ticker: str
           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch 1-hour options and equity data for analysis.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        opra_1h, eq_1h, eq_1d (1-hour options, 1-hour equity, daily equity)
    """
    # Lazily import databento to avoid hard dependency during module import
    try:
        import databento as db  # type: ignore
    except Exception as e:
        raise ImportError(
            "databento is required for fetching data. Install it or set up the project venv."
        ) from e
    client = db.Historical(API_KEY)
    opt_symbol = f"{ticker}.opt"
    
    # Fetch 1-hour options data (primary timeframe)
    opra_1h = client.timeseries.get_range(
        dataset="OPRA.PILLAR", stype_in="parent", symbols=[opt_symbol],
        schema="OHLCV-1H", start=start, end=end
    ).to_df().reset_index()
    
    # Fetch 1-hour equity data
    eq_1h = client.timeseries.get_range(
        dataset="XNAS.ITCH", symbols=[ticker],
        schema="OHLCV-1H", start=start, end=end
    ).to_df().reset_index()
    
    # Fetch daily equity data for longer-term context (2 years of history)
    eq_1d = client.timeseries.get_range(
        dataset="XNAS.ITCH", symbols=[ticker],
        schema="OHLCV-1D", start=start - pd.DateOffset(years=2), end=end
    ).to_df().reset_index()
    
    # Ensure datetime columns are properly formatted
    for df in (opra_1h, eq_1h, eq_1d):
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    
    return opra_1h, eq_1h, eq_1d

def _populate_atm(conn: sqlite3.Connection, ticker: str, timeframe: str = "1h") -> None:
    """Populate ATM slices table for specified timeframe."""
    
    if timeframe == "1h":
        source_table = "processed_merged_1h"
        target_table = "atm_slices_1h"
    else:
        source_table = "processed_merged_1m"
        target_table = "atm_slices_1m"
    
    q = f"""
    INSERT OR REPLACE INTO {target_table} (
        ticker, ts_event, expiry_date, opt_symbol, stock_symbol,
        opt_close, stock_close, opt_volume, stock_volume,
        option_type, strike_price, time_to_expiry, moneyness
    )
    SELECT 
        pm.ticker, pm.ts_event, pm.expiry_date, pm.opt_symbol, pm.stock_symbol,
        pm.opt_close, pm.stock_close, pm.opt_volume, pm.stock_volume,
        pm.option_type, pm.strike_price, pm.time_to_expiry, pm.moneyness
    FROM (
        SELECT 
            pm.ticker, pm.ts_event, pm.expiry_date, pm.opt_symbol, pm.stock_symbol,
            pm.opt_close, pm.stock_close, pm.opt_volume, pm.stock_volume,
            pm.option_type, pm.strike_price, pm.time_to_expiry, pm.moneyness,
            ROW_NUMBER() OVER (
              PARTITION BY pm.ticker, pm.ts_event, pm.expiry_date
              ORDER BY ABS(pm.strike_price - pm.stock_close)
            ) rn
        FROM {source_table} pm
        WHERE pm.ticker = ?
    ) pm
    WHERE pm.rn = 1;
    """
    conn.execute(q, (ticker,))
    conn.commit()

def check_data_exists(conn: sqlite3.Connection, ticker: str, start: pd.Timestamp, end: pd.Timestamp, 
                     timeframe: str = "1h") -> bool:
    """Check if data exists for a ticker in the specified time window and timeframe."""
    try:
        # Define tables to check based on timeframe
        if timeframe == "1h":
            tables_to_check = ["atm_slices_1h", "processed_merged_1h", "merged_1h"]
        else:
            tables_to_check = ["atm_slices_1m", "processed_merged_1m", "merged_1m"]
        
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        for table in tables_to_check:
            # Check if table exists
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", 
                (table,)
            ).fetchone()
            
            if table_exists:
                # Check if data exists for ticker in time window
                q = f"""SELECT COUNT(1) FROM {table}
                       WHERE ticker=? AND ts_event >= ? AND ts_event <= ?"""
                count = conn.execute(q, (ticker, start_str, end_str)).fetchone()[0]
                if count > 0:
                    print(f"[EXISTS] {ticker} found in {table} ({count} rows, {timeframe})")
                    return True
        
        print(f"[MISSING] {ticker} not found in any {timeframe} table for window {start.date()} to {end.date()}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Checking data existence for {ticker}: {e}")
        return False


def preprocess_and_store(API_KEY: str, start: pd.Timestamp, end: pd.Timestamp,
                         ticker: str, conn: sqlite3.Connection, force: bool=False, 
                         timeframe: str = "1h") -> None:
    """
    Preprocess and store options and equity data.
    
    Parameters
    ----------
    timeframe : str
        '1h' for 1-hour analysis (default), '1m' for 1-minute analysis
    """
    # Check if we already have this data (skip only if exists AND not forced)
    try:
        has_data = check_data_exists(conn, ticker, start, end, timeframe)
    except Exception:
        has_data = False
    if has_data and not force:
        print(
            f"[SKIP] {ticker} already has {timeframe} data for window {start.date()} to {end.date()}"
        )
        return

    opra_1h, eq_1h, eq_1d = _fetch(API_KEY, start, end, ticker)

    # Store raw 1-hour data
    opra_db = opra_1h[["ts_event","open","high","low","close","volume","symbol"]].copy()
    opra_db.insert(0,"ticker",ticker); opra_db["ts_event"] = _iso_utc(opra_db["ts_event"])
    _upsert(conn, opra_db, "opra_1h")

    eq1h_db = eq_1h[["ts_event","open","high","low","close","volume","symbol"]].copy()
    eq1h_db.insert(0,"ticker",ticker); eq1h_db["ts_event"] = _iso_utc(eq1h_db["ts_event"])
    _upsert(conn, eq1h_db, "equity_1h")

    eq1d_db = eq_1d[["ts_event","open","high","low","close","volume"]].copy()
    eq1d_db.insert(0,"ticker",ticker); eq1d_db["ts_event"] = _iso_utc(eq1d_db["ts_event"])
    _upsert(conn, eq1d_db, "equity_1d")

    # Merge 1-hour options and equity data
    # For 1-hour data, use standard market hours (9:30 AM - 4:00 PM ET = 14:30-21:00 UTC)
    merged = pd.merge(
        opra_1h.rename(columns={"close":"opt_close","volume":"opt_volume","symbol":"opt_symbol"}),
        eq_1h.rename(columns={"close":"stock_close","volume":"stock_volume","symbol":"stock_symbol"}),
        on="ts_event", how="inner"
    )
    merged = merged.copy()
    
    # For 1-hour data, filter to market hours (9:30 AM - 4:00 PM ET)
    # Convert to ET for filtering, then back to UTC
    merged_et = merged.copy()
    merged_et["ts_et"] = merged_et["ts_event"].dt.tz_convert("US/Eastern")
    market_hours_mask = (
        (merged_et["ts_et"].dt.hour >= 10) |  # 10 AM ET and later
        ((merged_et["ts_et"].dt.hour == 9) & (merged_et["ts_et"].dt.minute >= 30))  # 9:30 AM ET
    ) & (merged_et["ts_et"].dt.hour <= 15)  # Before 4 PM ET
    
    merged = merged[market_hours_mask]

    # Parse option symbols: YYMMDD [C|P] ########
    ex = merged["opt_symbol"].astype(str).str.extract(r"(\d{6})([CP])(\d{8})")
    merged["expiry_date"]  = pd.to_datetime(ex[0], format="%y%m%d", utc=True, errors="coerce")
    merged["option_type"]  = ex[1]
    merged["strike_price"] = pd.to_numeric(ex[2], errors="coerce") / 1000.0
    merged["time_to_expiry"] = ((merged["expiry_date"] - merged["ts_event"]).dt.total_seconds()
                                /(365*24*3600)).clip(lower=0.0)
    merged["moneyness"] = np.where(
        merged["option_type"].eq("C"),
        merged["stock_close"] - merged["strike_price"],
        np.where(merged["option_type"].eq("P"),
                 merged["strike_price"] - merged["stock_close"], np.nan)
    )
    merged = merged.dropna(subset=["expiry_date","strike_price","option_type",
                                   "opt_close","stock_close","time_to_expiry"])

    # Store merged and processed 1-hour data
    m_db = merged[["ts_event","opt_symbol","stock_symbol","opt_close","stock_close","opt_volume","stock_volume"]].copy()
    m_db.insert(0,"ticker",ticker); m_db["ts_event"] = _iso_utc(m_db["ts_event"])
    _upsert(conn, m_db, "merged_1h")

    p_db = merged[["ts_event","opt_symbol","stock_symbol","opt_close","stock_close","opt_volume","stock_volume",
                   "expiry_date","option_type","strike_price","time_to_expiry","moneyness"]].copy()
    p_db.insert(0,"ticker",ticker)
    p_db["ts_event"]   = _iso_utc(p_db["ts_event"])
    p_db["expiry_date"]= _iso_utc(p_db["expiry_date"])
    _upsert(conn, p_db, "processed_merged_1h")

    # Populate ATM slices for 1-hour data
    _populate_atm(conn, ticker, timeframe="1h")

    # Calculate realized volatility from daily data for better estimates
    sigma = _calculate_sigma_realized(eq_1d)
    print(f"[DONE] {ticker}: {len(p_db)} 1h rows, sigma_annual≈{sigma:.4f}" if np.isfinite(sigma) else
          f"[DONE] {ticker}: {len(p_db)} 1h rows, sigma_annual≈nan")

def fetch_and_save(API_KEY: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                   db_path: Path | str, force: bool=False, timeframe: str = "1h") -> Path:
    """Fetch and save data for specified timeframe."""
    db_path = _resolve_db_path(db_path)
    conn = get_conn(db_path)
    init_schema(conn)
    preprocess_and_store(API_KEY, start, end, ticker, conn, force=force, timeframe=timeframe)
    conn.close()
    return db_path


def auto_fetch_missing_data(tickers: list, start: pd.Timestamp, end: pd.Timestamp, 
                           db_path: Path | str, API_KEY: str = None, timeframe: str = "1h") -> dict:
    """
    Automatically check for missing data and fetch it if needed.
    
    Parameters
    ----------
    tickers : list
        List of ticker symbols to check
    start : pd.Timestamp
        Start date for data window
    end : pd.Timestamp
        End date for data window
    db_path : Path
        Path to SQLite database
    API_KEY : str, optional
        Databento API key (will try to get from environment if not provided)
        
    Returns
    -------
    dict
        Summary of fetch results: {"fetched": [...], "skipped": [...], "failed": [...]}
    """
    if API_KEY is None:
        import os
        API_KEY = os.getenv("DATABENTO_API_KEY")
        if not API_KEY:
            print("❌ No DATABENTO_API_KEY found in environment")
            return {"fetched": [], "skipped": [], "failed": list(tickers)}
    
    results = {"fetched": [], "skipped": [], "failed": []}
    
    # Ensure database exists and is initialized
    db_path = _resolve_db_path(db_path)
    if not db_path.exists():
        print(f"📁 Creating new database: {db_path}")
        conn = get_conn(db_path)
        init_schema(conn)
        conn.close()
    
    print(f"🔍 Checking data availability for {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            conn = get_conn(db_path)
            
            # Check if data exists
            if check_data_exists(conn, ticker, start, end, timeframe):
                results["skipped"].append(ticker)
                conn.close()
                continue
                
            conn.close()
            
            # Data missing - attempt to fetch
            print(f"📥 Fetching missing {timeframe} data for {ticker}...")
            try:
                fetch_and_save(API_KEY, ticker, start, end, db_path, force=False, timeframe=timeframe)
                results["fetched"].append(ticker)
                print(f"  ✅ Successfully fetched {ticker} ({timeframe})")
            except Exception as fetch_error:
                print(f"  ❌ Failed to fetch {ticker}: {fetch_error}")
                results["failed"].append(ticker)
                
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            results["failed"].append(ticker)
    
    print(f"📊 Fetch summary: {len(results['fetched'])} fetched, "
          f"{len(results['skipped'])} skipped, {len(results['failed'])} failed")
    
    return results


def ensure_data_availability(tickers: list, start: pd.Timestamp, end: pd.Timestamp,
                            db_path: Path | str, auto_fetch: bool = True, timeframe: str = "1h") -> bool:
    """
    Ensure data is available for all tickers, fetching if needed.
    
    Parameters
    ----------
    tickers : list
        List of ticker symbols
    start : pd.Timestamp
        Start date
    end : pd.Timestamp
        End date
    db_path : Path
        Database path
    auto_fetch : bool
        Whether to automatically fetch missing data
        
    Returns
    -------
    bool
        True if all data is available, False otherwise
    """
    if not auto_fetch:
        print("⚠️  Auto-fetch disabled - data availability not guaranteed")
        return True
        
    try:
        results = auto_fetch_missing_data(tickers, start, end, db_path, timeframe=timeframe)
        
        # Check if we have any failures
        if results["failed"]:
            print(f"⚠️  Some {timeframe} data could not be fetched: {results['failed']}")
            return False
        
        print(f"✅ All required {timeframe} data is available")
        return True
        
    except Exception as e:
        print(f"❌ Error ensuring {timeframe} data availability: {e}")
        return False

def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Fetch options and equity data for analysis")
    ap.add_argument("--db", required=False, type=Path, default=Path(os.getenv("IV_DB_PATH", "data/iv_data_1h.db")), help="Database path (or set IV_DB_PATH)")
    ap.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    ap.add_argument("--start", required=True, help="Start date")
    ap.add_argument("--end", required=True, help="End date")
    ap.add_argument("--force", action="store_true", help="Force re-download")
    ap.add_argument("--timeframe", choices=["1h", "1m"], default="1h", 
                    help="Analysis timeframe (default: 1h)")
    args = ap.parse_args()

    API_KEY = os.getenv("DATABENTO_API_KEY")
    if not API_KEY:
        raise ValueError("Missing DATABENTO_API_KEY")

    start = pd.Timestamp(args.start, tz="UTC")
    end   = pd.Timestamp(args.end, tz="UTC")
    
    print(f"📊 Fetching {args.timeframe} data for {len(args.tickers)} tickers")
    for t in args.tickers:
        print(f"[DL] {t}  {start.date()} → {end.date()} ({args.timeframe})")
        fetch_and_save(API_KEY, t, start, end, db_path=args.db, force=args.force, timeframe=args.timeframe)
    print(f"[OK] SQLite: {args.db.resolve()}")

if __name__ == "__main__":
    main()
