
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional, List

import xgboost as xgb

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
    threshold: float = 0.0
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
    pair_only: bool = False
    strike: Optional[float] = None
    expiry: Optional[str] = None  # YYYY-MM-DD
    strike_tol: float = 0.0

@dataclass
class ModelBundle:
    model: xgb.XGBRegressor
    feature_names: List[str]
