
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.modeling.schemas import BacktestConfig
from src.modeling.trainer import train_xgb_pooled
from src.modeling.predict import make_prediction_frame
from src.backtest.split import build_and_split_pooled
from src.backtest.trades import make_relative_value_trades, summarize_trades

def run_backtest(cfg: BacktestConfig) -> Dict[str, object]:
    # Build + split
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, strike_tr, strike_te, expiry_tr, expiry_te = build_and_split_pooled(cfg)

    # Train pooled model (time-respecting)
    bundle = train_xgb_pooled(X_tr, y_tr)

    # Predictions
    preds = make_prediction_frame(bundle.model, X_te, y_te, ts_te, strike_te, expiry_te)

    # Trades
    trades = make_relative_value_trades(
        preds,
        top_k=cfg.top_k,
        threshold=cfg.threshold,
        min_roll_trades=cfg.min_roll_trades,
        roll_window=cfg.roll_window,
        roll_by_rows=cfg.roll_by_rows,
        roll_bars=cfg.roll_bars,
        pair_only=cfg.pair_only,
        strike=cfg.strike,
        expiry=cfg.expiry,
        strike_tol=cfg.strike_tol,
        group_freq=cfg.group_freq,
    )
    summary = summarize_trades(trades)

    # Optional save
    if cfg.save_trades_csv:
        out_path = Path(cfg.save_trades_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out_path, index=False)

    return {"summary": summary, "trades": trades, "preds_head": preds.head(5)}
