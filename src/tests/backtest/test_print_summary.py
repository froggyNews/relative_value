from __future__ import annotations

from pathlib import Path
import types
import sys
import numpy as np
import pandas as pd


def test_print_summary_sanity(monkeypatch):
    """Smoke test that prints a Summary line from run_backtest.

    Uses lightweight module stubs to avoid heavy deps and I/O, and ensures
    sys.modules is restored after the test via monkeypatch.
    """
    # Stub xgboost to avoid binary import
    monkeypatch.setitem(sys.modules, "xgboost", types.ModuleType("xgboost"))

    # Stub schemas with a minimal BacktestConfig
    schemas_stub = types.ModuleType("src.modeling.schemas")

    class BacktestConfig:  # minimal placeholder
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    schemas_stub.BacktestConfig = BacktestConfig
    monkeypatch.setitem(sys.modules, "src.modeling.schemas", schemas_stub)

    # Stub trainer with a no-op predictor and ModelBundle
    trainer_stub = types.ModuleType("src.modeling.trainer")

    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    class ModelBundle:
        def __init__(self, model, feature_names):
            self.model = model
            self.feature_names = feature_names

    def train_xgb_pooled(X_tr, y_tr):
        return ModelBundle(DummyModel(), list(X_tr.columns))

    trainer_stub.ModelBundle = ModelBundle
    trainer_stub.train_xgb_pooled = train_xgb_pooled
    monkeypatch.setitem(sys.modules, "src.modeling.trainer", trainer_stub)

    # Stub split to return small, deterministic splits
    split_stub = types.ModuleType("src.backtest.split")
    n = 10
    X = pd.DataFrame({"a": np.linspace(0, 1, n), "b": np.linspace(1, 0, n)})
    y = pd.Series(np.linspace(-0.1, 0.1, n))
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    strike = pd.Series(np.full(n, 100.0))
    expiry = pd.Series(pd.Timestamp("2024-02-01", tz="UTC")).repeat(n).reset_index(drop=True)

    def build_and_split_pooled(cfg):
        return X, X, y, y, ts, ts, strike, strike, expiry, expiry

    split_stub.build_and_split_pooled = build_and_split_pooled
    monkeypatch.setitem(sys.modules, "src.backtest.split", split_stub)

    # Import and run
    from src.backtest.run import run_backtest
    from src.modeling.schemas import BacktestConfig as Cfg

    cfg = Cfg(
        tickers=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-01-02",
        auto_fetch=False,
        # Trading params with defaults expected by run_backtest/make_relative_value_trades
        top_k=1,
        threshold=0.0,
        group_freq="1min",
        min_roll_trades=0.0,
        roll_window="30min",
        roll_by_rows=True,
        roll_bars=30,
        pair_only=False,
        strike=None,
        expiry=None,
        strike_tol=0.0,
        granularity="slice",
        weighting_scheme="opt_volume",
        dte_range=(20, 45),
        moneyness_range=None,
        save_trades_csv=None,
    )
    res = run_backtest(cfg)
    print("Summary:", res.get("summary"))
    assert "summary" in res
