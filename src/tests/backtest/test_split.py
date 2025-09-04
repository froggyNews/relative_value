from pathlib import Path
import pandas as pd
import numpy as np



def _make_pooled_df(n=200):
    # Create a minimal pooled dataset consistent with split expectations
    ts = pd.date_range("2024-01-01 14:30", periods=n, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_event": ts,
            "iv_ret_fwd": np.linspace(-0.01, 0.01, n),
            "core_iv_ret_fwd_abs": np.linspace(0.0, 0.02, n),
            "sym_AAA": 1.0,
            "sym_BBB": 0.0,
        }
    )
    return df


def test_build_and_split_pooled_monkeypatched(monkeypatch, tmp_path: Path):
    # Stub xgboost before importing modules that require it indirectly
    import sys, types
    # Provide a minimal stub so importing schemas doesn't crash
    stub = types.ModuleType("xgboost")
    stub.XGBRegressor = object  # not used in this test path
    sys.modules.setdefault("xgboost", stub)

    from src.backtest import split as bt_split
    # Remove stub so other tests (which may importorskip xgboost) behave correctly
    sys.modules.pop("xgboost", None)
    # Monkeypatch load_cores_with_auto_fetch to avoid I/O
    def fake_load_cores(tickers, start, end, db_path, auto_fetch=True, drop_zero_iv_ret=True, atm_only=True):
        # Not used directly since we also monkeypatch the dataset builder
        return {t: pd.DataFrame() for t in tickers}

    monkeypatch.setattr(bt_split, "load_cores_with_auto_fetch", fake_load_cores)

    # Monkeypatch pooled dataset builder to return our synthetic pooled df
    # Monkeypatch the function as imported inside bt_split
    monkeypatch.setattr(bt_split, "build_pooled_iv_return_dataset_time_safe", lambda *a, **k: _make_pooled_df(240))

    class Cfg:
        tickers = ["AAA", "BBB"]
        start = "2024-01-01"
        end = "2024-02-01"
        r = 0.045
        forward_steps = 1
        tolerance = "15s"
        test_frac = 0.2
        db_path = str(tmp_path / "db.sqlite")
        auto_fetch = False
        atm_only = True

    cfg = Cfg()

    X_tr, X_te, y_tr, y_te, ts_tr, ts_te, *_ = bt_split.build_and_split_pooled(cfg)

    # Basic sanity checks on split sizes and ordering
    assert len(X_tr) > 10 and len(X_te) > 10
    assert ts_tr.iloc[-1] <= ts_te.iloc[0]
