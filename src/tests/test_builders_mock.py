# tests/test_builders_mock.py
import numpy as np
import pandas as pd
import pytest

import feature_engineering as m

# ---------------------------
# Synthetic core builders
# ---------------------------

def _mk_ts(n=60, start="2025-08-01 14:00:00Z", freq="1h"):
    return pd.date_range(start=pd.to_datetime(start), periods=n, freq=freq, tz="UTC")

def _core_frame(ticker: str, n=60, freq="1h") -> pd.DataFrame:
    ts = _mk_ts(n=n, freq=freq)
    # underlying
    S = 20.0 + 0.05 * np.arange(n) + 0.2 * np.sin(np.linspace(0, 3*np.pi, n))
    # iv (clipped)
    iv = 0.4 + 0.02 * np.sin(np.linspace(0, 2*np.pi, n))
    # strikes near S
    K = S * (1.0 + 0.02 * np.sin(np.linspace(0, 4*np.pi, n)))
    # time to expiry (in years)
    T = np.linspace(20/365, 200/365, n)
    # option volume
    vol = (100 + 10*np.random.RandomState(0).randn(n)).clip(1, None)
    # call/put mix
    opt_types = np.where((np.arange(n) % 2) == 0, "C", "P")

    df = pd.DataFrame({
        "ts_event": ts,
        "symbol": ticker,
        "stock_close": S,
        "iv_clip": np.clip(iv, 1e-4, None),
        "strike_price": np.clip(K, 1e-4, None),
        "time_to_expiry": np.clip(T, 1e-6, None),
        "option_type": opt_types,
        "opt_volume": vol,
    })
    return df

def _mk_cores(tickers=("QBTS","IONQ","RGTI","QUBT"), n=80) -> dict:
    return {t: _core_frame(t, n=n) for t in tickers}


# ---------------------------
# Unit-style tests (mock cores)
# ---------------------------

@pytest.mark.unit
def test_add_all_features_includes_core_and_sabr_fields():
    df = _core_frame("QBTS", n=64)
    out = m.add_all_features(df, forward_steps=1, r=0.045, validate=True)
    # Target columns
    assert "iv_ret_fwd" in out.columns
    assert "iv_ret_fwd_abs" in out.columns
    # Time features
    for c in ("hour", "minute", "day_of_week", "days_to_expiry", "option_type_enc"):
        assert c in out.columns
    # Greeks
    for c in ("delta", "gamma", "vega"):
        assert c in out.columns
    # IV momentum/stat features (1h)
    for c in ("iv_ret_1h","iv_ret_3h","iv_ret_6h","iv_sma_3h","iv_sma_6h","iv_std_6h","iv_rsi_6h","iv_zscore_6h"):
        assert c in out.columns
    # Volume features
    for c in ("opt_vol_change_1h","opt_vol_roll_3h","opt_vol_roll_6h","opt_vol_roll_24h","opt_vol_zscore_6h"):
        assert c in out.columns
    # SABR-derived
    for c in ("sabr_alpha","sabr_beta","sabr_rho","sabr_nu","moneyness","log_moneyness"):
        assert c in out.columns
    # _add_sabr_features drops stock_close at the end
    assert "stock_close" not in out.columns

@pytest.mark.unit
def test_build_iv_panel_shapes_and_columns():
    cores = _mk_cores(n=72)
    panel = m.build_iv_panel(cores, tolerance="15s", agg="median")
    assert isinstance(panel, pd.DataFrame)
    assert not panel.empty
    # Should have ts_event plus IV_*/IVRET_* per ticker (at least some)
    assert "ts_event" in panel.columns
    iv_cols = [c for c in panel.columns if c.startswith("IV_")]
    ivret_cols = [c for c in panel.columns if c.startswith("IVRET_")]
    assert len(iv_cols) >= 1
    assert len(ivret_cols) >= 1

@pytest.mark.unit
def test_build_pooled_iv_return_dataset_time_safe_on_mock_cores():
    tickers = ["QBTS","IONQ","RGTI","QUBT"]
    cores = _mk_cores(tickers, n=96)
    pooled = m.build_pooled_iv_return_dataset_time_safe(
        tickers=tickers,
        r=0.045,
        forward_steps=1,
        tolerance="15s",
        cores=cores,
        debug=False,
    )
    assert isinstance(pooled, pd.DataFrame)
    assert not pooled.empty
    # Must keep the label, panel columns, and one-hots
    assert "iv_ret_fwd" in pooled.columns
    onehots = [f"sym_{t}" for t in tickers]
    for c in onehots:
        assert c in pooled.columns
    # Normalization attrs should exist
    assert "norm_means" in pooled.attrs and "norm_stds" in pooled.attrs

@pytest.mark.unit
def test_build_iv_return_dataset_time_safe_per_ticker_on_mock_cores():
    tickers = ["QBTS","IONQ"]
    cores = _mk_cores(tickers, n=72)
    datasets = m.build_iv_return_dataset_time_safe(
        tickers=tickers,
        r=0.045,
        forward_steps=3,
        tolerance="15s",
        cores=cores,
        debug=False,
    )
    assert isinstance(datasets, dict)
    assert set(datasets.keys()) == set(tickers)
    for t, df in datasets.items():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Per-ticker finalize drops peer IV/IVRET leak columns
        assert all(not c.startswith("IV_") for c in df.columns if c != "ts_event")
        assert all(not c.startswith("IVRET_") for c in df.columns if c != "ts_event")
        # Target present
        assert "iv_ret_fwd" in df.columns
        # Normalization attrs
        assert "norm_means" in df.attrs and "norm_stds" in df.attrs

@pytest.mark.unit
def test_build_target_peer_dataset_mock():
    tickers = ["QBTS","IONQ","RGTI"]
    target = "IONQ"
    cores = _mk_cores(tickers, n=64)
    ds = m.build_target_peer_dataset(
        target=target,
        tickers=tickers,
        r=0.045,
        forward_steps=1,
        tolerance="15s",
        cores=cores,
        target_kind="iv_ret",   # maps to 'iv_ret_fwd'
        debug=False,
    )
    assert isinstance(ds, pd.DataFrame)
    assert not ds.empty
    assert "y" in ds.columns   # renamed target
    # Should not include symbol after finalize for target-peer
    assert "symbol" not in ds.columns
    # No raw IV_ peer columns
    assert all(not c.startswith("IV_") for c in ds.columns if c != "ts_event")
    assert all(not c.startswith("IVRET_") for c in ds.columns if c != "ts_event")
