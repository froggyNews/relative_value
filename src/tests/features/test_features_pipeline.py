import numpy as np
import pandas as pd

from src.features import (
    _validate_input_data,
    add_all_features,
    build_iv_panel,
    finalize_dataset,
    _normalize_numeric_features,
)


def _sample_core(n=30):
    ts = pd.date_range("2024-01-01 14:30", periods=n, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_event": ts,
            "iv_clip": np.linspace(0.15, 0.25, n),
            "strike_price": 100.0,
            "time_to_expiry": 30.0 / 365.0,
            "option_type": ["C"] * n,
            "stock_close": 100.0,
            "opt_volume": 10,
        }
    )
    return df


def test_validate_input_data_filters_invalid():
    df = _sample_core()
    df.loc[0, "iv_clip"] = -1.0
    df.loc[1, "strike_price"] = -1.0
    df2 = _validate_input_data(df)
    assert len(df2) <= len(df) - 2


def test_add_all_features_basic():
    df = _sample_core()
    out = add_all_features(df)
    # target and key engineered features
    for col in ("iv_ret_fwd", "delta", "gamma", "vega", "days_to_expiry", "option_type_enc"):
        assert col in out.columns
    # stock_close should be dropped by SABR stage
    assert "stock_close" not in out.columns


def test_finalize_dataset_and_normalization():
    df = _sample_core()
    df = add_all_features(df)
    fin = finalize_dataset(df, target_col="iv_ret_fwd", drop_symbol=True, debug=False)
    assert "iv_ret_fwd" in fin.columns
    # leak-prone columns removed
    assert not any(c.startswith("IV_") or c.startswith("IVRET_") for c in fin.columns)
    # attrs should exist after normalization
    assert "norm_means" in fin.attrs and "norm_stds" in fin.attrs
    # ensure normalization helper works idempotently
    fin2 = _normalize_numeric_features(fin.copy(), target_col="iv_ret_fwd")
    assert "norm_means" in fin2.attrs


def test_build_iv_panel_aggregates():
    core_a = _sample_core(20)
    core_b = _sample_core(18)
    cores = {"AAA": core_a, "BBB": core_b}
    panel = build_iv_panel(cores, tolerance="2H", agg="median")
    assert "ts_event" in panel.columns
    assert any(c.startswith("IV_") for c in panel.columns)
    assert any(c.startswith("IVRET_") for c in panel.columns)

