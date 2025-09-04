import numpy as np
import pandas as pd
import pytest

# Robustly skip if xgboost cannot be imported or initialized in this env
try:
    import xgboost as _xgb  # noqa: F401
except Exception as _e:  # includes XGBoostError during DLL load
    pytest.skip(f"xgboost unavailable: {_e}", allow_module_level=True)


def _tiny_model_with_cols():
    import xgboost as xgb
    from xgboost import XGBRegressor
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series([0.0, 0.1, 0.2, 0.3])
    m = XGBRegressor(n_estimators=10, max_depth=2, subsample=1.0, colsample_bytree=1.0, random_state=0)
    m.fit(X, y)
    return m


def test_aligns_missing_expected_columns_and_order():
    from types import SimpleNamespace
    from src.modeling.predict import _align_columns_to_model

    # Dummy model with expected feature names (no setter required)
    m = SimpleNamespace(feature_names_in_=np.array(["a", "b", "c"]))

    X_te = pd.DataFrame({"b": [1.0, 2.0], "a": [0.0, 0.0]})
    X_aligned = _align_columns_to_model(m, X_te.copy())

    # Column added with zeros and order matches model expectation
    assert list(X_aligned.columns) == ["a", "b", "c"]
    assert (X_aligned["c"] == 0.0).all()


def test_generic_feature_names_are_left_unchanged():
    from types import SimpleNamespace
    from src.modeling.predict import _align_columns_to_model

    # Generic names imply no reindexing should be applied
    m = SimpleNamespace(feature_names_in_=np.array(["f0", "f1"]))

    X_te = pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4]})
    X_aligned = _align_columns_to_model(m, X_te.copy())
    assert list(X_aligned.columns) == list(X_te.columns)


def test_make_prediction_frame_optional_fields_and_edges():
    import xgboost as xgb
    from src.modeling.predict import make_prediction_frame

    m = _tiny_model_with_cols()
    n = 8
    ts = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    # strictly increasing IV level for symbol AAA so edges are positive and known
    iv = np.linspace(0.1, 0.1 + 0.001*(n-1), n)
    X_te = pd.DataFrame({
        "a": np.linspace(0, 1, n),
        "b": np.linspace(1, 0, n),
        "sym_AAA": 1.0,
        "panel_IV_AAA": iv,
        "ts_event": ts,
    })
    y_te = pd.Series(np.linspace(0, 1, n))

    out = make_prediction_frame(m, X_te, y_te, ts_te=ts)

    # symbol recovered from dummies; opt_volume defaults to 0 when absent
    assert "symbol" in out.columns and out["symbol"].notna().all()
    assert "opt_volume" in out.columns and (out["opt_volume"] == 0).all()

    # iv_level should copy from panel per symbol
    assert "iv_level" in out.columns
    assert np.isclose(pd.to_numeric(out.loc[out["symbol"] == "AAA", "iv_level"]).dropna().values, iv).all()

    # edges computed as forward log returns; check a couple values
    def fwd_log(v, h):
        v = np.array(v, dtype=float)
        out = np.full_like(v, np.nan, dtype=float)
        out[:-h] = np.log(v[h:]) - np.log(v[:-h])
        return out
    exp_1 = fwd_log(iv, 1)
    got_1 = pd.to_numeric(out["edge_1m"], errors="coerce").values
    assert np.allclose(got_1[:-1], exp_1[:-1], equal_nan=True)


def test_make_prediction_frame_dte_and_strike_passthrough():
    import xgboost as xgb
    from src.modeling.predict import make_prediction_frame

    m = _tiny_model_with_cols()
    n = 5
    ts = pd.date_range("2024-01-01", periods=n, freq="H", tz="UTC")
    expiry = pd.Series(pd.Timestamp("2024-01-03", tz="UTC")).repeat(n).reset_index(drop=True)
    strike = pd.Series([10, 10, 10, 10, 10], dtype=float)
    X_te = pd.DataFrame({"a": np.zeros(n), "b": np.ones(n)})
    y_te = pd.Series(np.zeros(n))

    out = make_prediction_frame(m, X_te, y_te, ts_te=ts, strike_te=strike, expiry_te=expiry)

    # Prediction frame uses 'strike_price' as the strike column
    assert "strike_price" in out.columns and float(out["strike_price"].iloc[0]) == 10.0
    assert "expiry" in out.columns and pd.to_datetime(out["expiry"]).notna().all()
    # dte ~ 48 hours for first row; allow tolerance due to hours->days computation
    assert "dte" in out.columns
    d0 = float(out["dte"].iloc[0])
    assert 1.9 < d0 < 2.1


def test_make_prediction_frame_no_ts_provided_sets_nat():
    import xgboost as xgb
    from src.modeling.predict import make_prediction_frame

    m = _tiny_model_with_cols()
    X_te = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    y_te = pd.Series([0.0, 1.0])
    out = make_prediction_frame(m, X_te, y_te)
    assert "ts_event" in out.columns
    assert out["ts_event"].isna().all()
