import numpy as np
import pandas as pd
import pytest

# Robustly skip if xgboost cannot be imported or initialized in this env
try:
    import xgboost as _xgb  # noqa: F401
except Exception as _e:  # includes XGBoostError during DLL load
    pytest.skip(f"xgboost unavailable: {_e}", allow_module_level=True)


def test_make_prediction_frame_with_xgb():
    import xgboost as xgb
    from src.modeling.predict import make_prediction_frame

    # Train a tiny model to get feature_names_in_
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y = pd.Series([0.0, 0.1, 0.2, 0.3])
    model = xgb.XGBRegressor(n_estimators=10, max_depth=2, subsample=1.0, colsample_bytree=1.0, random_state=0)
    model.fit(X, y)

    # Create test frame with symbol dummies and extra columns
    n = 20
    ts = pd.date_range("2024-01-01", periods=n, freq="H", tz="UTC")
    X_te = pd.DataFrame(
        {
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "sym_AAA": [1.0] * n,
            "core_opt_volume": np.random.randint(0, 100, size=n),
            "panel_IV_AAA": np.linspace(0.1, 0.2, n),
            "ts_event": ts,
        }
    )
    y_te = pd.Series(np.random.randn(n))

    out = make_prediction_frame(model, X_te, y_te, ts_te=ts)

    # Required columns present
    for c in ("ts_event", "y_true", "y_pred", "symbol", "opt_volume"):
        assert c in out.columns
    # Edges computed or set to NaN
    for c in ("edge_1m", "edge_15m", "edge_60m"):
        assert c in out.columns
