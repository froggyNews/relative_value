import pandas as pd
import numpy as np
import pytest

# Robustly skip if xgboost cannot be imported or initialized in this env
try:
    import xgboost as _xgb  # noqa: F401
except Exception as _e:  # includes XGBoostError during DLL load
    pytest.skip(f"xgboost unavailable: {_e}", allow_module_level=True)


def test_train_xgb_pooled_returns_bundle_with_feature_names():
    import xgboost as xgb
    from src.modeling.trainer import train_xgb_pooled

    rng = np.random.default_rng(0)
    X_tr = pd.DataFrame({
        "feat_a": rng.normal(size=50),
        "feat_b": rng.normal(size=50),
        "sym_QBTS": (rng.random(50) > 0.5).astype(float),
    })
    y_tr = pd.Series(rng.normal(size=50))

    bundle = train_xgb_pooled(X_tr, y_tr)
    assert hasattr(bundle, "model")
    assert hasattr(bundle, "feature_names")
    # Should preserve DataFrame column names when available
    assert list(bundle.feature_names) == list(X_tr.columns)


def test_train_xgb_pooled_works_with_minimal_input():
    import xgboost as xgb
    from src.modeling.trainer import train_xgb_pooled

    X_tr = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y_tr = pd.Series([0.0, 0.1, 0.2, 0.3])

    bundle = train_xgb_pooled(X_tr, y_tr)
    preds = bundle.model.predict(X_tr)
    assert len(preds) == len(y_tr)
