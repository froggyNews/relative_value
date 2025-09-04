import sys
from pathlib import Path

# Ensure both repository root and src directory are importable for tests
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Provide a lightweight xgboost stub if xgboost isn't installed in the test env
try:
    import xgboost  # type: ignore
except Exception:  # pragma: no cover
    import types
    import numpy as _np
    import pandas as _pd

    xgb_stub = types.ModuleType('xgboost')

    class XGBRegressor:  # minimal stub for tests
        def __init__(self, **kwargs):
            self.feature_names_in_ = None
            self._kwargs = kwargs

        def fit(self, X: _pd.DataFrame, y):
            try:
                self.feature_names_in_ = _np.array(list(X.columns))
            except Exception:
                self.feature_names_in_ = None
            return self

        def predict(self, X: _pd.DataFrame):
            return _np.zeros(len(X), dtype=float)

    xgb_stub.XGBRegressor = XGBRegressor
    sys.modules['xgboost'] = xgb_stub
