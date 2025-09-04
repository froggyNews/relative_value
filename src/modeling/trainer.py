
from __future__ import annotations

from typing import List
import xgboost as xgb
import pandas as pd

from .schemas import ModelBundle

def train_xgb_pooled(X_tr: pd.DataFrame, y_tr: pd.Series) -> ModelBundle:
    """Train a simple XGBRegressor and return a bundle with feature names."""
    params = dict(
        objective="reg:squarederror",
        n_estimators=350, learning_rate=0.05,
        max_depth=6, subsample=0.9, colsample_bytree=0.9,
        random_state=42,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    # Resolve feature names as best we can
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = list(X_tr.columns)
    return ModelBundle(model=model, feature_names=list(names))
