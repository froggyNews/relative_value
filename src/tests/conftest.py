import os
import sys
import types
from pathlib import Path

# Ensure project root is importable for `import src.*`
_here = Path(__file__).resolve()
_project_root = _here.parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import pytest


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    db = tmp_path / "test_iv.db"
    # Ensure parent directories exist
    db.parent.mkdir(parents=True, exist_ok=True)
    return db


@pytest.fixture(autouse=True)
def _set_env(tmp_path: Path, request):
    # Point to a temp DB path by default
    os.environ.setdefault("IV_DB_PATH", str(tmp_path / "default.db"))
    yield


@pytest.fixture
def simple_core_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01 14:30", periods=20, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_event": ts,
            "expiry_date": ts[0] + pd.Timedelta(days=30),
            "opt_symbol": ["240201C00100000"] * len(ts),
            "stock_symbol": ["XYZ"] * len(ts),
            "opt_close": 2.5,
            "stock_close": 100.0,
            "opt_volume": 100,
            "stock_volume": 1_000,
            "option_type": "C",
            "strike_price": 100.0,
            "time_to_expiry": 30.0 / 365.0,
            "moneyness": 0.0,
        }
    )
    # Provide iv_clip since downstream features rely on it
    df["iv"] = 0.2
    df["iv_clip"] = 0.2
    df["symbol"] = "XYZ"
    return df


@pytest.fixture
def stub_databento_module(monkeypatch):
    """Provide a stub for the 'databento' import used by fetch module."""
    fake = types.ModuleType("databento")

    class _FakeTS:
        def get_range(self, *args, **kwargs):
            # Return a minimal frame with required columns
            idx = pd.date_range("2024-01-01", periods=3, freq="H", tz="UTC")
            return pd.DataFrame({"ts_event": idx, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "symbol": "X"})

    class _FakeClient:
        def __init__(self, *_a, **_k):
            self.timeseries = _FakeTS()

    fake.Historical = _FakeClient
    sys.modules["databento"] = fake
    yield
    # Cleanup not strictly needed as test process is isolated, but keep tidy
    sys.modules.pop("databento", None)
