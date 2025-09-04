import os
from pathlib import Path
import pytest


@pytest.mark.integration
def test_end_to_end_real_data_if_available():
    # Require xgboost to be installed and loadable
    try:
        import xgboost as _xgb  # noqa: F401
    except Exception as e:
        pytest.skip(f"xgboost unavailable: {e}")

    # Only run when explicitly enabled and DB exists
    run_it = os.getenv("RUN_INTEGRATION", "0") == "1"
    db_path = Path(os.getenv("IV_DB_PATH", "IV_DB_PATH"))
    if not run_it or not db_path.exists():
        pytest.skip("Integration test disabled or DB not found")

    from src.modeling.schemas import BacktestConfig
    from src.backtest.run import run_backtest

    cfg = BacktestConfig(
        tickers=["QBTS", "IONQ", "RGTI", "QUBT"],
        start="2025-06-02",
        end="2025-06-16",
        db_path=db_path,
        auto_fetch=False,  # Require local DB to avoid network in CI
        atm_only=True,

        forward_steps=1,
        test_frac=0.2,
        tolerance="15s",
        r=0.045,

        # Trading / filters
        top_k=1,
        threshold=0.0,
        group_freq="1min",
        min_roll_trades=0.0,
        roll_by_rows=True,
        roll_bars=30,
        roll_window="30min",
        pair_only=False,
        strike=None,
        expiry=None,
        strike_tol=0.0,
        granularity="slice",
        weighting_scheme="opt_volume",
        dte_range=(20, 45),
        moneyness_range=None,
    )

    res = run_backtest(cfg)
    print("[INTEGRATION] Summary:", res.get("summary"))
    # Basic sanity checks
    assert isinstance(res, dict)
    assert "summary" in res and "trades" in res and "preds_head" in res
    # Do not assert on the number of trades since it can legitimately be 0
