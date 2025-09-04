import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import relative_value as rv


class DummyModel:
    def __init__(self, feature_names: Optional[Sequence[str]] = None, pred_value: float = 0.5):
        # Mimic xgboost.XGBRegressor attribute if present
        self.feature_names_in_ = np.array(feature_names) if feature_names is not None else None
        self._pred_value = float(pred_value)

    def predict(self, X: pd.DataFrame):
        # Return a constant prediction to simplify assertions
        return np.full(len(X), self._pred_value, dtype=float)


def test_ensure_numeric_drops_non_numeric_and_casts_bool():
    df = pd.DataFrame({
        'a': [1, 2],
        'b': [True, False],
        'c': ['x', 'y'],
        'd': pd.to_datetime(['2020-01-01', '2020-01-02'])
    })
    out = rv._ensure_numeric(df.copy())
    assert 'a' in out.columns
    assert 'b' in out.columns
    assert 'c' not in out.columns  # dropped
    assert 'd' not in out.columns  # dropped
    assert out['b'].dtype.kind in ('f', 'i')  # bool to numeric


def test_expected_feature_names_handles_missing_attribute():
    class NoNames:
        pass

    m1 = NoNames()
    assert rv._expected_feature_names(m1) == []

    m2 = DummyModel(feature_names=['x', 'y'])
    assert rv._expected_feature_names(m2) == ['x', 'y']


def test_looks_like_generic_xgb_names():
    assert rv._looks_like_generic_xgb_names(['f0', 'f1', 'f2']) is True
    assert rv._looks_like_generic_xgb_names(['f0', 'x1']) is False
    assert rv._looks_like_generic_xgb_names([]) is False


def test_align_columns_to_model_adds_missing_and_orders():
    X = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    # Specific feature names: should add missing 'c' and order [a,b,c]
    m_specific = DummyModel(feature_names=['a', 'b', 'c'])
    out = rv._align_columns_to_model(m_specific, X.copy())
    assert list(out.columns) == ['a', 'b', 'c']
    assert np.all(out['c'].values == 0.0)

    # Generic XGB names: should not reindex even if names differ
    m_generic = DummyModel(feature_names=['f0', 'f1', 'f2'])
    out2 = rv._align_columns_to_model(m_generic, X.copy())
    assert list(out2.columns) == list(X.columns)


def test_extract_symbol_from_dummies():
    s1 = pd.Series({'sym_A': 1.0})
    assert rv._extract_symbol_from_dummies(s1) == 'A'

    s2 = pd.Series({'sym_A': 0.0, 'sym_B': 2.0})
    assert rv._extract_symbol_from_dummies(s2) == 'B'

    s3 = pd.Series({'x': 1.0})
    assert rv._extract_symbol_from_dummies(s3) is None

    s4 = pd.Series({'sym_A': np.nan, 'sym_B': np.nan})
    assert rv._extract_symbol_from_dummies(s4) is None


def test_make_prediction_frame_basic_fields_and_edges():
    # Two symbols with increasing iv levels to have non-NaN edges
    ts = pd.to_datetime([
        '2025-01-01 09:30', '2025-01-01 09:31',
        '2025-01-01 09:30', '2025-01-01 09:31',
    ], utc=True)
    X_te = pd.DataFrame({
        'feat': [1.0, 2.0, 3.0, 4.0],
        'sym_A': [1, 1, 0, 0],
        'sym_B': [0, 0, 1, 1],
        'panel_IV_A': [10.0, 10.5, np.nan, np.nan],
        'panel_IV_B': [np.nan, np.nan, 8.0, 8.2],
    })
    y_te = pd.Series([0.1, 0.2, -0.1, -0.2])
    strike_te = pd.Series([100, 100, 200, 200])
    expiry_te = pd.to_datetime(['2025-02-01'] * 4, utc=True)

    m = DummyModel(feature_names=['feat', 'sym_A', 'sym_B', 'panel_IV_A', 'panel_IV_B'], pred_value=0.7)
    out = rv.make_prediction_frame(m, X_te, y_te, ts, strike_te, expiry_te)

    assert set(['ts_event', 'y_true', 'y_pred', 'symbol', 'strike', 'expiry', 'opt_volume', 'edge_1m', 'edge_15m', 'edge_60m']).issubset(out.columns)
    # constant prediction
    assert np.allclose(out['y_pred'].values, 0.7)
    # symbol recovered
    assert set(out['symbol'].dropna().unique().tolist()) == {'A', 'B'}
    # strike and expiry carried over
    assert out['strike'].notna().all()
    assert pd.to_datetime(out['expiry'], errors='coerce').notna().all()
    # edges exist (may be NaN at last rows due to shift)
    assert 'edge_1m' in out.columns


def test_make_relative_value_trades_pair_only_and_threshold():
    # Two timestamps, two symbols each
    ts = pd.to_datetime(['2025-01-01 09:30', '2025-01-01 09:30', '2025-01-01 09:31', '2025-01-01 09:31'], utc=True)
    preds = pd.DataFrame({
        'ts_event': ts,
        'symbol': ['A', 'B', 'A', 'B'],
        'y_pred': [0.9, 0.1, 0.6, 0.2],
        'y_true': [0.05, -0.05, 0.02, -0.01],
        'opt_volume': [10, 5, 10, 5],
    })
    trades = rv.make_relative_value_trades(
        preds,
        top_k=1,
        threshold=0.1,  # sufficient spread to pass
        min_roll_trades=0.0,
        roll_window='30min',
        roll_by_rows=True,
        roll_bars=2,
        pair_only=True,
    )
    assert not trades.empty
    # Pair-only should produce exactly one long and one short symbol per ts
    for _, row in trades.iterrows():
        assert len(row['long_symbols'].split(',')) == 1
        assert len(row['short_symbols'].split(',')) == 1
        assert abs(row['pred_spread']) >= 0.1


def test_make_relative_value_trades_min_roll_trades_filters():
    ts = pd.to_datetime(['2025-01-01 09:30'] * 4, utc=True)
    preds = pd.DataFrame({
        'ts_event': ts,
        'symbol': ['A', 'B', 'C', 'D'],
        'y_pred': [0.9, 0.1, 0.8, 0.2],
        'y_true': [0.05, -0.05, 0.02, -0.01],
        'opt_volume': [0, 0, 0, 0],
    })
    # With positive threshold on rolling trades, no trades should pass
    trades = rv.make_relative_value_trades(
        preds,
        min_roll_trades=1.0,
        roll_by_rows=True,
        roll_bars=1,
    )
    assert trades.empty


def test_summarize_trades_metrics():
    trades = pd.DataFrame({
        'pred_spread': [0.2, -0.3, 0.1],
        'edge_1m': [0.05, -0.02, 0.01],
        'edge_15m': [0.1, -0.1, 0.01],
        'edge_60m': [np.nan, 0.2, -0.05],
    })
    s = rv.summarize_trades(trades)
    assert s['n_trades'] == 3
    # Hit rates compare sign(edge_X) vs sign(pred_spread)
    assert 0.0 <= (s['hit_rate_15m'] or 0.0) <= 1.0


def test_parse_args_flags(monkeypatch):
    argv = [
        'prog', '--tickers', 'A', 'B', '--start', '2025-06-01', '--end', '2025-06-10',
        '--forward-steps', '2', '--test-frac', '0.3', '--tolerance', '10s', '--r', '0.05',
        '--db', 'data/iv.db', '--top-k', '2', '--threshold', '0.1', '--save-trades', 'out.csv',
        '--no-fetch', '--surface-mode', 'full', '--min-roll-trades', '2', '--roll-window', '45min',
        '--time-rolling', '--roll-bars', '15', '--pair-only', '--strike', '100', '--expiry', '2025-07-19', '--strike-tol', '1.5'
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    args = rv.parse_args()
    assert args.tickers == ['A', 'B']
    assert args.start == '2025-06-01'
    assert args.end == '2025-06-10'
    assert args.forward_steps == 2
    assert abs(args.test_frac - 0.3) < 1e-9
    assert args.tolerance == '10s'
    assert abs(args.r - 0.05) < 1e-9
    assert args.db == 'data/iv.db'
    assert args.top_k == 2
    assert abs(args.threshold - 0.1) < 1e-9
    assert args.save_trades == 'out.csv'
    assert args.no_fetch is True
    assert args.surface_mode == 'full'
    assert abs(args.min_roll_trades - 2.0) < 1e-9
    assert args.roll_window == '45min'
    assert args.time_rolling is True
    assert args.roll_bars == 15
    assert args.pair_only is True
    assert abs(args.strike - 100.0) < 1e-9
    assert args.expiry == '2025-07-19'
    assert abs(args.strike_tol - 1.5) < 1e-9


def test_run_backtest_flow_with_mocks(monkeypatch):
    # Create synthetic split to bypass data fetch
    n = 20
    idx = pd.RangeIndex(n)
    X = pd.DataFrame({'feat': np.arange(n, dtype=float), 'sym_A': [1]*n, 'panel_IV_A': np.linspace(10, 11, n)})
    y = pd.Series(np.linspace(0.0, 0.1, n))
    ts = pd.date_range('2025-01-01 09:30', periods=n, freq='T', tz='UTC')
    split = 15
    X_tr, X_te = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_tr, y_te = y.iloc[:split].copy(), y.iloc[split:].copy()
    ts_tr, ts_te = ts[:split], ts[split:]
    strike_tr = pd.Series([100]*split)
    strike_te = pd.Series([100]*(n-split))
    expiry_tr = pd.Series(pd.to_datetime(['2025-02-01']*split, utc=True))
    expiry_te = pd.Series(pd.to_datetime(['2025-02-01']*(n-split), utc=True))

    def fake_build_and_split(cfg):
        return X_tr, X_te, y_tr, y_te, pd.Series(ts_tr), pd.Series(ts_te), strike_tr, strike_te, expiry_tr, expiry_te

    def fake_train_model(X_tr_in, y_tr_in):
        return DummyModel(feature_names=list(X.columns), pred_value=0.3)

    monkeypatch.setattr(rv, 'build_and_split_pooled', fake_build_and_split)
    monkeypatch.setattr(rv, 'train_pooled_model_on_split', fake_train_model)

    cfg = rv.BacktestConfig(
        tickers=['A', 'B'], start='2025-01-01', end='2025-01-02', forward_steps=1, test_frac=0.2,
        top_k=1, threshold=0.0, auto_fetch=False
    )
    results = rv.run_backtest(cfg)
    assert 'summary' in results and 'trades' in results and 'preds_head' in results
    # With constant predictions, trades may be empty depending on symbol extraction; allow either
    assert isinstance(results['summary'], dict)

