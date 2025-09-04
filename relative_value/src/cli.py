
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json

from modeling.schemas import BacktestConfig
from backtest.run import run_backtest

def _load_json_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _config_to_backtest(cfg: dict) -> BacktestConfig:
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    liq = cfg.get("liquidity", {})
    trading = cfg.get("trading", {})
    output = cfg.get("output", {})

    return BacktestConfig(
        tickers=data.get("tickers", ["QBTS", "IONQ", "RGTI", "QUBT"]),
        start=data.get("start", "2025-06-02"),
        end=data.get("end", "2025-06-16"),
        db_path=Path(data.get("db_path", "data/iv_data_1m.db")),
        auto_fetch=bool(data.get("auto_fetch", True)),
        atm_only=(str(data.get("surface_mode", "atm")).lower() == "atm"),

        forward_steps=int(model.get("forward_steps", 1)),
        test_frac=float(model.get("test_frac", 0.2)),
        tolerance=str(model.get("tolerance", "15s")),
        r=float(model.get("r", 0.045)),

        min_roll_trades=float(liq.get("min_roll_trades", 0.0)),
        roll_by_rows=bool(liq.get("roll_by_rows", True)),
        roll_bars=int(liq.get("roll_bars", 30)) if liq.get("roll_bars", 30) is not None else None,
        roll_window=str(liq.get("roll_window", "30min")),
        volume_col=liq.get("volume_col", None),

        top_k=int(trading.get("top_k", 1)),
        threshold=float(trading.get("threshold", 0.0)),
        pair_only=bool(trading.get("pair_only", False)),
        strike=trading.get("strike", None),
        expiry=trading.get("expiry", None),
        strike_tol=float(trading.get("strike_tol", 0.0)),
        group_freq=str(trading.get("group_freq", "1min")),

        save_trades_csv=Path(output.get("save_trades_csv")) if output.get("save_trades_csv") else None,
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relative value backtest using pooled IV-returns model.")
    p.add_argument("--config", type=str, default="", help="Path to JSON config file.")
    p.add_argument("--write-config-template", type=str, default="", help="Write a config template JSON to this path and exit.")
    p.add_argument("--tickers", nargs="+", default=["QBTS", "IONQ", "RGTI", "QUBT"], help="Universe tickers")
    p.add_argument("--start", required=False, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=False, help="End date YYYY-MM-DD")
    p.add_argument("--forward-steps", type=int, default=1, help="Forward horizon in steps")
    p.add_argument("--test-frac", type=float, default=0.2, help="Test fraction for time split")
    p.add_argument("--tolerance", default="15s", help="As-of merge tolerance for panel")
    p.add_argument("--r", type=float, default=0.045, help="Risk-free rate for Greeks/IV calc")
    p.add_argument("--db", type=str, default="data/iv_data_1m.db", help="SQLite DB path")
    p.add_argument("--top-k", type=int, default=1, help="Number of names in each leg")
    p.add_argument("--threshold", type=float, default=0.0, help="Min predicted spread to trade")
    p.add_argument("--save-trades", type=str, default="", help="Optional CSV path to save trades")
    p.add_argument("--no-fetch", action="store_true", help="Disable auto-fetch; require data to already exist")
    p.add_argument("--surface-mode", choices=["atm", "full"], default="atm", help="Use ATM-only or full surface cores")
    # Liquidity filter options
    p.add_argument("--min-roll-trades", type=float, default=0.0, help="Minimum rolling net trades/volume over window")
    p.add_argument("--roll-window", type=str, default="30min", help="Time-based rolling window (e.g., '30min')")
    p.add_argument("--time-rolling", action="store_true", help="Use time-based rolling (default is row-based)")
    p.add_argument("--roll-bars", type=int, default=30, help="Row-based rolling bars")
    p.add_argument("--volume-col", type=str, default=None, help="Preferred volume column to use if available")
    # Pair/contract constraints
    p.add_argument("--pair-only", action="store_true", help="Only trade two different tickers (top vs bottom)")
    p.add_argument("--strike", type=float, default=None, help="Restrict to a specific strike")
    p.add_argument("--expiry", type=str, default=None, help="Restrict to a specific expiry YYYY-MM-DD")
    p.add_argument("--strike-tol", type=float, default=0.0, help="Tolerance for strike matching (abs)")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Write template and exit
    if args.write_config_template:
        from pathlib import Path as _P
        outp = _P(args.write_config_template)
        outp.parent.mkdir(parents=True, exist_ok=True)
        template = {
            "data": {
                "tickers": ["QBTS", "IONQ", "RGTI", "QUBT"],
                "start": "2025-06-02",
                "end": "2025-06-16",
                "db_path": "data/iv_data_1m.db",
                "auto_fetch": True,
                "surface_mode": "atm"
            },
            "model": {
                "forward_steps": 1,
                "test_frac": 0.2,
                "tolerance": "15s",
                "r": 0.045
            },
            "liquidity": {
                "min_roll_trades": 0.0,
                "roll_by_rows": True,
                "roll_bars": 30,
                "roll_window": "30min",
                "volume_col": None
            },
            "trading": {
                "top_k": 1,
                "threshold": 0.0,
                "pair_only": False,
                "strike": None,
                "expiry": None,
                "strike_tol": 0.0,
                "group_freq": "1min"
            },
            "output": {
                "save_trades_csv": "outputs/relative_value_trades.csv"
            }
        }
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
        print(f"[TEMPLATE] Wrote config template to {outp}")
        return

    # Load config or fall back to minimal overrides
    if args.config:
        cfg_dict = _load_json_config(Path(args.config))
        cfg = _config_to_backtest(cfg_dict)
        if args.start:
            cfg.start = args.start
        if args.end:
            cfg.end = args.end
        if args.save_trades:
            cfg.save_trades_csv = Path(args.save_trades)
    else:
        cfg = BacktestConfig(
            tickers=args.tickers,
            start=args.start or "2025-06-02",
            end=args.end or "2025-06-16",
        )
        if args.save_trades:
            cfg.save_trades_csv = Path(args.save_trades)

    results = run_backtest(cfg)
    print("Summary:")
    print(results["summary"])

if __name__ == "__main__":
    main()
