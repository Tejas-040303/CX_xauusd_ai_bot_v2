"""
Project MIDAS v2 — Strategy C Runner
Session Range Breakout — backtest + forward test.

Usage:
    python run_strategy_c.py                    # Full backtest + forward test
    python run_strategy_c.py --mode backtest    # Backtest only
    python run_strategy_c.py --diag             # Include diagnostics
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_RAW, DATA_PROCESSED, BACKTEST_DIR, ATR_PERIOD
from feature_engineering import FeatureEngine
from backtester import Backtester
from strategies.strategy_c import StrategyC


def load_and_prepare_data() -> pd.DataFrame:
    """Load M5 data with session range features computed."""
    engine = FeatureEngine()

    m5_processed = DATA_PROCESSED / "XAUUSD_M5_features.csv"
    if m5_processed.exists():
        print("Loading processed M5 data...")
        df_m5 = pd.read_csv(m5_processed, parse_dates=["datetime"])
    else:
        print("Processing M5 data from raw...")
        df_m5 = pd.read_csv(DATA_RAW / "XAUUSD_M5.csv", parse_dates=["datetime"])
        df_m5 = engine.compute_shared(df_m5, timeframe="M5")
        df_m5.to_csv(m5_processed, index=False)

    # Compute session range features
    print("Computing session range features on M5...")
    df_m5 = engine.compute_session_range(df_m5)

    print(f"Dataset: {len(df_m5):,} M5 candles with {len(df_m5.columns)} columns")
    return df_m5


def run_strategy_c(mode: str = "both", show_diag: bool = False):
    """Run Strategy C backtest and/or forward test."""

    df = load_and_prepare_data()
    bt = Backtester()
    results = {}

    if mode in ("both", "backtest"):
        print("\n" + "=" * 60)
        print("  STRATEGY C (SESSION BREAKOUT) — BACKTEST")
        print("=" * 60)
        strategy = StrategyC()
        results["backtest"] = bt.run(df, strategy, mode="backtest")
        bt.save_results(results["backtest"], "StrategyC_SessionBreakout_backtest.json")
        if show_diag:
            strategy.print_diagnostics()
        try:
            bt.plot_equity(
                results["backtest"],
                save_path=str(BACKTEST_DIR / "StrategyC_SessionBreakout_backtest_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    if mode in ("both", "forward"):
        print("\n" + "=" * 60)
        print("  STRATEGY C (SESSION BREAKOUT) — FORWARD TEST")
        print("=" * 60)
        strategy = StrategyC()
        results["forward"] = bt.run(df, strategy, mode="forward")
        bt.save_results(results["forward"], "StrategyC_SessionBreakout_forward.json")
        if show_diag:
            strategy.print_diagnostics()
        try:
            bt.plot_equity(
                results["forward"],
                save_path=str(BACKTEST_DIR / "StrategyC_SessionBreakout_forward_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    # ─── Comparison ───────────────────────────────────
    if "backtest" in results and "forward" in results:
        print("\n" + "=" * 60)
        print("  STRATEGY C (SESSION BREAKOUT) — COMPARISON SUMMARY")
        print("=" * 60)

        mb = results["backtest"]["metrics"]
        mf = results["forward"]["metrics"]

        print(f"\n  {'Metric':<30} {'Backtest':>12} {'Forward':>12} {'Status':>10}")
        print("  " + "-" * 66)

        comparisons = [
            ("Total P&L", f"${mb['total_pnl']:+.2f}", f"${mf['total_pnl']:+.2f}",
             "✅" if mf["total_pnl"] > 0 else "❌"),
            ("Win Rate", f"{mb['win_rate']:.1f}%", f"{mf['win_rate']:.1f}%",
             "✅" if mf["win_rate"] > 50 else "❌"),
            ("Profit Factor", f"{mb['profit_factor']:.2f}", f"{mf['profit_factor']:.2f}",
             "✅" if mf["profit_factor"] > 1.5 else "❌"),
            ("Max Drawdown", f"{mb['max_drawdown_pct']:.1f}%", f"{mf['max_drawdown_pct']:.1f}%",
             "✅" if mf["max_drawdown_pct"] < 8 else "❌"),
            ("Trades/Month", f"{mb['trades_per_month']:.1f}", f"{mf['trades_per_month']:.1f}",
             "✅" if mf["trades_per_month"] > 5 else "⚠️"),
            ("Sharpe Ratio", f"{mb['sharpe_ratio']:.2f}", f"{mf['sharpe_ratio']:.2f}",
             "✅" if mf["sharpe_ratio"] > 0 else "❌"),
            ("Avg RR", f"{mb['avg_rr_achieved']:.2f}", f"{mf['avg_rr_achieved']:.2f}",
             "✅" if mf["avg_rr_achieved"] > 0 else "❌"),
            ("Max Consec Loss", f"{mb['max_consecutive_losses']}", f"{mf['max_consecutive_losses']}",
             "✅" if mf["max_consecutive_losses"] < 6 else "❌"),
            ("Avg Win/Loss", f"{mb['avg_win_loss_ratio']:.2f}", f"{mf['avg_win_loss_ratio']:.2f}",
             "✅" if mf["avg_win_loss_ratio"] > 1.0 else "❌"),
        ]

        for label, bt_val, fw_val, status in comparisons:
            print(f"  {label:<30} {bt_val:>12} {fw_val:>12} {status:>10}")

        print("\n  " + "-" * 66)
        fw_passing = sum(1 for _, _, _, s in comparisons if s == "✅")
        if fw_passing >= 6:
            print("  ASSESSMENT: Strategy C PASSES initial validation ✅")
        elif fw_passing >= 4:
            print("  ASSESSMENT: Strategy C shows PROMISE but needs tuning ⚠️")
        else:
            print("  ASSESSMENT: Strategy C FAILS validation ❌")

    print("\nResults saved to:", BACKTEST_DIR)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDAS v2 — Strategy C Runner")
    parser.add_argument("--mode", choices=["backtest", "forward", "both"],
                        default="both", help="Which test to run")
    parser.add_argument("--diag", action="store_true",
                        help="Show diagnostic counters")
    args = parser.parse_args()

    results = run_strategy_c(args.mode, args.diag)