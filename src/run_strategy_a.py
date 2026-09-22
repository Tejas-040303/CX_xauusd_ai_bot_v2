"""
Project MIDAS v2 — Strategy A Runner
Handles multi-timeframe data merging and runs backtest + forward test.

Usage:
    python run_strategy_a.py                    # Full backtest + forward test
    python run_strategy_a.py --mode backtest    # Backtest only
    python run_strategy_a.py --diag             # Include diagnostics
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
from strategies.strategy_a import StrategyA


def load_and_prepare_data() -> pd.DataFrame:
    """Load M5 and M15, compute SMC features on M15, merge into M5."""
    engine = FeatureEngine()

    # ─── M15 ───────────────────────────────────────────
    m15_processed = DATA_PROCESSED / "XAUUSD_M15_features.csv"
    if m15_processed.exists():
        print("Loading processed M15 data...")
        df_m15 = pd.read_csv(m15_processed, parse_dates=["datetime"])
    else:
        print("Processing M15 data from raw...")
        df_m15 = pd.read_csv(DATA_RAW / "XAUUSD_M15.csv", parse_dates=["datetime"])
        df_m15 = engine.compute_shared(df_m15, timeframe="M15")
        df_m15.to_csv(m15_processed, index=False)

    # Compute SMC features on M15
    print("Computing SMC features on M15...")
    df_m15 = engine.compute_smc(df_m15)

    # ─── M5 ────────────────────────────────────────────
    m5_processed = DATA_PROCESSED / "XAUUSD_M5_features.csv"
    if m5_processed.exists():
        print("Loading processed M5 data...")
        df_m5 = pd.read_csv(m5_processed, parse_dates=["datetime"])
    else:
        print("Processing M5 data from raw...")
        df_m5 = pd.read_csv(DATA_RAW / "XAUUSD_M5.csv", parse_dates=["datetime"])
        df_m5 = engine.compute_shared(df_m5, timeframe="M5")
        df_m5.to_csv(m5_processed, index=False)

    # ─── Merge M15 into M5 ────────────────────────────
    print("Merging M15 features into M5 (asof merge)...")
    df_merged = merge_timeframes(df_m5, df_m15)

    print(f"Merged dataset: {len(df_merged):,} M5 candles with {len(df_merged.columns)} columns")
    return df_merged


def merge_timeframes(df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    """Merge M15 SMC + indicator features into M5."""
    df_m5 = df_m5.sort_values("datetime").reset_index(drop=True)
    df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

    m15_cols = ["datetime"]

    # SMC columns
    smc_cols = [c for c in df_m15.columns if c.startswith(("bos_", "ob_", "equal_", "last_equal_"))]
    m15_cols.extend(smc_cols)

    # Indicator columns
    indicator_cols = [
         "open", "high", "low", "close",
        "ema_9", "ema_21", "ema_50", "ema_200",
        "ema_50_slope", "ema_stack",
        "ema_partial_bull", "ema_partial_bear",
        "above_ema200",
        f"atr_{ATR_PERIOD}", f"rsi_{ATR_PERIOD}",
        f"adx_{ATR_PERIOD}", "trend_strong",
        "candle_range", "body", "is_bullish",
        "last_swing_high", "last_swing_low",
        "vol_regime",
    ]
    for col in indicator_cols:
        if col in df_m15.columns:
            m15_cols.append(col)

    m15_cols = list(dict.fromkeys(m15_cols))
    df_m15_subset = df_m15[m15_cols].copy()

    m15_original_index = df_m15_subset.index.values

    rename_map = {col: f"m15_{col}" for col in df_m15_subset.columns if col != "datetime"}
    df_m15_subset = df_m15_subset.rename(columns=rename_map)
    df_m15_subset = df_m15_subset.rename(columns={"datetime": "m15_datetime_key"})
    df_m15_subset["m15_idx"] = m15_original_index

    df_merged = pd.merge_asof(
        df_m5,
        df_m15_subset,
        left_on="datetime",
        right_on="m15_datetime_key",
        direction="backward",
    )

    df_merged = df_merged.rename(columns={"m15_datetime_key": "m15_datetime"})
    return df_merged


def run_strategy_a(mode: str = "both", show_diag: bool = False):
    """Run Strategy A backtest and/or forward test."""

    df = load_and_prepare_data()
    bt = Backtester()
    results = {}

    if mode in ("both", "backtest"):
        print("\n" + "=" * 60)
        print("  STRATEGY A (SMC OB) — BACKTEST")
        print("=" * 60)
        strategy = StrategyA()
        results["backtest"] = bt.run(df, strategy, mode="backtest")
        bt.save_results(results["backtest"], "StrategyA_SMC_OB_backtest.json")
        if show_diag:
            strategy.print_diagnostics()
        try:
            bt.plot_equity(
                results["backtest"],
                save_path=str(BACKTEST_DIR / "StrategyA_SMC_OB_backtest_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    if mode in ("both", "forward"):
        print("\n" + "=" * 60)
        print("  STRATEGY A (SMC OB) — FORWARD TEST")
        print("=" * 60)
        strategy = StrategyA()
        results["forward"] = bt.run(df, strategy, mode="forward")
        bt.save_results(results["forward"], "StrategyA_SMC_OB_forward.json")
        if show_diag:
            strategy.print_diagnostics()
        try:
            bt.plot_equity(
                results["forward"],
                save_path=str(BACKTEST_DIR / "StrategyA_SMC_OB_forward_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    # ─── Comparison ───────────────────────────────────
    if "backtest" in results and "forward" in results:
        print("\n" + "=" * 60)
        print("  STRATEGY A (SMC OB) — COMPARISON SUMMARY")
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
             "✅" if mf["trades_per_month"] > 3 else "⚠️"),
            ("Sharpe Ratio", f"{mb['sharpe_ratio']:.2f}", f"{mf['sharpe_ratio']:.2f}",
             "✅" if mf["sharpe_ratio"] > 0 else "❌"),
            ("Avg RR", f"{mb['avg_rr_achieved']:.2f}", f"{mf['avg_rr_achieved']:.2f}",
             "✅" if mf["avg_rr_achieved"] > 0 else "❌"),
            ("Max Consec Loss", f"{mb['max_consecutive_losses']}", f"{mf['max_consecutive_losses']}",
             "✅" if mf["max_consecutive_losses"] < 6 else "❌"),
            ("Avg Win/Loss", f"{mb['avg_win_loss_ratio']:.2f}", f"{mf['avg_win_loss_ratio']:.2f}",
             "✅" if mf["avg_win_loss_ratio"] > 1.2 else "❌"),
        ]

        for label, bt_val, fw_val, status in comparisons:
            print(f"  {label:<30} {bt_val:>12} {fw_val:>12} {status:>10}")

        print("\n  " + "-" * 66)
        fw_passing = sum(1 for _, _, _, s in comparisons if s == "✅")
        if fw_passing >= 6:
            print("  ASSESSMENT: Strategy A PASSES initial validation ✅")
        elif fw_passing >= 4:
            print("  ASSESSMENT: Strategy A shows PROMISE but needs tuning ⚠️")
        else:
            print("  ASSESSMENT: Strategy A FAILS validation ❌")

    print("\nResults saved to:", BACKTEST_DIR)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDAS v2 — Strategy A Runner")
    parser.add_argument("--mode", choices=["backtest", "forward", "both"],
                        default="both", help="Which test to run")
    parser.add_argument("--diag", action="store_true",
                        help="Show diagnostic counters")
    args = parser.parse_args()

    results = run_strategy_a(args.mode, args.diag)