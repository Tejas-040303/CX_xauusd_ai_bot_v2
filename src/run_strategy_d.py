"""
Project MIDAS v2 — Strategy D Runner
Handles multi-timeframe data merging and runs backtest + forward test.

Usage:
    python run_strategy_d.py                # Full backtest + forward test
    python run_strategy_d.py --mode backtest    # Backtest only
    python run_strategy_d.py --mode forward     # Forward test only
    python run_strategy_d.py --diag             # Include diagnostics
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
from strategies.strategy_d import StrategyD


def load_and_prepare_data(force_recompute: bool = False) -> pd.DataFrame:
    """
    Load M5 and M15 data, compute features including trend-specific ones,
    and merge M15 features into M5.

    Returns:
        M5 DataFrame with M15 features merged and trend features computed
    """
    engine = FeatureEngine()

    # ─── Load and process M15 (trend direction timeframe) ──
    m15_processed = DATA_PROCESSED / "XAUUSD_M15_features.csv"
    if m15_processed.exists() and not force_recompute:
        print("Loading processed M15 data...")
        df_m15 = pd.read_csv(m15_processed, parse_dates=["datetime"])
    else:
        print("Processing M15 data from raw...")
        df_m15 = pd.read_csv(DATA_RAW / "XAUUSD_M15.csv", parse_dates=["datetime"])
        df_m15 = engine.compute_shared(df_m15, timeframe="M15")
        df_m15.to_csv(m15_processed, index=False)

    # ─── Load and process M5 (entry timeframe) ────────────
    m5_processed = DATA_PROCESSED / "XAUUSD_M5_features.csv"
    if m5_processed.exists() and not force_recompute:
        print("Loading processed M5 data...")
        df_m5 = pd.read_csv(m5_processed, parse_dates=["datetime"])
    else:
        print("Processing M5 data from raw...")
        df_m5 = pd.read_csv(DATA_RAW / "XAUUSD_M5.csv", parse_dates=["datetime"])
        df_m5 = engine.compute_shared(df_m5, timeframe="M5")
        df_m5.to_csv(m5_processed, index=False)

    # Compute trend-specific features on M5
    print("Computing trend features on M5...")
    df_m5 = engine.compute_trend(df_m5)

    # ─── Merge M15 features into M5 ──────────────────────
    print("Merging M15 features into M5 (asof merge)...")
    df_merged = merge_timeframes(df_m5, df_m15)

    print(f"Merged dataset: {len(df_merged):,} M5 candles with {len(df_merged.columns)} columns")
    return df_merged


def merge_timeframes(df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    """
    Merge M15 features into M5 using an as-of merge.
    For each M5 candle, attach the most recent M15 candle's features.
    No lookahead bias — M5 only sees M15 data from the past.
    """
    df_m5 = df_m5.sort_values("datetime").reset_index(drop=True)
    df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

    # Select M15 columns to merge
    m15_cols = ["datetime"]

    # Key M15 indicators for trend strategy
    indicator_cols = [
        "ema_9", "ema_21", "ema_50", "ema_200",
        "ema_50_slope", "ema_stack",
        "ema_partial_bull", "ema_partial_bear",
        "above_ema200",
        f"atr_{ATR_PERIOD}",
        f"adx_{ATR_PERIOD}", "trend_strong", "trend_very_strong", "trend_weak",
        "last_swing_high", "last_swing_low",
        "vol_regime",
    ]
    for col in indicator_cols:
        if col in df_m15.columns:
            m15_cols.append(col)

    df_m15_subset = df_m15[m15_cols].copy()

    # Store index before rename
    m15_original_index = df_m15_subset.index.values

    # Rename with prefix
    rename_map = {col: f"m15_{col}" for col in df_m15_subset.columns if col != "datetime"}
    df_m15_subset = df_m15_subset.rename(columns=rename_map)
    df_m15_subset = df_m15_subset.rename(columns={"datetime": "m15_datetime_key"})

    # Add index after rename
    df_m15_subset["m15_idx"] = m15_original_index

    # As-of merge
    df_merged = pd.merge_asof(
        df_m5,
        df_m15_subset,
        left_on="datetime",
        right_on="m15_datetime_key",
        direction="backward",
    )

    df_merged = df_merged.rename(columns={"m15_datetime_key": "m15_datetime"})

    return df_merged


def run_strategy_d(mode: str = "both", show_diag: bool = False):
    """Run Strategy D backtest and/or forward test."""

    df = load_and_prepare_data()

    bt = Backtester()
    results = {}

    if mode in ("both", "backtest"):
        print("\n" + "=" * 60)
        print("  STRATEGY D (TREND) — BACKTEST")
        print("=" * 60)
        strategy_bt = StrategyD()
        results["backtest"] = bt.run(df, strategy_bt, mode="backtest")
        bt.save_results(results["backtest"], "StrategyD_Trend_backtest.json")
        if show_diag:
            strategy_bt.print_diagnostics()
        try:
            bt.plot_equity(
                results["backtest"],
                save_path=str(BACKTEST_DIR / "StrategyD_Trend_backtest_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    if mode in ("both", "forward"):
        print("\n" + "=" * 60)
        print("  STRATEGY D (TREND) — FORWARD TEST")
        print("=" * 60)
        strategy_fw = StrategyD()
        results["forward"] = bt.run(df, strategy_fw, mode="forward")
        bt.save_results(results["forward"], "StrategyD_Trend_forward.json")
        if show_diag:
            strategy_fw.print_diagnostics()
        try:
            bt.plot_equity(
                results["forward"],
                save_path=str(BACKTEST_DIR / "StrategyD_Trend_forward_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    # ─── Comparison Summary ───────────────────────────────
    if "backtest" in results and "forward" in results:
        print("\n" + "=" * 60)
        print("  STRATEGY D (TREND) — COMPARISON SUMMARY")
        print("=" * 60)

        mb = results["backtest"]["metrics"]
        mf = results["forward"]["metrics"]

        headers = f"  {'Metric':<30} {'Backtest':>12} {'Forward':>12} {'Status':>10}"
        print(headers)
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
             "✅" if mf["trades_per_month"] > 15 else "⚠️"),
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
            print("  ASSESSMENT: Strategy D PASSES initial validation ✅")
            print("  Ready for combined testing with Strategy B.")
        elif fw_passing >= 4:
            print("  ASSESSMENT: Strategy D shows PROMISE but needs tuning ⚠️")
            print("  Review failing metrics before proceeding.")
        else:
            print("  ASSESSMENT: Strategy D FAILS validation ❌")
            print("  Major issues need fixing before combining with other strategies.")

    print("\nResults saved to:", BACKTEST_DIR)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDAS v2 — Strategy D Runner")
    parser.add_argument("--mode", choices=["backtest", "forward", "both"],
                        default="both", help="Which test to run")
    parser.add_argument("--diag", action="store_true",
                        help="Show diagnostic counters after each run")
    args = parser.parse_args()

    results = run_strategy_d(args.mode, args.diag)