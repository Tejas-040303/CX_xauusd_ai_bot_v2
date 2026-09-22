"""
Project MIDAS v2 — Strategy B Runner
Handles multi-timeframe data merging and runs backtest + forward test.

Usage:
    python run_strategy_b.py              # Full backtest + forward test
    python run_strategy_b.py --mode backtest   # Backtest only
    python run_strategy_b.py --mode forward    # Forward test only
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
from strategies.strategy_b import StrategyB


def load_and_prepare_data(force_recompute: bool = False) -> pd.DataFrame:
    """
    Load M5 and M15 data, compute features, detect FVGs on M15,
    and merge M15 features into M5 for multi-timeframe analysis.

    Returns:
        M5 DataFrame with M15 FVG features merged in (prefixed 'm15_')
    """
    engine = FeatureEngine()

    # ─── Load and process M15 (structure/detection timeframe) ──
    m15_processed = DATA_PROCESSED / "XAUUSD_M15_features.csv"
    if m15_processed.exists() and not force_recompute:
        print("Loading processed M15 data...")
        df_m15 = pd.read_csv(m15_processed, parse_dates=["datetime"])
    else:
        print("Processing M15 data from raw...")
        df_m15 = pd.read_csv(DATA_RAW / "XAUUSD_M15.csv", parse_dates=["datetime"])
        df_m15 = engine.compute_shared(df_m15, timeframe="M15")
        df_m15.to_csv(m15_processed, index=False)

    # Compute FVG features on M15
    print("Detecting FVGs on M15...")
    df_m15 = engine.compute_fvg(df_m15)

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

    # ─── Merge M15 features into M5 ──────────────────────
    print("Merging M15 features into M5 (asof merge)...")
    df_merged = merge_timeframes(df_m5, df_m15)

    print(f"Merged dataset: {len(df_merged):,} M5 candles with {len(df_merged.columns)} columns")
    return df_merged


def merge_timeframes(df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    """
    Merge M15 features into M5 using an as-of merge.
    For each M5 candle, attach the most recent M15 candle's features.
    This ensures no lookahead bias — M5 only sees M15 data from the past.

    M15 columns are prefixed with 'm15_' in the merged output.
    """
    # Ensure sorted by time
    df_m5 = df_m5.sort_values("datetime").reset_index(drop=True)
    df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

    # Select M15 columns to merge (FVG features + key indicators)
    m15_cols = ["datetime"]
    # FVG columns
    fvg_cols = [c for c in df_m15.columns if c.startswith("fvg_")]
    m15_cols.extend(fvg_cols)
    # Key M15 indicators that strategies might need
    indicator_cols = [
        "ema_50", "ema_50_slope", "ema_200", "ema_stack",
        f"atr_{ATR_PERIOD}", f"rsi_{ATR_PERIOD}",
        f"adx_{ATR_PERIOD}", "trend_strong",
        "candle_range", "body", "is_bullish",
        "last_swing_high", "last_swing_low",
    ]
    for col in indicator_cols:
        if col in df_m15.columns:
            m15_cols.append(col)

    df_m15_subset = df_m15[m15_cols].copy()

    # Store the original M15 index separately (added AFTER rename to avoid double prefix)
    m15_original_index = df_m15_subset.index.values

    # Rename M15 columns with prefix (except datetime which we merge on)
    rename_map = {col: f"m15_{col}" for col in df_m15_subset.columns if col != "datetime"}
    df_m15_subset = df_m15_subset.rename(columns=rename_map)
    df_m15_subset = df_m15_subset.rename(columns={"datetime": "m15_datetime_key"})

    # NOW add the index column (after rename, so it doesn't get double-prefixed)
    df_m15_subset["m15_idx"] = m15_original_index

    # As-of merge: for each M5 row, find the most recent M15 row
    # that has datetime <= M5 datetime
    df_merged = pd.merge_asof(
        df_m5,
        df_m15_subset,
        left_on="datetime",
        right_on="m15_datetime_key",
        direction="backward",  # Use most recent M15 candle that's <= M5 time
    )

    # Rename the key column to m15_datetime
    df_merged = df_merged.rename(columns={"m15_datetime_key": "m15_datetime"})

    return df_merged


def run_strategy_b(mode: str = "both"):
    """Run Strategy B backtest and/or forward test."""

    # Load and prepare data
    df = load_and_prepare_data()

    # Initialize
    bt = Backtester()
    strategy = StrategyB()

    results = {}

    if mode in ("both", "backtest"):
        print("\n" + "=" * 60)
        print("  STRATEGY B (FVG) — BACKTEST")
        print("=" * 60)
        strategy_bt = StrategyB()  # Fresh instance
        results["backtest"] = bt.run(df, strategy_bt, mode="backtest")
        bt.save_results(results["backtest"], "StrategyB_FVG_backtest.json")
        try:
            bt.plot_equity(
                results["backtest"],
                save_path=str(BACKTEST_DIR / "StrategyB_FVG_backtest_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    if mode in ("both", "forward"):
        print("\n" + "=" * 60)
        print("  STRATEGY B (FVG) — FORWARD TEST")
        print("=" * 60)
        strategy_fw = StrategyB()  # Fresh instance
        results["forward"] = bt.run(df, strategy_fw, mode="forward")
        bt.save_results(results["forward"], "StrategyB_FVG_forward.json")
        try:
            bt.plot_equity(
                results["forward"],
                save_path=str(BACKTEST_DIR / "StrategyB_FVG_forward_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    # ─── Comparison Summary (if both ran) ─────────────────
    if "backtest" in results and "forward" in results:
        print("\n" + "=" * 60)
        print("  STRATEGY B (FVG) — COMPARISON SUMMARY")
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
        ]

        for label, bt_val, fw_val, status in comparisons:
            print(f"  {label:<30} {bt_val:>12} {fw_val:>12} {status:>10}")

        # Overall assessment
        print("\n  " + "-" * 66)
        fw_passing = sum(1 for _, _, _, s in comparisons if s == "✅")
        if fw_passing >= 6:
            print("  ASSESSMENT: Strategy B PASSES initial validation ✅")
            print("  Ready for combined testing with other strategies.")
        elif fw_passing >= 4:
            print("  ASSESSMENT: Strategy B shows PROMISE but needs tuning ⚠️")
            print("  Review failing metrics before proceeding.")
        else:
            print("  ASSESSMENT: Strategy B FAILS validation ❌")
            print("  Major issues need fixing before moving to Strategy D.")

    print("\nResults saved to:", BACKTEST_DIR)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDAS v2 — Strategy B Runner")
    parser.add_argument("--mode", choices=["backtest", "forward", "both"],
                        default="both", help="Which test to run")
    args = parser.parse_args()

    results = run_strategy_b(args.mode)