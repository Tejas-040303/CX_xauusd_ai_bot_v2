"""
Project MIDAS v2 — Combined Strategy B+C+D Runner
Runs FVG, Session Breakout, and Trend strategies side by side with voting logic.

Voting Rules:
  - Only 1 strategy fires → take that signal
  - 2+ fire SAME direction → take highest confidence (+10% boost per agreement)
  - Any conflict (opposite directions) → skip all

Usage:
    python run_combined_bcd.py
    python run_combined_bcd.py --mode backtest
    python run_combined_bcd.py --mode forward
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_RAW, DATA_PROCESSED, BACKTEST_DIR, ATR_PERIOD
from feature_engineering import FeatureEngine
from backtester import Backtester, StrategyBase, TradeSignal
from strategies.strategy_b import StrategyB
from strategies.strategy_c import StrategyC
from strategies.strategy_d import StrategyD


class CombinedBCD(StrategyBase):
    """
    Runs Strategy B (FVG), C (Session Breakout), D (Trend) on every candle.
    Applies voting logic when multiple fire.
    """

    name = "Combined_B+C+D"

    def __init__(self):
        self.strategy_b = StrategyB()
        self.strategy_c = StrategyC()
        self.strategy_d = StrategyD()
        self.diag = {
            "candles_processed": 0,
            "b_only": 0,
            "c_only": 0,
            "d_only": 0,
            "two_agree": 0,
            "three_agree": 0,
            "conflict": 0,
            "no_signal": 0,
        }

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        self.diag["candles_processed"] += 1

        # Get signals from all three
        signal_b = self.strategy_b.generate_signal(df, idx)
        signal_c = self.strategy_c.generate_signal(df, idx)
        signal_d = self.strategy_d.generate_signal(df, idx)

        signals = []
        if signal_b is not None:
            signals.append(("B", signal_b))
        if signal_c is not None:
            signals.append(("C", signal_c))
        if signal_d is not None:
            signals.append(("D", signal_d))

        if len(signals) == 0:
            self.diag["no_signal"] += 1
            return None

        if len(signals) == 1:
            name, sig = signals[0]
            self.diag[f"{name.lower()}_only"] += 1
            sig.sub_conditions["source"] = f"{name}_only"
            return sig

        # Multiple signals — check direction agreement
        directions = [s.direction for _, s in signals]
        unique_dirs = set(directions)

        if len(unique_dirs) > 1:
            # Conflict — skip
            self.diag["conflict"] += 1
            return None

        # All agree on direction
        if len(signals) == 3:
            self.diag["three_agree"] += 1
        else:
            self.diag["two_agree"] += 1

        # Pick highest confidence
        best_name, best_signal = max(signals, key=lambda x: x[1].confidence)

        # Boost confidence by 10% per agreeing strategy
        boost = (len(signals) - 1) * 10
        best_signal.confidence = min(100, best_signal.confidence + boost)

        sources = "+".join(n for n, _ in signals)
        best_signal.sub_conditions["source"] = f"{sources}_agree_{best_name}_wins"
        best_signal.sub_conditions["strategies_agreed"] = len(signals)

        return best_signal

    def print_diagnostics(self):
        d = self.diag
        total = d["b_only"] + d["c_only"] + d["d_only"] + d["two_agree"] + d["three_agree"]
        print(f"\n  COMBINED B+C+D VOTING DIAGNOSTICS:")
        print(f"  Candles processed:        {d['candles_processed']:>10,}")
        print(f"  No signal:                {d['no_signal']:>10,}")
        print(f"  Strategy B only:          {d['b_only']:>10,}")
        print(f"  Strategy C only:          {d['c_only']:>10,}")
        print(f"  Strategy D only:          {d['d_only']:>10,}")
        print(f"  Two agree (taken):        {d['two_agree']:>10,}")
        print(f"  Three agree (taken):      {d['three_agree']:>10,}")
        print(f"  Conflict (skipped):       {d['conflict']:>10,}")
        print(f"  Total signals produced:   {total:>10,}")

        if total > 0:
            print(f"\n  Signal source breakdown:")
            print(f"    B alone:       {d['b_only']:>6} ({d['b_only']/total*100:.1f}%)")
            print(f"    C alone:       {d['c_only']:>6} ({d['c_only']/total*100:.1f}%)")
            print(f"    D alone:       {d['d_only']:>6} ({d['d_only']/total*100:.1f}%)")
            print(f"    2 agree:       {d['two_agree']:>6} ({d['two_agree']/total*100:.1f}%)")
            print(f"    3 agree:       {d['three_agree']:>6} ({d['three_agree']/total*100:.1f}%)")


def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare data with ALL features for B, C, D."""
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

    # FVG features on M15 (Strategy B)
    print("Detecting FVGs on M15...")
    df_m15 = engine.compute_fvg(df_m15)

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

    # Trend features on M5 (Strategy D)
    print("Computing trend features on M5...")
    df_m5 = engine.compute_trend(df_m5)

    # Session range features on M5 (Strategy C)
    print("Computing session range features on M5...")
    df_m5 = engine.compute_session_range(df_m5)

    # ─── Merge M15 into M5 ────────────────────────────
    print("Merging M15 features into M5 (asof merge)...")
    df_merged = merge_timeframes(df_m5, df_m15)

    print(f"Merged dataset: {len(df_merged):,} M5 candles with {len(df_merged.columns)} columns")
    return df_merged


def merge_timeframes(df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    """Merge M15 features into M5 — includes FVG + trend indicator columns."""
    df_m5 = df_m5.sort_values("datetime").reset_index(drop=True)
    df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

    m15_cols = ["datetime"]

    # FVG columns (Strategy B)
    fvg_cols = [c for c in df_m15.columns if c.startswith("fvg_")]
    m15_cols.extend(fvg_cols)

    # Indicator columns (Strategy D + shared)
    indicator_cols = [
        "ema_9", "ema_21", "ema_50", "ema_200",
        "ema_50_slope", "ema_stack",
        "ema_partial_bull", "ema_partial_bear",
        "above_ema200",
        f"atr_{ATR_PERIOD}", f"rsi_{ATR_PERIOD}",
        f"adx_{ATR_PERIOD}", "trend_strong", "trend_very_strong", "trend_weak",
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


def run_combined(mode: str = "both"):
    """Run combined B+C+D backtest and/or forward test."""

    df = load_and_prepare_data()
    bt = Backtester()
    results = {}

    if mode in ("both", "backtest"):
        print("\n" + "=" * 60)
        print("  COMBINED B+C+D — BACKTEST")
        print("=" * 60)
        strategy = CombinedBCD()
        results["backtest"] = bt.run(df, strategy, mode="backtest")
        bt.save_results(results["backtest"], "Combined_BCD_backtest.json")
        strategy.print_diagnostics()
        try:
            bt.plot_equity(
                results["backtest"],
                save_path=str(BACKTEST_DIR / "Combined_BCD_backtest_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    if mode in ("both", "forward"):
        print("\n" + "=" * 60)
        print("  COMBINED B+C+D — FORWARD TEST")
        print("=" * 60)
        strategy = CombinedBCD()
        results["forward"] = bt.run(df, strategy, mode="forward")
        bt.save_results(results["forward"], "Combined_BCD_forward.json")
        strategy.print_diagnostics()
        try:
            bt.plot_equity(
                results["forward"],
                save_path=str(BACKTEST_DIR / "Combined_BCD_forward_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    # ─── Comparison ───────────────────────────────────
    if "backtest" in results and "forward" in results:
        print("\n" + "=" * 60)
        print("  COMBINED B+C+D — COMPARISON SUMMARY")
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
             "✅" if mf["trades_per_month"] > 15 else "⚠️"),
            ("Sharpe Ratio", f"{mb['sharpe_ratio']:.2f}", f"{mf['sharpe_ratio']:.2f}",
             "✅" if mf["sharpe_ratio"] > 0 else "❌"),
            ("Avg RR", f"{mb['avg_rr_achieved']:.2f}", f"{mf['avg_rr_achieved']:.2f}",
             "✅" if mf["avg_rr_achieved"] > 0 else "❌"),
            ("Max Consec Loss", f"{mb['max_consecutive_losses']}", f"{mf['max_consecutive_losses']}",
             "✅" if mf["max_consecutive_losses"] < 6 else "❌"),
            ("Avg Win/Loss", f"{mb['avg_win_loss_ratio']:.2f}", f"{mf['avg_win_loss_ratio']:.2f}",
             "✅" if mf["avg_win_loss_ratio"] > 1.0 else "❌"),
            ("Profitable Months", f"{mb['profitable_months_pct']:.1f}%", f"{mf['profitable_months_pct']:.1f}%",
             "✅" if mf["profitable_months_pct"] > 60 else "❌"),
        ]

        for label, bt_val, fw_val, status in comparisons:
            print(f"  {label:<30} {bt_val:>12} {fw_val:>12} {status:>10}")

        print("\n  " + "-" * 66)

        # Reference table
        print(f"\n  REFERENCE — All Results (Forward Test):")
        print(f"  {'Strategy':<25} {'PnL':>10} {'WR':>8} {'PF':>8} {'Trades/Mo':>10} {'MaxDD':>8}")
        print(f"  {'-'*70}")
        print(f"  {'B (FVG)':<25} {'$663':>10} {'70.4%':>8} {'1.71':>8} {'32.6':>10} {'2.5%':>8}")
        print(f"  {'C (Session Breakout)':<25} {'$497':>10} {'51.6%':>8} {'1.25':>8} {'11.0':>10} {'6.2%':>8}")
        print(f"  {'D (Trend)':<25} {'$36':>10} {'41.8%':>8} {'1.17':>8} {'2.7':>10} {'1.2%':>8}")
        print(f"  {'B+D Combined':<25} {'$703':>10} {'68.4%':>8} {'1.63':>8} {'34.4':>10} {'3.0%':>8}")
        print(f"  {'B+C+D Combined':<25} "
      f"{f'${mf['total_pnl']:+.0f}':>10} "
      f"{f'{mf['win_rate']:.1f}%':>8} "
      f"{f'{mf['profit_factor']:.2f}':>8} "
      f"{f'{mf['trades_per_month']:.1f}':>10} "
      f"{f'{mf['max_drawdown_pct']:.1f}%':>8}")
        # Key comparison
        bd_pnl = 702.69
        combined_pnl = mf["total_pnl"]
        print(f"\n  B+C+D (${combined_pnl:+.0f}) vs B+D (${bd_pnl:+.0f}): ", end="")
        if combined_pnl > bd_pnl:
            print(f"Adding C improved by ${combined_pnl - bd_pnl:+.0f} ✅")
        else:
            print(f"Adding C reduced by ${combined_pnl - bd_pnl:.0f} ⚠️")

        fw_passing = sum(1 for _, _, _, s in comparisons if s == "✅")
        print(f"\n  OVERALL SCORE: {fw_passing}/10 forward test checks passed")

    print("\nResults saved to:", BACKTEST_DIR)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDAS v2 — Combined B+C+D Runner")
    parser.add_argument("--mode", choices=["backtest", "forward", "both"],
                        default="both", help="Which test to run")
    args = parser.parse_args()

    results = run_combined(args.mode)