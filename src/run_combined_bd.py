"""
Project MIDAS v2 — Combined Strategy B+D Runner
Runs both FVG and Trend strategies side by side with basic voting logic.

Voting Rules:
  - Only 1 strategy fires → take that signal
  - Both fire SAME direction → take higher confidence signal
  - Both fire OPPOSITE directions → skip (conflict)

Usage:
    python run_combined_bd.py                   # Full backtest + forward test
    python run_combined_bd.py --mode backtest   # Backtest only
    python run_combined_bd.py --mode forward    # Forward test only
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_RAW, DATA_PROCESSED, BACKTEST_DIR, ATR_PERIOD
from feature_engineering import FeatureEngine
from backtester import Backtester, StrategyBase, TradeSignal
from strategies.strategy_b import StrategyB
from strategies.strategy_d import StrategyD


# ─── Combined Strategy Wrapper ───────────────────────────

class CombinedBD(StrategyBase):
    """
    Runs Strategy B (FVG) and Strategy D (Trend) on every candle.
    Applies voting logic when both fire.
    """

    name = "Combined_B+D"

    def __init__(self):
        self.strategy_b = StrategyB()
        self.strategy_d = StrategyD()
        self.diag = {
            "candles_processed": 0,
            "b_only": 0,
            "d_only": 0,
            "both_agree": 0,
            "both_conflict": 0,
            "no_signal": 0,
        }

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        """
        Run both strategies and apply voting logic.
        """
        self.diag["candles_processed"] += 1

        # Get signals from both strategies
        signal_b = self.strategy_b.generate_signal(df, idx)
        signal_d = self.strategy_d.generate_signal(df, idx)

        # ── Voting Logic ──
        if signal_b is None and signal_d is None:
            self.diag["no_signal"] += 1
            return None

        if signal_b is not None and signal_d is None:
            # Only B fires
            self.diag["b_only"] += 1
            signal_b.sub_conditions["source"] = "B_only"
            return signal_b

        if signal_b is None and signal_d is not None:
            # Only D fires
            self.diag["d_only"] += 1
            signal_d.sub_conditions["source"] = "D_only"
            return signal_d

        # Both fired — check agreement
        if signal_b.direction == signal_d.direction:
            # Same direction — take higher confidence
            self.diag["both_agree"] += 1
            if signal_b.confidence >= signal_d.confidence:
                winner = signal_b
                winner.sub_conditions["source"] = "B+D_agree_B_wins"
            else:
                winner = signal_d
                winner.sub_conditions["source"] = "B+D_agree_D_wins"

            # Boost confidence slightly when both agree (max 100)
            winner.confidence = min(100, winner.confidence + 10)
            winner.sub_conditions["both_agreed"] = True
            return winner
        else:
            # Opposite directions — conflict, skip
            self.diag["both_conflict"] += 1
            return None

    def print_diagnostics(self):
        """Print combined voting diagnostics."""
        d = self.diag
        total_signals = d["b_only"] + d["d_only"] + d["both_agree"]
        print(f"\n  COMBINED B+D VOTING DIAGNOSTICS:")
        print(f"  Candles processed:        {d['candles_processed']:>10,}")
        print(f"  No signal:                {d['no_signal']:>10,}")
        print(f"  Strategy B only:          {d['b_only']:>10,}")
        print(f"  Strategy D only:          {d['d_only']:>10,}")
        print(f"  Both agree (taken):       {d['both_agree']:>10,}")
        print(f"  Both conflict (skipped):  {d['both_conflict']:>10,}")
        print(f"  Total signals produced:   {total_signals:>10,}")

        if total_signals > 0:
            print(f"\n  Signal source breakdown:")
            print(f"    From B alone:  {d['b_only']:>6} ({d['b_only']/total_signals*100:.1f}%)")
            print(f"    From D alone:  {d['d_only']:>6} ({d['d_only']/total_signals*100:.1f}%)")
            print(f"    Both agreed:   {d['both_agree']:>6} ({d['both_agree']/total_signals*100:.1f}%)")

        # Print individual strategy diagnostics too
        print("\n  --- Strategy B internals ---")
        self.strategy_b.print_diagnostics()
        print("\n  --- Strategy D internals ---")
        self.strategy_d.print_diagnostics()


# ─── Data Loading ────────────────────────────────────────

def load_and_prepare_data() -> pd.DataFrame:
    """
    Load M5 and M15 data, compute ALL features needed by both strategies,
    and merge M15 features into M5.
    """
    engine = FeatureEngine()

    # ─── Load M15 ──────────────────────────────────────
    m15_processed = DATA_PROCESSED / "XAUUSD_M15_features.csv"
    if m15_processed.exists():
        print("Loading processed M15 data...")
        df_m15 = pd.read_csv(m15_processed, parse_dates=["datetime"])
    else:
        print("Processing M15 data from raw...")
        df_m15 = pd.read_csv(DATA_RAW / "XAUUSD_M15.csv", parse_dates=["datetime"])
        df_m15 = engine.compute_shared(df_m15, timeframe="M15")
        df_m15.to_csv(m15_processed, index=False)

    # Compute FVG features on M15 (needed by Strategy B)
    print("Detecting FVGs on M15...")
    df_m15 = engine.compute_fvg(df_m15)

    # ─── Load M5 ───────────────────────────────────────
    m5_processed = DATA_PROCESSED / "XAUUSD_M5_features.csv"
    if m5_processed.exists():
        print("Loading processed M5 data...")
        df_m5 = pd.read_csv(m5_processed, parse_dates=["datetime"])
    else:
        print("Processing M5 data from raw...")
        df_m5 = pd.read_csv(DATA_RAW / "XAUUSD_M5.csv", parse_dates=["datetime"])
        df_m5 = engine.compute_shared(df_m5, timeframe="M5")
        df_m5.to_csv(m5_processed, index=False)

    # Compute trend features on M5 (needed by Strategy D)
    print("Computing trend features on M5...")
    df_m5 = engine.compute_trend(df_m5)

    # ─── Merge M15 into M5 ────────────────────────────
    print("Merging M15 features into M5 (asof merge)...")
    df_merged = merge_timeframes(df_m5, df_m15)

    print(f"Merged dataset: {len(df_merged):,} M5 candles with {len(df_merged.columns)} columns")
    return df_merged


def merge_timeframes(df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    """
    Merge M15 features into M5 using as-of merge.
    Includes BOTH FVG columns (for B) and trend columns (for D).
    """
    df_m5 = df_m5.sort_values("datetime").reset_index(drop=True)
    df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

    # Select M15 columns — everything both strategies need
    m15_cols = ["datetime"]

    # FVG columns (Strategy B)
    fvg_cols = [c for c in df_m15.columns if c.startswith("fvg_")]
    m15_cols.extend(fvg_cols)

    # Trend/indicator columns (Strategy D + shared)
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

    # Deduplicate
    m15_cols = list(dict.fromkeys(m15_cols))

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


# ─── Main Runner ─────────────────────────────────────────

def run_combined(mode: str = "both"):
    """Run combined B+D backtest and/or forward test."""

    df = load_and_prepare_data()
    bt = Backtester()
    results = {}

    if mode in ("both", "backtest"):
        print("\n" + "=" * 60)
        print("  COMBINED B+D — BACKTEST")
        print("=" * 60)
        strategy = CombinedBD()
        results["backtest"] = bt.run(df, strategy, mode="backtest")
        bt.save_results(results["backtest"], "Combined_BD_backtest.json")
        strategy.print_diagnostics()
        try:
            bt.plot_equity(
                results["backtest"],
                save_path=str(BACKTEST_DIR / "Combined_BD_backtest_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    if mode in ("both", "forward"):
        print("\n" + "=" * 60)
        print("  COMBINED B+D — FORWARD TEST")
        print("=" * 60)
        strategy = CombinedBD()
        results["forward"] = bt.run(df, strategy, mode="forward")
        bt.save_results(results["forward"], "Combined_BD_forward.json")
        strategy.print_diagnostics()
        try:
            bt.plot_equity(
                results["forward"],
                save_path=str(BACKTEST_DIR / "Combined_BD_forward_equity.png")
            )
        except Exception as e:
            print(f"Plotting skipped: {e}")

    # ─── Comparison: Individual vs Combined ───────────
    if "backtest" in results and "forward" in results:
        print("\n" + "=" * 60)
        print("  COMBINED B+D — COMPARISON SUMMARY")
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
             "✅" if mf["avg_win_loss_ratio"] > 1.2 else "❌"),
            ("Profitable Months", f"{mb['profitable_months_pct']:.1f}%", f"{mf['profitable_months_pct']:.1f}%",
             "✅" if mf["profitable_months_pct"] > 60 else "❌"),
        ]

        for label, bt_val, fw_val, status in comparisons:
            print(f"  {label:<30} {bt_val:>12} {fw_val:>12} {status:>10}")

        print("\n  " + "-" * 66)

        # Individual strategy reference
        print(f"\n  REFERENCE — Individual Strategy Results (Forward Test):")
        print(f"  {'Strategy':<20} {'PnL':>10} {'WR':>8} {'PF':>8} {'Trades/Mo':>10}")
        print(f"  {'B (FVG)':.<20} {'$663':>10} {'70.4%':>8} {'1.71':>8} {'32.6':>10}")
        print(f"  {'D (Trend)':.<20} {'$36':>10} {'41.8%':>8} {'1.17':>8} {'2.7':>10}")
        # print(f"  {'B+D Combined':.<20} {f'${mf[\"total_pnl\"]:+.0f}':>10} {f'{mf[\"win_rate\"]:.1f}%':>8} {f'{mf[\"profit_factor\"]:.2f}':>8} {f'{mf[\"trades_per_month\"]:.1f}':>10}")
        print(
    f"  {'B+D Combined':.<20} "
    f"${mf['total_pnl']:+.0f} "
    f"{mf['win_rate']:.1f}% "
    f"{mf['profit_factor']:.2f} "
    f"{mf['trades_per_month']:.1f}"
)
        
        
        # Key question: is combined better than B alone?
        print(f"\n  KEY QUESTION: Is B+D combined better than B alone?")
        b_alone_pnl = 663.49  # From locked Strategy B forward test
        combined_pnl = mf["total_pnl"]
        if combined_pnl > b_alone_pnl:
            print(f"  YES — Combined ${combined_pnl:+.0f} > B alone ${b_alone_pnl:+.0f} ✅")
        elif combined_pnl > 0:
            print(f"  MIXED — Combined ${combined_pnl:+.0f} is profitable but < B alone ${b_alone_pnl:+.0f} ⚠️")
            print(f"  Strategy D may be adding noise. Consider keeping both but reviewing D's contribution.")
        else:
            print(f"  NO — Combined ${combined_pnl:+.0f} is worse than B alone ${b_alone_pnl:+.0f} ❌")
            print(f"  Strategy D is hurting. Consider dropping D or rethinking the combination.")

        fw_passing = sum(1 for _, _, _, s in comparisons if s == "✅")
        print(f"\n  OVERALL SCORE: {fw_passing}/10 forward test checks passed")

    print("\nResults saved to:", BACKTEST_DIR)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDAS v2 — Combined B+D Runner")
    parser.add_argument("--mode", choices=["backtest", "forward", "both"],
                        default="both", help="Which test to run")
    args = parser.parse_args()

    results = run_combined(args.mode)