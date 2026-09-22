"""
Project MIDAS v2 — Phase 5: Strategy Tuning
Tests multiple combined configurations to find the optimal settings.

Tests:
  - Per-strategy confidence thresholds
  - Drop/keep Strategy D
  - Different threshold combinations
  
Selects winner based on forward test: Sharpe × PF (risk-adjusted return)

Usage:
    python run_phase5_tuning.py
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from config import DATA_RAW, DATA_PROCESSED, BACKTEST_DIR, ATR_PERIOD
from feature_engineering import FeatureEngine
from backtester import Backtester, StrategyBase, TradeSignal
from strategies.strategy_a import StrategyA
from strategies.strategy_b import StrategyB
from strategies.strategy_c import StrategyC
from strategies.strategy_d import StrategyD


# ─── Configurable Combined Strategy ─────────────────────

class TunableCombined(StrategyBase):
    """
    Combined strategy with per-strategy confidence thresholds
    and option to enable/disable individual strategies.
    """

    def __init__(
        self,
        thresh_a: int = 65,
        thresh_b: int = 65,
        thresh_c: int = 65,
        thresh_d: int = 65,
        enable_a: bool = True,
        enable_b: bool = True,
        enable_c: bool = True,
        enable_d: bool = True,
        label: str = "",
    ):
        self.thresh_a = thresh_a
        self.thresh_b = thresh_b
        self.thresh_c = thresh_c
        self.thresh_d = thresh_d
        self.enable_a = enable_a
        self.enable_b = enable_b
        self.enable_c = enable_c
        self.enable_d = enable_d
        self.label = label
        self.name = f"Tuned_{label}" if label else "Tuned_Combined"

        # Create strategies (they use config.CONFIDENCE_THRESHOLD internally)
        # We set it low so all signals pass internal filter,
        # then apply per-strategy thresholds here
        self.strategy_a = StrategyA() if enable_a else None
        self.strategy_b = StrategyB() if enable_b else None
        self.strategy_c = StrategyC() if enable_c else None
        self.strategy_d = StrategyD() if enable_d else None

        self.source_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "multi": 0, "conflict": 0}

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        signals = []

        if self.strategy_a:
            sig = self.strategy_a.generate_signal(df, idx)
            if sig and sig.confidence >= self.thresh_a:
                signals.append(("A", sig))

        if self.strategy_b:
            sig = self.strategy_b.generate_signal(df, idx)
            if sig and sig.confidence >= self.thresh_b:
                signals.append(("B", sig))

        if self.strategy_c:
            sig = self.strategy_c.generate_signal(df, idx)
            if sig and sig.confidence >= self.thresh_c:
                signals.append(("C", sig))

        if self.strategy_d:
            sig = self.strategy_d.generate_signal(df, idx)
            if sig and sig.confidence >= self.thresh_d:
                signals.append(("D", sig))

        if len(signals) == 0:
            return None

        if len(signals) == 1:
            name, sig = signals[0]
            self.source_counts[name] += 1
            sig.sub_conditions["source"] = f"{name}_only"
            return sig

        # Multiple — check direction
        directions = set(s.direction for _, s in signals)
        if len(directions) > 1:
            self.source_counts["conflict"] += 1
            return None

        # All agree
        self.source_counts["multi"] += 1
        best_name, best_signal = max(signals, key=lambda x: x[1].confidence)
        boost = (len(signals) - 1) * 10
        best_signal.confidence = min(100, best_signal.confidence + boost)
        sources = "+".join(n for n, _ in signals)
        best_signal.sub_conditions["source"] = f"{sources}_agree"
        return best_signal


# ─── Data Loading (shared across all configs) ───────────

def load_data() -> pd.DataFrame:
    """Load and prepare data with all features. Called ONCE."""
    engine = FeatureEngine()

    # M15
    m15_processed = DATA_PROCESSED / "XAUUSD_M15_features.csv"
    if m15_processed.exists():
        print("Loading processed M15 data...")
        df_m15 = pd.read_csv(m15_processed, parse_dates=["datetime"])
    else:
        print("Processing M15 data from raw...")
        df_m15 = pd.read_csv(DATA_RAW / "XAUUSD_M15.csv", parse_dates=["datetime"])
        df_m15 = engine.compute_shared(df_m15, timeframe="M15")
        df_m15.to_csv(m15_processed, index=False)

    print("Detecting FVGs on M15...")
    df_m15 = engine.compute_fvg(df_m15)

    print("Computing SMC features on M15...")
    df_m15 = engine.compute_smc(df_m15)

    # M5
    m5_processed = DATA_PROCESSED / "XAUUSD_M5_features.csv"
    if m5_processed.exists():
        print("Loading processed M5 data...")
        df_m5 = pd.read_csv(m5_processed, parse_dates=["datetime"])
    else:
        print("Processing M5 data from raw...")
        df_m5 = pd.read_csv(DATA_RAW / "XAUUSD_M5.csv", parse_dates=["datetime"])
        df_m5 = engine.compute_shared(df_m5, timeframe="M5")
        df_m5.to_csv(m5_processed, index=False)

    print("Computing trend features on M5...")
    df_m5 = engine.compute_trend(df_m5)

    print("Computing session range features on M5...")
    df_m5 = engine.compute_session_range(df_m5)

    # Merge
    print("Merging M15 features into M5...")
    df_merged = merge_timeframes(df_m5, df_m15)
    print(f"Merged: {len(df_merged):,} candles, {len(df_merged.columns)} columns\n")
    return df_merged


def merge_timeframes(df_m5, df_m15):
    """Full merge with all features."""
    df_m5 = df_m5.sort_values("datetime").reset_index(drop=True)
    df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

    m15_cols = ["datetime"]
    fvg_cols = [c for c in df_m15.columns if c.startswith("fvg_")]
    smc_cols = [c for c in df_m15.columns if c.startswith(("bos_", "ob_", "equal_", "last_equal_"))]
    m15_cols.extend(fvg_cols)
    m15_cols.extend(smc_cols)
    m15_cols.extend(["open", "high", "low", "close"])

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
        df_m5, df_m15_subset,
        left_on="datetime", right_on="m15_datetime_key",
        direction="backward",
    )
    df_merged = df_merged.rename(columns={"m15_datetime_key": "m15_datetime"})
    return df_merged


# ─── Configuration Definitions ───────────────────────────

CONFIGS = [
    # (label, thresh_a, thresh_b, thresh_c, thresh_d, enable_a, enable_b, enable_c, enable_d)
    ("Baseline_65",           65, 65, 65, 65, True, True, True, True),
    ("Raise_C75",             65, 65, 75, 65, True, True, True, True),
    ("Drop_D",                65, 65, 65, 65, True, True, True, False),
    ("All_70",                70, 70, 70, 70, True, True, True, True),
    ("A60_B65_C75_D70",       60, 65, 75, 70, True, True, True, True),
    ("Drop_D_Raise_C75",      65, 65, 75, 65, True, True, True, False),
    ("A70_B65_C65_D65",       70, 65, 65, 65, True, True, True, True),
    ("A60_B65_C65_D65",       60, 65, 65, 65, True, True, True, True),
    ("A65_B70_C70_D65",       65, 70, 70, 65, True, True, True, True),
    ("Drop_CD",               65, 65, 65, 65, True, True, False, False),
    ("A70_B65_C75_D_off",     70, 65, 75, 65, True, True, True, False),
    ("A60_B65_C70_D_off",     60, 65, 70, 65, True, True, True, False),
]


# ─── Runner ──────────────────────────────────────────────

def run_tuning():
    """Run all configurations and compare."""

    # Lower global threshold so strategies generate all signals
    # Per-strategy filtering happens in TunableCombined
    original_threshold = config.CONFIDENCE_THRESHOLD
    config.CONFIDENCE_THRESHOLD = 55  # Below any config we test

    # Load data ONCE
    print("=" * 70)
    print("  PHASE 5: STRATEGY TUNING — Loading Data")
    print("=" * 70)
    df = load_data()

    bt = Backtester(verbose=False)

    results_table = []

    print("=" * 70)
    print(f"  PHASE 5: Testing {len(CONFIGS)} configurations")
    print("=" * 70)

    for i, (label, ta, tb, tc, td, ea, eb, ec, ed) in enumerate(CONFIGS):
        active = []
        if ea: active.append(f"A{ta}")
        if eb: active.append(f"B{tb}")
        if ec: active.append(f"C{tc}")
        if ed: active.append(f"D{td}")
        active_str = "/".join(active)

        print(f"\n  [{i+1}/{len(CONFIGS)}] {label} ({active_str})")

        start_time = time.time()

        # Run backtest
        strategy_bt = TunableCombined(
            thresh_a=ta, thresh_b=tb, thresh_c=tc, thresh_d=td,
            enable_a=ea, enable_b=eb, enable_c=ec, enable_d=ed,
            label=label,
        )
        res_bt = bt.run(df, strategy_bt, mode="backtest")

        # Run forward test
        strategy_fw = TunableCombined(
            thresh_a=ta, thresh_b=tb, thresh_c=tc, thresh_d=td,
            enable_a=ea, enable_b=eb, enable_c=ec, enable_d=ed,
            label=label,
        )
        res_fw = bt.run(df, strategy_fw, mode="forward")

        elapsed = time.time() - start_time

        mb = res_bt["metrics"]
        mf = res_fw["metrics"]

        # Composite score: Sharpe × PF (risk-adjusted return)
        composite = mf["sharpe_ratio"] * mf["profit_factor"] if mf["profit_factor"] > 0 else 0

        # Count passing checks
        checks = [
            mf["total_pnl"] > 0,
            mf["win_rate"] > 50,
            mf["profit_factor"] > 1.5,
            mf["max_drawdown_pct"] < 8,
            mf["avg_win_loss_ratio"] > 1.0,
            mf["max_consecutive_losses"] < 6,
            mf["profitable_months_pct"] > 60,
            mf["trades_per_month"] > 15,
            mf["sharpe_ratio"] > 0,
            mf["avg_rr_achieved"] > 0,
        ]
        score = sum(1 for c in checks if c)

        row = {
            "config": label,
            "strategies": active_str,
            "bt_pnl": mb["total_pnl"],
            "bt_pf": mb["profit_factor"],
            "bt_wr": mb["win_rate"],
            "bt_trades": mb["num_trades"],
            "fw_pnl": mf["total_pnl"],
            "fw_pf": mf["profit_factor"],
            "fw_wr": mf["win_rate"],
            "fw_sharpe": mf["sharpe_ratio"],
            "fw_maxdd": mf["max_drawdown_pct"],
            "fw_max_cl": mf["max_consecutive_losses"],
            "fw_trades": mf["num_trades"],
            "fw_tr_mo": mf["trades_per_month"],
            "fw_wl": mf["avg_win_loss_ratio"],
            "fw_prof_mo": mf["profitable_months_pct"],
            "composite": round(composite, 2),
            "score": f"{score}/10",
            "time_s": round(elapsed, 1),
            "source_a": strategy_fw.source_counts["A"],
            "source_b": strategy_fw.source_counts["B"],
            "source_c": strategy_fw.source_counts["C"],
            "source_d": strategy_fw.source_counts["D"],
        }
        results_table.append(row)

        print(f"    BT: ${mb['total_pnl']:+.0f} | FW: ${mf['total_pnl']:+.0f} | "
              f"WR: {mf['win_rate']:.1f}% | PF: {mf['profit_factor']:.2f} | "
              f"Sharpe: {mf['sharpe_ratio']:.2f} | MaxDD: {mf['max_drawdown_pct']:.1f}% | "
              f"Score: {score}/10 | Composite: {composite:.2f} | "
              f"[{elapsed:.0f}s]")

    # ─── Results Comparison ───────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 5: TUNING RESULTS COMPARISON (Forward Test)")
    print("=" * 70)

    # Sort by composite score
    results_table.sort(key=lambda x: x["composite"], reverse=True)

    print(f"\n  {'#':<3} {'Config':<25} {'FW PnL':>10} {'WR':>7} {'PF':>7} "
          f"{'Sharpe':>8} {'MaxDD':>7} {'MaxCL':>6} {'Tr/Mo':>7} {'Prof%':>7} "
          f"{'Score':>7} {'Comp':>7}")
    print("  " + "-" * 112)

    for i, r in enumerate(results_table):
        marker = " 🏆" if i == 0 else ("  ⬆" if i <= 2 else "")
        print(f"  {i+1:<3} {r['config']:<25} ${r['fw_pnl']:>+8,.0f} "
              f"{r['fw_wr']:>6.1f}% {r['fw_pf']:>6.2f} "
              f"{r['fw_sharpe']:>7.2f} {r['fw_maxdd']:>6.1f}% "
              f"{r['fw_max_cl']:>5} {r['fw_tr_mo']:>6.1f} "
              f"{r['fw_prof_mo']:>6.1f}% {r['score']:>6} "
              f"{r['composite']:>6.2f}{marker}")

    # Winner
    winner = results_table[0]
    print(f"\n  {'='*112}")
    print(f"  🏆 WINNER: {winner['config']}")
    print(f"     Strategies: {winner['strategies']}")
    print(f"     FW PnL: ${winner['fw_pnl']:+,.2f}")
    print(f"     PF: {winner['fw_pf']:.2f} | Sharpe: {winner['fw_sharpe']:.2f}")
    print(f"     MaxDD: {winner['fw_maxdd']:.1f}% | Max Consec Loss: {winner['fw_max_cl']}")
    print(f"     Trades/Month: {winner['fw_tr_mo']:.1f}")
    print(f"     Score: {winner['score']} | Composite: {winner['composite']:.2f}")

    # Signal source for winner
    print(f"\n     Signal sources (FW): A={winner['source_a']} B={winner['source_b']} "
          f"C={winner['source_c']} D={winner['source_d']}")

    # Top 3 comparison
    print(f"\n  TOP 3 CONFIGS:")
    for i, r in enumerate(results_table[:3]):
        print(f"    {i+1}. {r['config']:<25} ${r['fw_pnl']:>+8,.0f}  PF:{r['fw_pf']:.2f}  "
              f"Sharpe:{r['fw_sharpe']:.2f}  MaxDD:{r['fw_maxdd']:.1f}%")

    # vs baseline
    baseline = next((r for r in results_table if r["config"] == "Baseline_65"), None)
    if baseline and winner["config"] != "Baseline_65":
        print(f"\n  Winner vs Baseline:")
        print(f"    PnL:    ${winner['fw_pnl']:+,.0f} vs ${baseline['fw_pnl']:+,.0f} "
              f"({'better' if winner['fw_pnl'] > baseline['fw_pnl'] else 'worse'})")
        print(f"    PF:     {winner['fw_pf']:.2f} vs {baseline['fw_pf']:.2f}")
        print(f"    Sharpe: {winner['fw_sharpe']:.2f} vs {baseline['fw_sharpe']:.2f}")
        print(f"    MaxDD:  {winner['fw_maxdd']:.1f}% vs {baseline['fw_maxdd']:.1f}%")

    # Save results
    results_df = pd.DataFrame(results_table)
    output_path = BACKTEST_DIR / "phase5_tuning_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n  Full results saved to: {output_path}")

    # Restore original threshold
    config.CONFIDENCE_THRESHOLD = original_threshold

    print(f"\n  THIS WINNER BECOMES THE RULE-BASED BASELINE FOR AI PHASES.")
    print(f"  Update config.py with the winning thresholds before proceeding.\n")

    return results_table


if __name__ == "__main__":
    results = run_tuning()