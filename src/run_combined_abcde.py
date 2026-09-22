"""
Project MIDAS v2 — Combined A+B+C+D+E Runner (FIXED)
======================================================
Fixes from previous version:
  1. Direction uses lowercase "buy"/"sell" to match Backtester._check_exit()
  2. Validator uses raw probability thresholds (not mapped confidence)

Usage:
    python run_combined_abcde.py
    python run_combined_abcde.py --mode both
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_RAW, DATA_PROCESSED, MODELS_DIR, BACKTEST_DIR,
    ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE, SWING_LOOKBACK,
    LOT_SIZE_SINGLE, LOT_SIZE_MULTI, SPREAD_SIMULATION_PIPS,
)
from feature_engineering import FeatureEngine
from backtester import Backtester, StrategyBase, TradeSignal
from strategies.strategy_a import StrategyA
from strategies.strategy_b import StrategyB
from strategies.strategy_c import StrategyC
from strategies.strategy_d import StrategyD
from run_combined_abcd import CombinedABCD, load_and_prepare_data


# ─── Strategy E Configuration ────────────────────────────
E_PROB_THRESHOLD = 0.68       # Best threshold from execution test

# Validator thresholds — using RAW PROBABILITY (not mapped confidence)
# E's output distribution clusters 0.35-0.65, so thresholds must match
E_VAL_STRONG_AGREE = 0.60    # prob ≥ 0.60 same direction → strong agree
E_VAL_AGREE = 0.52           # prob ≥ 0.52 same direction → normal agree
# Below 0.52 same direction → weak agree
# Opposite direction with prob ≥ 0.60 → conflict, SKIP
E_VAL_CONFLICT_SKIP = 0.60   # prob ≥ 0.60 opposite direction → skip

# Confidence adjustments
E_STRONG_BOOST = 10
E_AGREE_BOOST = 0
E_WEAK_PENALTY = -10

# Independent signal settings
E_INDEPENDENT_MIN_PROB = E_PROB_THRESHOLD  # Same as standalone

# SL/TP for E independent
E_SL_ATR_MULT = 1.5
E_TP_RR_BASE = 1.5
E_TP_RR_HIGH = 2.5

# Walk-forward models
WF_MODEL_MAP = [
    {"model": "strategy_e_clean_W2.txt",
     "start": "2024-07-01", "end": "2024-12-31"},
    {"model": "strategy_e_clean_W3.txt",
     "start": "2025-01-01", "end": "2025-06-30"},
    {"model": "strategy_e_clean_final.txt",
     "start": "2025-07-01", "end": "2026-12-31"},
]


def precompute_e_predictions() -> dict:
    """Precompute E predictions. Returns {datetime_str: probability}."""
    print("\n  Loading Strategy E predictions...")

    meta_path = MODELS_DIR / "strategy_e_clean_metadata.json"
    if not meta_path.exists():
        print("  ERROR: Run strategy_e_clean_retrain.py first.")
        return {}

    with open(meta_path) as f:
        feature_cols = json.load(f)["feature_cols"]

    clean_path = DATA_PROCESSED / "strategy_e_train_ready_clean.parquet"
    if not clean_path.exists():
        print("  ERROR: Run strategy_e_clean_retrain.py first.")
        return {}

    df_clean = pd.read_parquet(clean_path)
    df_clean["datetime"] = pd.to_datetime(df_clean["datetime"])

    predictions = {}
    total = 0

    for seg in WF_MODEL_MAP:
        model_path = MODELS_DIR / seg["model"]
        if not model_path.exists():
            print(f"    WARNING: {seg['model']} not found")
            continue

        model = lgb.Booster(model_file=str(model_path))
        mask = ((df_clean["datetime"] >= seg["start"]) &
                (df_clean["datetime"] <= seg["end"]))
        seg_data = df_clean[mask]

        if len(seg_data) == 0:
            continue

        probs = model.predict(seg_data[feature_cols].values)
        for dt, prob in zip(seg_data["datetime"].values, probs):
            predictions[str(pd.Timestamp(dt))] = float(prob)
        total += len(probs)
        print(f"    {seg['model']}: {len(probs):,} predictions")

    print(f"  Total: {total:,} predictions")
    return predictions


class CombinedABCDE(StrategyBase):
    """
    A+B+C+D+E hybrid strategy.

    Key fixes:
      - Direction uses lowercase "buy"/"sell" matching Backtester
      - Validator uses raw probability thresholds (E clusters 0.35-0.65)
    """

    name = "Combined_A+B+C+D+E"

    def __init__(self, e_predictions: dict,
                 enable_validator: bool = True,
                 enable_independent: bool = True):
        self.strategy_a = StrategyA()
        self.strategy_b = StrategyB()
        self.strategy_c = StrategyC()
        self.strategy_d = StrategyD()

        self.e_predictions = e_predictions
        self.enable_validator = enable_validator
        self.enable_independent = enable_independent

        self.diag = {
            "candles": 0,
            "ad_signal": 0, "ad_conflict": 0, "ad_no_signal": 0,
            "e_strong_agree": 0, "e_agree": 0, "e_weak": 0,
            "e_conflict_skip": 0, "e_no_pred": 0,
            "e_indep_fired": 0, "e_indep_below": 0, "e_indep_no_pred": 0,
            "sources": {},
        }

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        self.diag["candles"] += 1

        # ── A-D signals ───────────────────────────────────
        signal_a = self.strategy_a.generate_signal(df, idx)
        signal_b = self.strategy_b.generate_signal(df, idx)
        signal_c = self.strategy_c.generate_signal(df, idx)
        signal_d = self.strategy_d.generate_signal(df, idx)

        signals = []
        if signal_a: signals.append(("A", signal_a))
        if signal_b: signals.append(("B", signal_b))
        if signal_c: signals.append(("C", signal_c))
        if signal_d: signals.append(("D", signal_d))

        # E prediction lookup
        bar_time = df.iloc[idx]["datetime"]
        e_prob = self.e_predictions.get(str(pd.Timestamp(bar_time)))

        # ── A-D fired ─────────────────────────────────────
        if signals:
            return self._handle_ad(signals, e_prob, df, idx)

        # ── A-D silent → E independent ────────────────────
        self.diag["ad_no_signal"] += 1
        if self.enable_independent:
            return self._handle_e_indep(e_prob, df, idx)
        return None

    def _handle_ad(self, signals, e_prob, df, idx):
        """Process A-D signals with E validation using raw probability."""
        # Direction check (A-D use lowercase "buy"/"sell")
        dirs = set(s.direction for _, s in signals)
        if len(dirs) > 1:
            self.diag["ad_conflict"] += 1
            return None

        self.diag["ad_signal"] += 1

        # Pick best + boost for agreement
        best_name, best_signal = max(signals, key=lambda x: x[1].confidence)
        n_agree = len(signals)
        boost = (n_agree - 1) * 10
        best_signal.confidence = min(100, best_signal.confidence + boost)

        sources = "+".join(n for n, _ in signals)

        # ── E validation using RAW PROBABILITY ────────────
        if self.enable_validator and e_prob is not None:
            # Determine E's directional opinion from raw prob
            # prob > 0.5 = E leans BUY, prob < 0.5 = E leans SELL
            # A-D use lowercase direction
            ad_direction = best_signal.direction  # "buy" or "sell"

            if ad_direction == "buy":
                # E agrees with BUY if prob > 0.5
                e_agrees = e_prob > 0.5
                e_strength = e_prob  # Higher = stronger BUY agreement
            else:
                # E agrees with SELL if prob < 0.5
                e_agrees = e_prob < 0.5
                e_strength = 1 - e_prob  # Higher = stronger SELL agreement

            if e_agrees:
                if e_strength >= E_VAL_STRONG_AGREE:
                    # Strong agreement
                    best_signal.confidence = min(100, best_signal.confidence + E_STRONG_BOOST)
                    self.diag["e_strong_agree"] += 1
                    sources += "+E_strong"
                elif e_strength >= E_VAL_AGREE:
                    # Normal agreement
                    best_signal.confidence = min(100, best_signal.confidence + E_AGREE_BOOST)
                    self.diag["e_agree"] += 1
                    sources += "+E_agree"
                else:
                    # Weak agreement (E barely leans same direction)
                    best_signal.confidence = max(0, best_signal.confidence + E_WEAK_PENALTY)
                    self.diag["e_weak"] += 1
                    sources += "+E_weak"
            else:
                # E disagrees
                if e_strength >= E_VAL_CONFLICT_SKIP:
                    # Strong disagreement → SKIP
                    self.diag["e_conflict_skip"] += 1
                    return None
                else:
                    # Mild disagreement → penalty
                    best_signal.confidence = max(0, best_signal.confidence + E_WEAK_PENALTY)
                    self.diag["e_weak"] += 1
                    sources += "+E_disagree"

        elif self.enable_validator:
            self.diag["e_no_pred"] += 1

        best_signal.sub_conditions["source"] = sources
        best_signal.sub_conditions["strategies_agreed"] = n_agree
        if e_prob is not None:
            best_signal.sub_conditions["e_prob"] = round(e_prob, 4)

        self.diag["sources"][sources] = self.diag["sources"].get(sources, 0) + 1
        return best_signal

    def _handle_e_indep(self, e_prob, df, idx):
        """
        E independent signal when A-D are silent.

        CRITICAL: Uses lowercase "buy"/"sell" to match Backtester._check_exit()
        """
        if e_prob is None:
            self.diag["e_indep_no_pred"] += 1
            return None

        # Check threshold
        if e_prob >= E_INDEPENDENT_MIN_PROB:
            direction = "buy"      # ← LOWERCASE to match Backtester
        elif e_prob <= (1 - E_INDEPENDENT_MIN_PROB):
            direction = "sell"     # ← LOWERCASE to match Backtester
        else:
            self.diag["e_indep_below"] += 1
            return None

        # Confidence from raw prob (for Backtester's CONFIDENCE_THRESHOLD check)
        # Map so that prob=0.68 → conf=~70, prob=0.75 → conf=~80
        effective_prob = max(e_prob, 1 - e_prob)
        confidence = 50 + (effective_prob - 0.5) * 100  # Linear: 0.68 → 68, 0.75 → 75
        confidence = max(50, min(100, confidence))

        row = df.iloc[idx]
        atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
        if pd.isna(atr) or atr <= 0:
            atr = 2.0

        close = row["close"]
        spread = SPREAD_SIMULATION_PIPS * PIP_VALUE

        sl_dist = E_SL_ATR_MULT * atr
        rr = E_TP_RR_HIGH if confidence >= 75 else E_TP_RR_BASE
        tp_dist = sl_dist * rr

        if direction == "buy":
            entry = close + spread
            sl, tp = entry - sl_dist, entry + tp_dist
        else:
            entry = close - spread
            sl, tp = entry + sl_dist, entry - tp_dist

        signal = TradeSignal(
            datetime=row["datetime"],
            direction=direction,           # ← lowercase
            strategy_name=self.name,
            confidence=confidence,
            entry_price=round(entry, 2),
            sl_price=round(sl, 2),
            tp_price=round(tp, 2),
            sub_conditions={
                "source": "E_independent",
                "e_prob": round(e_prob, 4),
                "strategies_agreed": 0,
            },
        )

        self.diag["e_indep_fired"] += 1
        self.diag["sources"]["E_independent"] = \
            self.diag["sources"].get("E_independent", 0) + 1
        return signal

    def print_diagnostics(self):
        d = self.diag
        total = d["ad_signal"] + d["e_indep_fired"]

        print(f"\n  A+B+C+D+E DIAGNOSTICS:")
        print(f"  {'─' * 50}")
        print(f"  Candles:            {d['candles']:>10,}")
        print(f"  A-D signals:        {d['ad_signal']:>10,}")
        print(f"  A-D conflicts:      {d['ad_conflict']:>10,}")
        print(f"  A-D no signal:      {d['ad_no_signal']:>10,}")

        if self.enable_validator:
            print(f"\n  E VALIDATOR (raw prob thresholds):")
            print(f"    Strong agree (≥{E_VAL_STRONG_AGREE}): {d['e_strong_agree']:>8,}")
            print(f"    Normal agree (≥{E_VAL_AGREE}): {d['e_agree']:>8,}")
            print(f"    Weak/disagree:     {d['e_weak']:>10,}")
            print(f"    Conflict→SKIP:     {d['e_conflict_skip']:>10,}")
            print(f"    No E prediction:   {d['e_no_pred']:>10,}")
            validated = d['e_strong_agree'] + d['e_agree'] + d['e_weak'] + d['e_conflict_skip']
            if validated > 0:
                print(f"    Skip rate:         {d['e_conflict_skip']/validated*100:>9.1f}%")
                agree_rate = (d['e_strong_agree'] + d['e_agree']) / validated * 100
                print(f"    Agreement rate:    {agree_rate:>9.1f}%")

        if self.enable_independent:
            print(f"\n  E INDEPENDENT:")
            print(f"    Fired:            {d['e_indep_fired']:>10,}")
            print(f"    Below threshold:  {d['e_indep_below']:>10,}")
            print(f"    No prediction:    {d['e_indep_no_pred']:>10,}")

        if d["sources"]:
            print(f"\n  SIGNAL SOURCES (top 15):")
            for src, cnt in sorted(d["sources"].items(), key=lambda x: -x[1])[:15]:
                print(f"    {src:<40s}: {cnt:>5} ({cnt/max(total,1)*100:.1f}%)")


def run(mode="forward"):
    print("=" * 70)
    print("MIDAS v2 — Combined A+B+C+D+E (FIXED)")
    print(f"  Mode: {mode} | E threshold: {E_PROB_THRESHOLD}")
    print(f"  FIX 1: lowercase direction (buy/sell) for Backtester")
    print(f"  FIX 2: raw prob thresholds for validator")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    df = load_and_prepare_data()

    # E predictions
    print("\n[2/3] Strategy E predictions...")
    e_preds = precompute_e_predictions()
    if not e_preds:
        print("  No E predictions. Running A-D only.")

    # Run all 4 configurations
    print("\n[3/3] Running backtests...")
    bt = Backtester()
    results = {}

    configs = [
        ("abcd",         "A+B+C+D (baseline)",          False, False),
        ("abcd_e_val",   "A+B+C+D + E validator",       True,  False),
        ("abcd_e_indep", "A+B+C+D + E independent",     False, True),
        ("abcde",        "A+B+C+D+E (full hybrid)",     True,  True),
    ]

    for key, label, val, indep in configs:
        print(f"\n  Running {label}...")
        if key == "abcd":
            strat = CombinedABCD()
        else:
            strat = CombinedABCDE(e_preds, enable_validator=val, enable_independent=indep)

        results[key] = bt.run(df, strat, mode=mode)
        strat.print_diagnostics()

        m = results[key]["metrics"]
        print(f"  → PnL: ${m['total_pnl']:+,.2f} | WR: {m['win_rate']:.1f}% | "
              f"PF: {m['profit_factor']:.2f} | Trades: {m['num_trades']}")

        try:
            bt.save_results(results[key], f"Combined_{key.upper()}.json")
            bt.plot_equity(results[key],
                           save_path=str(BACKTEST_DIR / f"Combined_{key.upper()}_equity.png"))
        except Exception:
            pass

    # ── Comparison ────────────────────────────────────────
    print(f"\n{'=' * 95}")
    print("COMBINED COMPARISON")
    print(f"{'=' * 95}")
    print(f"  {'Mode':<34s} {'PnL':>9} {'WR%':>6} {'PF':>6} "
          f"{'Trades':>7} {'T/Mo':>6} {'MaxDD':>6} {'Sharpe':>7}")
    print(f"  {'-'*82}")

    for key, label, _, _ in configs:
        m = results[key]["metrics"]
        sharpe = m.get("sharpe_ratio", 0)
        print(f"  {label:<34s} ${m['total_pnl']:>7,.0f} "
              f"{m['win_rate']:>5.1f} {m['profit_factor']:>5.2f} "
              f"{m['num_trades']:>7} {m['trades_per_month']:>5.1f} "
              f"{m['max_drawdown_pct']:>5.1f}% {sharpe:>6.2f}")

    # ── E Contribution ────────────────────────────────────
    abcd_pnl = results["abcd"]["metrics"]["total_pnl"]
    abcde_pnl = results["abcde"]["metrics"]["total_pnl"]
    val_pnl = results["abcd_e_val"]["metrics"]["total_pnl"]
    indep_pnl = results["abcd_e_indep"]["metrics"]["total_pnl"]

    e_total = abcde_pnl - abcd_pnl
    val_delta = val_pnl - abcd_pnl
    indep_delta = indep_pnl - abcd_pnl

    print(f"\n{'=' * 70}")
    print("STRATEGY E CONTRIBUTION")
    print(f"{'=' * 70}")
    print(f"  A-D baseline:          ${abcd_pnl:>+10,.2f}")
    print(f"  + E validator only:    ${val_pnl:>+10,.2f}  (Δ ${val_delta:>+,.2f})")
    print(f"  + E independent only:  ${indep_pnl:>+10,.2f}  (Δ ${indep_delta:>+,.2f})")
    print(f"  + E both (hybrid):     ${abcde_pnl:>+10,.2f}  (Δ ${e_total:>+,.2f})")

    if abcd_pnl != 0:
        lift = e_total / abs(abcd_pnl) * 100
        print(f"\n  Total lift: {lift:+.1f}%")

    # ── Verdict ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    for label, delta in [("Validator", val_delta), ("Independent", indep_delta), ("Combined", e_total)]:
        if delta > 0:
            print(f"  ✅ {label}: +${delta:,.2f}")
        elif delta > -50:
            print(f"  🟡 {label}: ${delta:,.2f} (negligible impact)")
        else:
            print(f"  ❌ {label}: ${delta:,.2f}")

    # Save
    save_data = {}
    for key in results:
        m = results[key]["metrics"]
        save_data[key] = {k: v for k, v in m.items()
                          if not isinstance(v, (pd.DataFrame, pd.Series, list))}
    save_data["e_contribution"] = {
        "total": round(e_total, 2),
        "validator": round(val_delta, 2),
        "independent": round(indep_delta, 2),
    }
    with open(BACKTEST_DIR / "combined_abcde_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("✅ Combined A+B+C+D+E backtest complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backtest", "forward", "both"],
                        default="forward")
    args = parser.parse_args()

    if args.mode == "both":
        for m in ["backtest", "forward"]:
            run(mode=m)
    else:
        run(mode=args.mode)