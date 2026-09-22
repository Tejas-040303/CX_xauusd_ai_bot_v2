"""
Project MIDAS v2 — Phase 9: Regime-Aware Combined A+B+C+D+E
=============================================================
Integrates the regime detector into the hybrid system.
Runs 8 configurations: 4 without regime × 4 with regime.

Regime adjustments per market condition:
  TRENDING:  Lower E threshold, slight confidence boost
  VOLATILE:  Higher E threshold, wider SL, confidence penalty
  QUIET:     Disable E independent, tight SL/TP
  RANGING:   Moderate E threshold, tighter TP

Usage:
    python run_phase9_regime.py
    python run_phase9_regime.py --mode both
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
    ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE,
    SPREAD_SIMULATION_PIPS, LOT_SIZE_SINGLE,
)
from feature_engineering import FeatureEngine
from backtester import Backtester, StrategyBase, TradeSignal
from strategies.strategy_a import StrategyA
from strategies.strategy_b import StrategyB
from strategies.strategy_c import StrategyC
from strategies.strategy_d import StrategyD
from run_combined_abcd import CombinedABCD, load_and_prepare_data
from regime_detector import RegimeDetector, REGIME_ADJUSTMENTS, REGIMES


# ─── Base E Config (no regime) ────────────────────────────
E_PROB_THRESHOLD = 0.68
E_VAL_STRONG_AGREE = 0.60
E_VAL_AGREE = 0.52
E_VAL_CONFLICT_SKIP = 0.60
E_STRONG_BOOST = 10
E_WEAK_PENALTY = -10
E_SL_ATR_MULT = 1.5
E_TP_RR_BASE = 1.5
E_TP_RR_HIGH = 2.5
E_INDEPENDENT_MIN_CONF = 70

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
    """Load clean models and precompute E predictions."""
    meta_path = MODELS_DIR / "strategy_e_clean_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        feature_cols = json.load(f)["feature_cols"]

    clean_path = DATA_PROCESSED / "strategy_e_train_ready_clean.parquet"
    if not clean_path.exists():
        return {}

    df_clean = pd.read_parquet(clean_path)
    df_clean["datetime"] = pd.to_datetime(df_clean["datetime"])

    predictions = {}
    for seg in WF_MODEL_MAP:
        model_path = MODELS_DIR / seg["model"]
        if not model_path.exists():
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

    print(f"  E predictions: {len(predictions):,}")
    return predictions


class CombinedABCDE_Regime(StrategyBase):
    """
    Regime-aware A+B+C+D+E hybrid strategy.
    Adjusts E's behavior based on detected market regime.
    """

    name = "Combined_A+B+C+D+E_Regime"

    def __init__(self, e_predictions: dict,
                 enable_validator: bool = True,
                 enable_independent: bool = True,
                 use_regime: bool = True):
        self.strategy_a = StrategyA()
        self.strategy_b = StrategyB()
        self.strategy_c = StrategyC()
        self.strategy_d = StrategyD()

        self.e_predictions = e_predictions
        self.enable_validator = enable_validator
        self.enable_independent = enable_independent
        self.use_regime = use_regime

        self.regime_detector = RegimeDetector() if use_regime else None

        self.diag = {
            "candles": 0,
            "ad_signal": 0, "ad_conflict": 0, "ad_no_signal": 0,
            "e_strong_agree": 0, "e_agree": 0, "e_weak": 0,
            "e_conflict_skip": 0, "e_no_pred": 0,
            "e_indep_fired": 0, "e_indep_below": 0, "e_indep_no_pred": 0,
            "e_indep_regime_blocked": 0,
            "regime_counts": {r: 0 for r in REGIMES},
            "regime_trade_counts": {r: 0 for r in REGIMES},
            "regime_trade_pnl": {r: 0.0 for r in REGIMES},
            "sources": {},
        }

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        self.diag["candles"] += 1
        row = df.iloc[idx]

        # ── Detect regime ─────────────────────────────────
        if self.use_regime:
            regime, adj = self.regime_detector.detect_and_adjust(row)
            self.diag["regime_counts"][regime] += 1
        else:
            regime = "RANGING"
            adj = REGIME_ADJUSTMENTS["RANGING"]

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

        # E prediction
        bar_time = row["datetime"]
        e_prob = self.e_predictions.get(str(pd.Timestamp(bar_time)))

        # ── A-D fired ─────────────────────────────────────
        if signals:
            signal = self._handle_ad(signals, e_prob, adj, regime, df, idx)
            if signal:
                signal.sub_conditions["regime"] = regime
            return signal

        # ── A-D silent → E independent ────────────────────
        self.diag["ad_no_signal"] += 1
        if self.enable_independent:
            signal = self._handle_e_indep(e_prob, adj, regime, df, idx)
            if signal:
                signal.sub_conditions["regime"] = regime
            return signal
        return None

    def _handle_ad(self, signals, e_prob, adj, regime, df, idx):
        """Process A-D signals with regime-adjusted E validation."""
        dirs = set(s.direction for _, s in signals)
        if len(dirs) > 1:
            self.diag["ad_conflict"] += 1
            return None

        self.diag["ad_signal"] += 1
        best_name, best_signal = max(signals, key=lambda x: x[1].confidence)
        n_agree = len(signals)
        boost = (n_agree - 1) * 10
        best_signal.confidence = min(100, best_signal.confidence + boost)

        sources = "+".join(n for n, _ in signals)

        # ── Regime-adjusted E validation ──────────────────
        if self.enable_validator and e_prob is not None:
            ad_direction = best_signal.direction

            # Use regime-specific thresholds
            val_strong = adj.get("e_val_strong_agree", E_VAL_STRONG_AGREE)
            val_agree = adj.get("e_val_agree", E_VAL_AGREE)
            val_conflict = adj.get("e_val_conflict_skip", E_VAL_CONFLICT_SKIP)

            if ad_direction == "buy":
                e_agrees = e_prob > 0.5
                e_strength = e_prob
            else:
                e_agrees = e_prob < 0.5
                e_strength = 1 - e_prob

            if e_agrees:
                if e_strength >= val_strong:
                    conf_adj = E_STRONG_BOOST + adj.get("e_conf_boost", 0)
                    best_signal.confidence = min(100, best_signal.confidence + conf_adj)
                    self.diag["e_strong_agree"] += 1
                    sources += "+E_strong"
                elif e_strength >= val_agree:
                    conf_adj = adj.get("e_conf_boost", 0)
                    best_signal.confidence = min(100, best_signal.confidence + conf_adj)
                    self.diag["e_agree"] += 1
                    sources += "+E_agree"
                else:
                    best_signal.confidence = max(0, best_signal.confidence + E_WEAK_PENALTY)
                    self.diag["e_weak"] += 1
                    sources += "+E_weak"
            else:
                if e_strength >= val_conflict:
                    self.diag["e_conflict_skip"] += 1
                    return None
                else:
                    best_signal.confidence = max(0, best_signal.confidence + E_WEAK_PENALTY)
                    self.diag["e_weak"] += 1
                    sources += "+E_disagree"
        elif self.enable_validator:
            self.diag["e_no_pred"] += 1

        # ── Regime-adjusted SL/TP ─────────────────────────
        if self.use_regime:
            sl_mult = adj.get("sl_mult", 1.0)
            tp_mult = adj.get("tp_mult", 1.0)
            if sl_mult != 1.0:
                # Adjust SL distance
                entry = best_signal.entry_price
                sl_dist = abs(entry - best_signal.sl_price)
                new_sl_dist = sl_dist * sl_mult
                if best_signal.direction == "buy":
                    best_signal.sl_price = round(entry - new_sl_dist, 2)
                else:
                    best_signal.sl_price = round(entry + new_sl_dist, 2)
            if tp_mult != 1.0:
                entry = best_signal.entry_price
                tp_dist = abs(entry - best_signal.tp_price)
                new_tp_dist = tp_dist * tp_mult
                if best_signal.direction == "buy":
                    best_signal.tp_price = round(entry + new_tp_dist, 2)
                else:
                    best_signal.tp_price = round(entry - new_tp_dist, 2)

        best_signal.sub_conditions["source"] = sources
        best_signal.sub_conditions["strategies_agreed"] = n_agree
        self.diag["sources"][sources] = self.diag["sources"].get(sources, 0) + 1
        self.diag["regime_trade_counts"][regime] += 1
        return best_signal

    def _handle_e_indep(self, e_prob, adj, regime, df, idx):
        """Regime-adjusted E independent signal."""
        if e_prob is None:
            self.diag["e_indep_no_pred"] += 1
            return None

        # Check if regime allows independent signals
        if self.use_regime and not adj.get("allow_e_independent", True):
            self.diag["e_indep_regime_blocked"] += 1
            return None

        # Use regime-specific threshold
        threshold = adj.get("e_indep_threshold", E_PROB_THRESHOLD)

        if e_prob >= threshold:
            direction = "buy"
        elif e_prob <= (1 - threshold):
            direction = "sell"
        else:
            self.diag["e_indep_below"] += 1
            return None

        effective_prob = max(e_prob, 1 - e_prob)
        confidence = 50 + (effective_prob - 0.5) * 100
        confidence += adj.get("e_conf_boost", 0)
        confidence = max(50, min(100, confidence))

        if confidence < E_INDEPENDENT_MIN_CONF + adj.get("e_conf_boost", 0):
            self.diag["e_indep_below"] += 1
            return None

        row = df.iloc[idx]
        atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
        if pd.isna(atr) or atr <= 0:
            atr = 2.0

        close = row["close"]
        spread = SPREAD_SIMULATION_PIPS * PIP_VALUE

        sl_dist = E_SL_ATR_MULT * atr * adj.get("sl_mult", 1.0)
        rr = E_TP_RR_HIGH if confidence >= 75 else E_TP_RR_BASE
        tp_dist = sl_dist * rr * adj.get("tp_mult", 1.0)

        if direction == "buy":
            entry = close + spread
            sl, tp = entry - sl_dist, entry + tp_dist
        else:
            entry = close - spread
            sl, tp = entry + sl_dist, entry - tp_dist

        signal = TradeSignal(
            datetime=row["datetime"],
            direction=direction,
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
        self.diag["regime_trade_counts"][regime] += 1
        return signal

    def print_diagnostics(self):
        d = self.diag
        total = d["ad_signal"] + d["e_indep_fired"]

        print(f"\n  {'A+B+C+D+E' + (' + REGIME' if self.use_regime else '')} DIAGNOSTICS:")
        print(f"  {'─' * 55}")
        print(f"  Candles:            {d['candles']:>10,}")
        print(f"  A-D signals:        {d['ad_signal']:>10,}")
        print(f"  A-D conflicts:      {d['ad_conflict']:>10,}")

        if self.enable_validator:
            print(f"\n  E VALIDATOR:")
            print(f"    Strong agree:     {d['e_strong_agree']:>10,}")
            print(f"    Normal agree:     {d['e_agree']:>10,}")
            print(f"    Weak/disagree:    {d['e_weak']:>10,}")
            print(f"    Conflict→SKIP:    {d['e_conflict_skip']:>10,}")
            validated = d['e_strong_agree'] + d['e_agree'] + d['e_weak'] + d['e_conflict_skip']
            if validated > 0:
                print(f"    Agreement rate:   {(d['e_strong_agree']+d['e_agree'])/validated*100:>9.1f}%")
                print(f"    Skip rate:        {d['e_conflict_skip']/validated*100:>9.1f}%")

        if self.enable_independent:
            print(f"\n  E INDEPENDENT:")
            print(f"    Fired:            {d['e_indep_fired']:>10,}")
            print(f"    Below threshold:  {d['e_indep_below']:>10,}")
            if self.use_regime:
                print(f"    Regime blocked:   {d['e_indep_regime_blocked']:>10,}")

        if self.use_regime and self.regime_detector:
            self.regime_detector.print_stats()
            print(f"\n  TRADES PER REGIME:")
            for r in REGIMES:
                cnt = d["regime_trade_counts"][r]
                if cnt > 0:
                    print(f"    {r:<16s}: {cnt:>5} trades")


def run(mode="forward"):
    print("=" * 70)
    print("MIDAS v2 — Phase 9: Regime-Aware A+B+C+D+E")
    print(f"  Mode: {mode}")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    df = load_and_prepare_data()

    # E predictions
    print("\n[2/3] Strategy E predictions...")
    e_preds = precompute_e_predictions()

    # Run all 8 configurations
    print("\n[3/3] Running backtests...")
    bt = Backtester()
    results = {}

    configs = [
        # Without regime
        ("abcd",              "A+B+C+D (baseline)",       False, False, False),
        ("abcde_no_regime",   "A+B+C+D+E (no regime)",    True,  True,  False),
        # With regime
        ("abcd_regime",       "A+B+C+D + regime",         False, False, True),
        ("abcde_val_regime",  "A-D + E_val + regime",      True,  False, True),
        ("abcde_ind_regime",  "A-D + E_indep + regime",    False, True,  True),
        ("abcde_regime",      "A+B+C+D+E + regime",       True,  True,  True),
    ]

    for key, label, val, indep, regime in configs:
        print(f"\n  Running {label}...")

        if key == "abcd":
            strat = CombinedABCD()
        elif key == "abcd_regime":
            # A-D only but with regime SL/TP adjustments
            strat = CombinedABCDE_Regime(
                e_preds, enable_validator=False, enable_independent=False,
                use_regime=True
            )
        else:
            strat = CombinedABCDE_Regime(
                e_preds, enable_validator=val, enable_independent=indep,
                use_regime=regime
            )

        results[key] = bt.run(df, strat, mode=mode)
        strat.print_diagnostics()

        m = results[key]["metrics"]
        print(f"  → PnL: ${m['total_pnl']:+,.2f} | WR: {m['win_rate']:.1f}% | "
              f"PF: {m['profit_factor']:.2f} | Trades: {m['num_trades']}")

        try:
            bt.save_results(results[key], f"Phase9_{key.upper()}.json")
            bt.plot_equity(results[key],
                           save_path=str(BACKTEST_DIR / f"Phase9_{key.upper()}_equity.png"))
        except Exception:
            pass

    # ── Comparison ────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("PHASE 9 — FULL COMPARISON")
    print(f"{'=' * 100}")
    print(f"  {'Mode':<35s} {'PnL':>9} {'WR%':>6} {'PF':>6} "
          f"{'Trades':>7} {'T/Mo':>6} {'MaxDD':>6} {'Sharpe':>7}")
    print(f"  {'-'*85}")

    for key, label, _, _, _ in configs:
        if key not in results:
            continue
        m = results[key]["metrics"]
        sharpe = m.get("sharpe_ratio", 0)
        print(f"  {label:<35s} ${m['total_pnl']:>7,.0f} "
              f"{m['win_rate']:>5.1f} {m['profit_factor']:>5.2f} "
              f"{m['num_trades']:>7} {m['trades_per_month']:>5.1f} "
              f"{m['max_drawdown_pct']:>5.1f}% {sharpe:>6.2f}")

    # ── Regime Impact Analysis ────────────────────────────
    print(f"\n{'=' * 70}")
    print("REGIME IMPACT ANALYSIS")
    print(f"{'=' * 70}")

    # Compare no-regime vs regime for each config
    comparisons = [
        ("A+B+C+D+E hybrid", "abcde_no_regime", "abcde_regime"),
    ]

    for label, no_key, with_key in comparisons:
        if no_key in results and with_key in results:
            no_m = results[no_key]["metrics"]
            with_m = results[with_key]["metrics"]
            delta_pnl = with_m["total_pnl"] - no_m["total_pnl"]
            delta_wr = with_m["win_rate"] - no_m["win_rate"]
            delta_pf = with_m["profit_factor"] - no_m["profit_factor"]
            delta_dd = with_m["max_drawdown_pct"] - no_m["max_drawdown_pct"]

            print(f"\n  {label}:")
            print(f"    {'Metric':<20s} {'No Regime':>12s} {'With Regime':>14s} {'Delta':>10s}")
            print(f"    {'-'*56}")
            print(f"    {'PnL':<20s} ${no_m['total_pnl']:>10,.0f} ${with_m['total_pnl']:>12,.0f} ${delta_pnl:>+9,.0f}")
            print(f"    {'Win Rate':<20s} {no_m['win_rate']:>10.1f}% {with_m['win_rate']:>12.1f}% {delta_wr:>+9.1f}%")
            print(f"    {'Profit Factor':<20s} {no_m['profit_factor']:>10.2f} {with_m['profit_factor']:>12.2f} {delta_pf:>+9.2f}")
            print(f"    {'Max Drawdown':<20s} {no_m['max_drawdown_pct']:>10.1f}% {with_m['max_drawdown_pct']:>12.1f}% {delta_dd:>+9.1f}%")
            print(f"    {'Trades':<20s} {no_m['num_trades']:>10} {with_m['num_trades']:>12} {with_m['num_trades']-no_m['num_trades']:>+9}")

    # ── Verdict ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    base_pnl = results["abcd"]["metrics"]["total_pnl"]

    best_key = max(
        [k for k in results if k != "abcd"],
        key=lambda k: results[k]["metrics"]["total_pnl"]
    )
    best_m = results[best_key]["metrics"]
    best_label = [l for k, l, _, _, _ in configs if k == best_key][0]

    print(f"\n  Baseline (A-D):     ${base_pnl:>+10,.2f}")
    print(f"  Best config:        {best_label}")
    print(f"  Best PnL:           ${best_m['total_pnl']:>+10,.2f} "
          f"(+${best_m['total_pnl'] - base_pnl:,.2f} over baseline)")
    print(f"  Best WR:            {best_m['win_rate']:.1f}%")
    print(f"  Best PF:            {best_m['profit_factor']:.2f}")
    print(f"  Best MaxDD:         {best_m['max_drawdown_pct']:.1f}%")

    # Regime contribution
    no_regime_pnl = results.get("abcde_no_regime", {}).get("metrics", {}).get("total_pnl", 0)
    regime_pnl = results.get("abcde_regime", {}).get("metrics", {}).get("total_pnl", 0)
    regime_delta = regime_pnl - no_regime_pnl

    if regime_delta > 0:
        print(f"\n  ✅ Regime detection IMPROVES hybrid by ${regime_delta:+,.2f}")
    elif regime_delta > -100:
        print(f"\n  🟡 Regime detection has negligible impact (${regime_delta:+,.2f})")
    else:
        print(f"\n  ❌ Regime detection HURTS hybrid by ${regime_delta:,.2f}")

    # Save
    save_data = {}
    for key in results:
        m = results[key]["metrics"]
        save_data[key] = {k: v for k, v in m.items()
                          if not isinstance(v, (pd.DataFrame, pd.Series, list))}
    save_data["regime_impact"] = round(regime_delta, 2)
    save_data["best_config"] = best_key

    with open(BACKTEST_DIR / "phase9_regime_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results saved: phase9_regime_results.json")
    print(f"\n{'=' * 70}")
    print("✅ Phase 9 complete! Next: Phase 8 (TQC lot sizing)")
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