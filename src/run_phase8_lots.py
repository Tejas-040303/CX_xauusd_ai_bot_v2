"""
Project MIDAS v2 — Phase 8: Dynamic Lot Sizing Backtest
=========================================================
Compares fixed lot sizing (0.01) vs dynamic lot sizing (0.01-0.05)
using the DynamicLotSizer with regime awareness.

Subclasses Backtester to support per-trade lot sizing without
modifying the original backtester.py.

Usage:
    python run_phase8_lots.py
    python run_phase8_lots.py --mode both
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_RAW, DATA_PROCESSED, MODELS_DIR, BACKTEST_DIR,
    ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE,
    SPREAD_SIMULATION_PIPS, LOT_SIZE_SINGLE,
    MAX_TRADE_RISK, MAX_DAILY_LOSS, COOLDOWN_SECONDS,
    SESSION_LIMITS, CONFIDENCE_THRESHOLD,
    BACKTEST_INITIAL_CAPITAL, BACKTEST_FORWARD_START, BACKTEST_TRAIN_END,
)
from backtester import Backtester, StrategyBase, TradeSignal, Trade, DailyState
from strategies.strategy_a import StrategyA
from strategies.strategy_b import StrategyB
from strategies.strategy_c import StrategyC
from strategies.strategy_d import StrategyD
from run_combined_abcd import CombinedABCD, load_and_prepare_data
from regime_detector import RegimeDetector, REGIME_ADJUSTMENTS, REGIMES
from dynamic_lot_sizer import DynamicLotSizer


# ─── E Config ────────────────────────────────────────────
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

WF_MODEL_MAP = [
    {"model": "strategy_e_clean_W2.txt",
     "start": "2024-07-01", "end": "2024-12-31"},
    {"model": "strategy_e_clean_W3.txt",
     "start": "2025-01-01", "end": "2025-06-30"},
    {"model": "strategy_e_clean_final.txt",
     "start": "2025-07-01", "end": "2026-12-31"},
]


def precompute_e_predictions() -> dict:
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


# ═════════════════════════════════════════════════════════
# DYNAMIC LOT BACKTESTER (extends Backtester)
# ═════════════════════════════════════════════════════════

class DynamicLotBacktester(Backtester):
    """
    Backtester subclass that reads lot_size from
    signal.sub_conditions["dynamic_lot"] if present.
    All other logic is identical to the parent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lot_stats = {"lots_used": [], "lot_distribution": defaultdict(int)}

    def run(self, df, strategy, mode="full", lot_override=None):
        """Override run to support dynamic lots from sub_conditions."""
        df = self._filter_by_mode(df, mode)
        if len(df) == 0:
            return self._empty_results()

        capital = self.initial_capital
        equity_curve = [capital]
        equity_times = [df["datetime"].iloc[0]]
        trades = []
        daily_states = {}
        active_trade = None
        last_trade_time = None

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Backtesting: {strategy.name}")
            print(f"Mode: {mode} | Lot: {'dynamic' if lot_override is None else f'fixed {lot_override}'}")
            print(f"Period: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
            print(f"{'='*60}\n")

        for idx in range(len(df)):
            row = df.iloc[idx]
            current_time = row["datetime"]
            current_date = current_time.strftime("%Y-%m-%d")

            if current_date not in daily_states:
                daily_states[current_date] = DailyState(date=current_date)
            daily = daily_states[current_date]

            # Check active trade
            if active_trade is not None:
                exit_result = self._check_exit(active_trade, row)
                if exit_result is not None:
                    exit_price, exit_reason = exit_result
                    pnl = self._calculate_pnl(
                        active_trade["direction"], active_trade["entry_price"],
                        exit_price, active_trade["lot_size"])
                    pnl_pips = self._calculate_pips(
                        active_trade["direction"], active_trade["entry_price"], exit_price)
                    risk_pips = abs(active_trade["entry_price"] - active_trade["sl_price"]) / PIP_VALUE
                    rr = pnl_pips / risk_pips if risk_pips > 0 else 0

                    trade = Trade(
                        entry_time=active_trade["entry_time"], exit_time=current_time,
                        direction=active_trade["direction"],
                        strategy_name=active_trade["strategy_name"],
                        confidence=active_trade["confidence"],
                        entry_price=active_trade["entry_price"], exit_price=exit_price,
                        sl_price=active_trade["sl_price"], tp_price=active_trade["tp_price"],
                        lot_size=active_trade["lot_size"],
                        pnl_dollars=pnl, pnl_pips=pnl_pips,
                        exit_reason=exit_reason, session=active_trade["session"],
                        sub_conditions=active_trade.get("sub_conditions", {}),
                        rr_achieved=rr,
                    )
                    trades.append(trade)
                    capital += pnl
                    daily.pnl += pnl
                    last_trade_time = current_time
                    active_trade = None
                    equity_curve.append(capital)
                    equity_times.append(current_time)

                    if daily.pnl <= -MAX_DAILY_LOSS:
                        daily.is_locked = True

                    # Record result for lot sizer streak tracking
                    if hasattr(strategy, 'lot_sizer') and strategy.lot_sizer:
                        strategy.lot_sizer.record_result(pnl)

                continue

            if daily.is_locked:
                continue

            session = row.get("session", "off")
            session_key = session if session in SESSION_LIMITS else "off"
            if daily.session_trades.get(session_key, 0) >= SESSION_LIMITS.get(session_key, 0):
                continue

            if last_trade_time is not None:
                elapsed = (current_time - last_trade_time).total_seconds()
                if elapsed < COOLDOWN_SECONDS:
                    continue

            signal = strategy.generate_signal(df, idx)
            if signal is None:
                continue
            if signal.confidence < CONFIDENCE_THRESHOLD:
                continue

            # ── DYNAMIC LOT: read from sub_conditions ─────
            if lot_override:
                lot = lot_override
            elif "dynamic_lot" in signal.sub_conditions:
                lot = signal.sub_conditions["dynamic_lot"]
            else:
                lot = LOT_SIZE_SINGLE

            entry_price = self._apply_spread(signal.entry_price, signal.direction)

            # Risk cap
            sl_dist_dollars = abs(entry_price - signal.sl_price) * lot * XAUUSD_POINT_VALUE
            if sl_dist_dollars > MAX_TRADE_RISK:
                max_lot = MAX_TRADE_RISK / (abs(entry_price - signal.sl_price) * XAUUSD_POINT_VALUE)
                lot = max(0.01, round(max_lot, 2))
                sl_dist_dollars = abs(entry_price - signal.sl_price) * lot * XAUUSD_POINT_VALUE
                if sl_dist_dollars > MAX_TRADE_RISK:
                    continue

            remaining = MAX_DAILY_LOSS + daily.pnl
            if sl_dist_dollars > remaining:
                continue

            # Track lot distribution
            self.lot_stats["lots_used"].append(lot)
            self.lot_stats["lot_distribution"][lot] += 1

            active_trade = {
                "entry_time": current_time, "entry_price": entry_price,
                "direction": signal.direction,
                "sl_price": signal.sl_price, "tp_price": signal.tp_price,
                "lot_size": lot,
                "strategy_name": signal.strategy_name,
                "confidence": signal.confidence,
                "session": session,
                "sub_conditions": signal.sub_conditions,
            }
            daily.session_trades[session_key] = daily.session_trades.get(session_key, 0) + 1
            daily.trades_taken += 1

        # Close remaining
        if active_trade is not None:
            last_row = df.iloc[-1]
            pnl = self._calculate_pnl(
                active_trade["direction"], active_trade["entry_price"],
                last_row["close"], active_trade["lot_size"])
            pnl_pips = self._calculate_pips(
                active_trade["direction"], active_trade["entry_price"], last_row["close"])
            risk_pips = abs(active_trade["entry_price"] - active_trade["sl_price"]) / PIP_VALUE
            trade = Trade(
                entry_time=active_trade["entry_time"], exit_time=last_row["datetime"],
                direction=active_trade["direction"],
                strategy_name=active_trade["strategy_name"],
                confidence=active_trade["confidence"],
                entry_price=active_trade["entry_price"], exit_price=last_row["close"],
                sl_price=active_trade["sl_price"], tp_price=active_trade["tp_price"],
                lot_size=active_trade["lot_size"],
                pnl_dollars=pnl, pnl_pips=pnl_pips,
                exit_reason="end_of_data", session=active_trade["session"],
                sub_conditions=active_trade.get("sub_conditions", {}),
                rr_achieved=pnl_pips / risk_pips if risk_pips > 0 else 0,
            )
            trades.append(trade)
            capital += pnl
            equity_curve.append(capital)
            equity_times.append(last_row["datetime"])

        metrics = self._compute_metrics(trades, equity_curve)
        results = {
            "strategy": strategy.name, "mode": mode,
            "start_date": str(df["datetime"].iloc[0]),
            "end_date": str(df["datetime"].iloc[-1]),
            "initial_capital": self.initial_capital,
            "final_capital": capital,
            "trades": trades, "equity_curve": equity_curve,
            "equity_times": equity_times, "daily_states": daily_states,
            "metrics": metrics,
        }

        if self.verbose:
            self.print_report(results)

        return results

    def print_lot_stats(self):
        """Print lot distribution statistics."""
        if not self.lot_stats["lots_used"]:
            return
        lots = np.array(self.lot_stats["lots_used"])
        print(f"\n  LOT SIZE DISTRIBUTION:")
        print(f"    Mean: {lots.mean():.3f} | Median: {np.median(lots):.3f}")
        print(f"    Min: {lots.min():.2f} | Max: {lots.max():.2f}")
        for lot_val, count in sorted(self.lot_stats["lot_distribution"].items()):
            pct = count / len(lots) * 100
            bar = "█" * int(pct / 2)
            print(f"    {lot_val:.2f}: {count:>5} ({pct:>5.1f}%) {bar}")


# ═════════════════════════════════════════════════════════
# STRATEGY WITH DYNAMIC LOT SIZING
# ═════════════════════════════════════════════════════════

class CombinedABCDE_DynamicLot(StrategyBase):
    """
    Full hybrid strategy with regime detection AND dynamic lot sizing.
    Embeds computed lot size in signal.sub_conditions["dynamic_lot"].
    """

    name = "Combined_A+B+C+D+E_DynLot"

    def __init__(self, e_predictions: dict, use_dynamic_lots: bool = True):
        self.strategy_a = StrategyA()
        self.strategy_b = StrategyB()
        self.strategy_c = StrategyC()
        self.strategy_d = StrategyD()

        self.e_predictions = e_predictions
        self.regime_detector = RegimeDetector()
        self.lot_sizer = DynamicLotSizer() if use_dynamic_lots else None
        self.use_dynamic_lots = use_dynamic_lots

        self._daily_pnl = 0.0
        self._current_date = None

        self.diag = {
            "candles": 0, "ad_signal": 0, "ad_conflict": 0, "ad_no_signal": 0,
            "e_strong_agree": 0, "e_agree": 0, "e_weak": 0,
            "e_conflict_skip": 0, "e_indep_fired": 0, "e_indep_below": 0,
            "e_indep_regime_blocked": 0,
            "sources": {},
        }

    def generate_signal(self, df, idx):
        self.diag["candles"] += 1
        row = df.iloc[idx]

        # Daily tracking
        bar_date = row["datetime"].strftime("%Y-%m-%d") if hasattr(row["datetime"], 'strftime') else str(row["datetime"])[:10]
        if self._current_date != bar_date:
            self._current_date = bar_date
            self._daily_pnl = 0.0
            if self.lot_sizer:
                self.lot_sizer.reset_daily()

        regime, adj = self.regime_detector.detect_and_adjust(row)

        # A-D signals
        signals = []
        for name, strat in [("A", self.strategy_a), ("B", self.strategy_b),
                             ("C", self.strategy_c), ("D", self.strategy_d)]:
            sig = strat.generate_signal(df, idx)
            if sig:
                signals.append((name, sig))

        bar_time = row["datetime"]
        e_prob = self.e_predictions.get(str(pd.Timestamp(bar_time)))

        atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
        if pd.isna(atr) or atr <= 0:
            atr = 2.0

        if signals:
            return self._handle_ad(signals, e_prob, adj, regime, atr, df, idx)

        self.diag["ad_no_signal"] += 1
        return self._handle_e_indep(e_prob, adj, regime, atr, df, idx)

    def _handle_ad(self, signals, e_prob, adj, regime, atr, df, idx):
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

        # E validation
        if e_prob is not None:
            ad_direction = best_signal.direction
            val_strong = adj.get("e_val_strong_agree", E_VAL_STRONG_AGREE)
            val_agree = adj.get("e_val_agree", E_VAL_AGREE)
            val_conflict = adj.get("e_val_conflict_skip", E_VAL_CONFLICT_SKIP)

            e_agrees = (e_prob > 0.5) if ad_direction == "buy" else (e_prob < 0.5)
            e_strength = e_prob if ad_direction == "buy" else (1 - e_prob)

            if e_agrees:
                if e_strength >= val_strong:
                    best_signal.confidence = min(100, best_signal.confidence + E_STRONG_BOOST + adj.get("e_conf_boost", 0))
                    self.diag["e_strong_agree"] += 1
                    sources += "+E_strong"
                elif e_strength >= val_agree:
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

        # SL/TP regime adjustment
        sl_mult = adj.get("sl_mult", 1.0)
        tp_mult = adj.get("tp_mult", 1.0)
        entry = best_signal.entry_price
        if sl_mult != 1.0:
            sl_dist = abs(entry - best_signal.sl_price)
            if best_signal.direction == "buy":
                best_signal.sl_price = round(entry - sl_dist * sl_mult, 2)
            else:
                best_signal.sl_price = round(entry + sl_dist * sl_mult, 2)
        if tp_mult != 1.0:
            tp_dist = abs(entry - best_signal.tp_price)
            if best_signal.direction == "buy":
                best_signal.tp_price = round(entry + tp_dist * tp_mult, 2)
            else:
                best_signal.tp_price = round(entry - tp_dist * tp_mult, 2)

        # ── DYNAMIC LOT SIZING ────────────────────────────
        if self.use_dynamic_lots and self.lot_sizer:
            sl_distance = abs(entry - best_signal.sl_price)
            lot_result = self.lot_sizer.compute(
                n_strategies=n_agree, e_prob=e_prob, regime=regime,
                daily_pnl=self._daily_pnl, atr=atr, sl_distance=sl_distance,
            )
            best_signal.sub_conditions["dynamic_lot"] = lot_result["lot_size"]
            best_signal.sub_conditions["lot_factors"] = lot_result["factors"]

        best_signal.sub_conditions["source"] = sources
        best_signal.sub_conditions["regime"] = regime
        best_signal.sub_conditions["strategies_agreed"] = n_agree
        self.diag["sources"][sources] = self.diag["sources"].get(sources, 0) + 1
        return best_signal

    def _handle_e_indep(self, e_prob, adj, regime, atr, df, idx):
        if e_prob is None:
            return None

        if not adj.get("allow_e_independent", True):
            self.diag["e_indep_regime_blocked"] += 1
            return None

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

        row = df.iloc[idx]
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

        sub_conditions = {
            "source": "E_independent",
            "e_prob": round(e_prob, 4),
            "regime": regime,
            "strategies_agreed": 0,
        }

        # Dynamic lot
        if self.use_dynamic_lots and self.lot_sizer:
            lot_result = self.lot_sizer.compute(
                n_strategies=0, e_prob=e_prob, regime=regime,
                daily_pnl=self._daily_pnl, atr=atr, sl_distance=sl_dist,
            )
            sub_conditions["dynamic_lot"] = lot_result["lot_size"]

        signal = TradeSignal(
            datetime=row["datetime"], direction=direction,
            strategy_name=self.name, confidence=confidence,
            entry_price=round(entry, 2), sl_price=round(sl, 2), tp_price=round(tp, 2),
            sub_conditions=sub_conditions,
        )

        self.diag["e_indep_fired"] += 1
        self.diag["sources"]["E_independent"] = self.diag["sources"].get("E_independent", 0) + 1
        return signal

    def print_diagnostics(self):
        d = self.diag
        total = d["ad_signal"] + d["e_indep_fired"]
        print(f"\n  DIAGNOSTICS ({'dynamic lots' if self.use_dynamic_lots else 'fixed lots'}):")
        print(f"  {'─' * 50}")
        print(f"  Candles: {d['candles']:,} | A-D: {d['ad_signal']:,} | E indep: {d['e_indep_fired']:,}")
        if d["e_conflict_skip"]:
            print(f"  E conflict skips: {d['e_conflict_skip']}")
        if d["e_indep_regime_blocked"]:
            print(f"  E regime blocked: {d['e_indep_regime_blocked']}")
        self.regime_detector.print_stats()


# ═════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════

def run(mode="forward"):
    print("=" * 70)
    print("MIDAS v2 — Phase 8: Dynamic Lot Sizing")
    print(f"  Mode: {mode} | Lot range: 0.01 → 0.05")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    df = load_and_prepare_data()

    print("\n[2/3] Strategy E predictions...")
    e_preds = precompute_e_predictions()

    print("\n[3/3] Running backtests...")

    configs = [
        ("baseline_fixed",   "A+B+C+D (fixed 0.01)",        False, False),
        ("hybrid_fixed",     "A+B+C+D+E+Regime (fixed)",    True,  False),
        ("hybrid_dynamic",   "A+B+C+D+E+Regime (dynamic)",  True,  True),
    ]

    results = {}
    for key, label, use_e, use_dyn in configs:
        print(f"\n  Running {label}...")

        if not use_e:
            # Baseline A-D only
            bt = Backtester(verbose=True)
            strat = CombinedABCD()
            results[key] = bt.run(df, strat, mode=mode)
        else:
            bt = DynamicLotBacktester(verbose=True)
            strat = CombinedABCDE_DynamicLot(e_preds, use_dynamic_lots=use_dyn)
            results[key] = bt.run(df, strat, mode=mode)
            strat.print_diagnostics()
            if use_dyn:
                bt.print_lot_stats()

        m = results[key]["metrics"]
        print(f"  → PnL: ${m['total_pnl']:+,.2f} | WR: {m['win_rate']:.1f}% | "
              f"PF: {m['profit_factor']:.2f} | Trades: {m['num_trades']}")

        try:
            bt_saver = Backtester(verbose=False)
            bt_saver.save_results(results[key], f"Phase8_{key.upper()}.json")
            bt_saver.plot_equity(results[key],
                                save_path=str(BACKTEST_DIR / f"Phase8_{key.upper()}_equity.png"))
        except Exception:
            pass

    # ── Comparison ────────────────────────────────────────
    print(f"\n{'=' * 95}")
    print("PHASE 8 — LOT SIZING COMPARISON")
    print(f"{'=' * 95}")
    print(f"  {'Mode':<38s} {'PnL':>9} {'WR%':>6} {'PF':>6} "
          f"{'Trades':>7} {'T/Mo':>6} {'MaxDD':>6} {'Sharpe':>7}")
    print(f"  {'-'*85}")

    for key, label, _, _ in configs:
        m = results[key]["metrics"]
        sharpe = m.get("sharpe_ratio", 0)
        print(f"  {label:<38s} ${m['total_pnl']:>7,.0f} "
              f"{m['win_rate']:>5.1f} {m['profit_factor']:>5.2f} "
              f"{m['num_trades']:>7} {m['trades_per_month']:>5.1f} "
              f"{m['max_drawdown_pct']:>5.1f}% {sharpe:>6.2f}")

    # ── Lift analysis ─────────────────────────────────────
    base_pnl = results["baseline_fixed"]["metrics"]["total_pnl"]
    fixed_pnl = results["hybrid_fixed"]["metrics"]["total_pnl"]
    dyn_pnl = results["hybrid_dynamic"]["metrics"]["total_pnl"]

    print(f"\n{'=' * 70}")
    print("DYNAMIC LOT SIZING IMPACT")
    print(f"{'=' * 70}")
    print(f"  A-D baseline (fixed 0.01):    ${base_pnl:>+10,.2f}")
    print(f"  Hybrid (fixed 0.01):          ${fixed_pnl:>+10,.2f}  (E+Regime lift: ${fixed_pnl-base_pnl:>+,.2f})")
    print(f"  Hybrid (dynamic 0.01-0.05):   ${dyn_pnl:>+10,.2f}  (+ dynamic lot: ${dyn_pnl-fixed_pnl:>+,.2f})")
    print(f"\n  Total lift over baseline:     ${dyn_pnl-base_pnl:>+10,.2f} ({(dyn_pnl-base_pnl)/abs(base_pnl)*100:+.0f}%)")

    lot_delta = dyn_pnl - fixed_pnl
    if lot_delta > 0:
        print(f"\n  ✅ Dynamic lots ADD ${lot_delta:+,.2f} over fixed lots")
    else:
        print(f"\n  ❌ Dynamic lots COST ${lot_delta:,.2f} vs fixed lots")

    # Check drawdown
    fixed_dd = results["hybrid_fixed"]["metrics"]["max_drawdown_pct"]
    dyn_dd = results["hybrid_dynamic"]["metrics"]["max_drawdown_pct"]
    print(f"  MaxDD: fixed={fixed_dd:.1f}% → dynamic={dyn_dd:.1f}%")

    if dyn_dd <= fixed_dd * 1.5:
        print(f"  ✅ Drawdown acceptable (within 1.5x of fixed)")
    else:
        print(f"  ⚠️ Drawdown increased significantly — tune lot multipliers")

    # Save
    save_data = {}
    for key in results:
        m = results[key]["metrics"]
        save_data[key] = {k: v for k, v in m.items()
                          if not isinstance(v, (pd.DataFrame, pd.Series, list))}
    save_data["dynamic_lot_impact"] = round(lot_delta, 2)

    with open(BACKTEST_DIR / "phase8_lot_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Results saved: phase8_lot_results.json")
    print(f"\n{'=' * 70}")
    print("✅ Phase 8 complete!")
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