"""
Project MIDAS v2 — Decision Engine
====================================
Hybrid decision-making system that combines:
  - Rule-based strategies (A, B, C, D) generating independent signals
  - AI Strategy E acting as both validator AND independent signal generator

Signal Flow:
  1. Every M5 bar: compute features → ask all strategies for signals
  2. If A/B/C/D fires → Strategy E validates → adjust lot → execute/skip
  3. If no A-D signal → check if Strategy E fires independently → execute if confident
  4. Risk engine has FINAL say (SL required, daily limits, session limits)

Usage:
    from decision_engine import DecisionEngine

    engine = DecisionEngine()

    # On each M5 bar:
    decision = engine.evaluate(features_dict, active_strategy_signals)
    # → {"action": "BUY", "lot_size": 0.02, "sl": ..., "tp": ..., "reason": ...}
"""

import numpy as np
from datetime import datetime

from config import (
    LOT_SIZE_MIN, LOT_SIZE_MAX, LOT_SIZE_SINGLE, LOT_SIZE_MULTI,
    MAX_DAILY_LOSS, MAX_TRADE_RISK, COOLDOWN_SECONDS,
    SESSION_LIMITS, CONFIDENCE_THRESHOLD,
    XAUUSD_POINT_VALUE, PIP_VALUE, ATR_PERIOD,
)
from strategies.strategy_e import StrategyE


# ─── Decision Engine Configuration ───────────────────────

# Minimum confidence from any source to consider a trade
MIN_CONFIDENCE = 60

# Strategy E independent signal requires higher confidence
E_INDEPENDENT_MIN_CONFIDENCE = 70

# When multiple strategies agree, lot size scales up
AGREEMENT_LOT_MAP = {
    1: LOT_SIZE_SINGLE,     # 0.01 — single strategy
    2: LOT_SIZE_MULTI,      # 0.02 — two strategies agree
    3: LOT_SIZE_MAX,        # 0.02 — three+ strategies agree (capped)
}

# Maximum lot after AI boost (safety cap)
MAX_LOT_AFTER_BOOST = LOT_SIZE_MAX  # Never exceed 0.02


class DecisionEngine:
    """
    Central decision-making hub.
    Receives signals from all strategies, resolves conflicts,
    applies AI validation, and determines final trade action.
    """

    def __init__(self, strategy_e: StrategyE = None, enable_ai: bool = True):
        """
        Args:
            strategy_e: Pre-initialized StrategyE instance.
                        If None and enable_ai=True, will attempt to load.
            enable_ai: If False, runs purely on A-D signals (fallback mode).
        """
        self.enable_ai = enable_ai

        if enable_ai:
            if strategy_e:
                self.strategy_e = strategy_e
            else:
                try:
                    self.strategy_e = StrategyE()
                except FileNotFoundError:
                    print("  WARNING: Strategy E model not found. "
                          "Running in rule-based only mode.")
                    self.strategy_e = None
                    self.enable_ai = False
        else:
            self.strategy_e = None

        # Daily tracking state
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_trade_time = None
        self._session_trade_counts = {
            "asian": 0, "london": 0, "new_york": 0,
            "london_ny_overlap": 0, "off": 0
        }
        self._current_date = None

        # Decision log (last N decisions for debugging)
        self._decision_log = []

    def evaluate(
        self,
        features: dict,
        strategy_signals: list,
        current_time: datetime = None,
        current_session: str = None,
    ) -> dict:
        """
        Main decision function — called on every M5 bar.

        Args:
            features: Feature dictionary for current M5 bar (all 167+ features).
            strategy_signals: List of signal dicts from strategies A-D.
                Each signal: {"strategy": "A", "direction": "BUY",
                              "confidence": 73.5, "sl_pips": 20, "tp_pips": 30}
                Empty list if no strategy fired.
            current_time: Current bar timestamp (for session/cooldown checks).
            current_session: Current session name ("asian", "london", etc.).

        Returns:
            dict with final decision:
              - action: "BUY", "SELL", or "SKIP"
              - lot_size: Position size (0 if SKIP)
              - sl_pips: Stop loss in pips
              - tp_pips: Take profit in pips
              - source: What triggered this trade
              - ai_validation: Strategy E's assessment (if enabled)
              - reason: Human-readable explanation
              - signals_received: All signals that were considered
        """
        # ── Reset daily counters if new day ───────────────
        if current_time:
            today = current_time.date()
            if self._current_date != today:
                self._reset_daily_state(today)

        # ── Pre-flight risk checks ────────────────────────
        risk_block = self._check_risk_limits(current_time, current_session)
        if risk_block:
            return self._make_skip_decision(
                reason=risk_block,
                signals=strategy_signals,
            )

        # ── Separate active signals by direction ──────────
        active_signals = [s for s in strategy_signals
                          if s.get("confidence", 0) >= MIN_CONFIDENCE
                          and s.get("direction") in ("BUY", "SELL")]

        buy_signals = [s for s in active_signals if s["direction"] == "BUY"]
        sell_signals = [s for s in active_signals if s["direction"] == "SELL"]

        # ── CASE 1: Strategy signals exist ────────────────
        if active_signals:
            return self._handle_strategy_signals(
                features, buy_signals, sell_signals,
                active_signals, current_session
            )

        # ── CASE 2: No strategy signal — check Strategy E ─
        if self.enable_ai and self.strategy_e:
            return self._handle_ai_independent(features, current_session)

        # ── CASE 3: Nothing to do ─────────────────────────
        return self._make_skip_decision(
            reason="No signals from any source",
            signals=strategy_signals,
        )

    def _handle_strategy_signals(
        self, features, buy_signals, sell_signals,
        all_signals, session
    ) -> dict:
        """
        Process signals from A-D strategies with AI validation.
        """
        # ── Resolve direction conflicts ───────────────────
        if buy_signals and sell_signals:
            # Conflict between strategies — use highest total confidence
            buy_conf_sum = sum(s["confidence"] for s in buy_signals)
            sell_conf_sum = sum(s["confidence"] for s in sell_signals)

            if buy_conf_sum > sell_conf_sum:
                direction = "BUY"
                primary_signals = buy_signals
                conflict_note = (f"BUY wins conflict (BUY conf sum={buy_conf_sum:.0f} "
                                 f"vs SELL conf sum={sell_conf_sum:.0f})")
            elif sell_conf_sum > buy_conf_sum:
                direction = "SELL"
                primary_signals = sell_signals
                conflict_note = (f"SELL wins conflict (SELL conf sum={sell_conf_sum:.0f} "
                                 f"vs BUY conf sum={buy_conf_sum:.0f})")
            else:
                return self._make_skip_decision(
                    reason="Equal BUY/SELL confidence — conflict unresolvable",
                    signals=all_signals,
                )
        elif buy_signals:
            direction = "BUY"
            primary_signals = buy_signals
            conflict_note = None
        else:
            direction = "SELL"
            primary_signals = sell_signals
            conflict_note = None

        # ── Determine base lot from agreement count ───────
        n_agreeing = len(primary_signals)
        base_lot = AGREEMENT_LOT_MAP.get(min(n_agreeing, 3), LOT_SIZE_SINGLE)

        # Best signal for SL/TP
        best_signal = max(primary_signals, key=lambda s: s["confidence"])
        sl_pips = best_signal.get("sl_pips", 20)
        tp_pips = best_signal.get("tp_pips", 30)

        # ── AI Validation ─────────────────────────────────
        ai_validation = None
        lot_multiplier = 1.0

        if self.enable_ai and self.strategy_e:
            ai_validation = self.strategy_e.validate_signal(
                features,
                other_direction=direction,
                other_confidence=best_signal["confidence"],
                other_strategy=best_signal["strategy"],
            )

            lot_multiplier = ai_validation["lot_multiplier"]

            # AI says SKIP → override
            if ai_validation["action"] == "SKIP":
                return self._make_skip_decision(
                    reason=ai_validation["reason"],
                    signals=all_signals,
                    ai_validation=ai_validation,
                )

        # ── Final lot size ────────────────────────────────
        final_lot = min(base_lot * lot_multiplier, MAX_LOT_AFTER_BOOST)
        final_lot = max(final_lot, LOT_SIZE_MIN)
        # Round to 0.01
        final_lot = round(final_lot * 100) / 100

        # ── Risk-check the final lot ──────────────────────
        max_lot_for_risk = self._max_lot_for_sl(sl_pips)
        final_lot = min(final_lot, max_lot_for_risk)

        # ── Build source description ──────────────────────
        source_strategies = "+".join(s["strategy"] for s in primary_signals)
        if ai_validation:
            source = f"{source_strategies}+E_validated"
        else:
            source = source_strategies

        # ── Build reason ──────────────────────────────────
        reasons = []
        for s in primary_signals:
            reasons.append(f"{s['strategy']}={direction}@{s['confidence']:.0f}%")
        if ai_validation:
            reasons.append(f"AI={ai_validation['ai_direction']}@"
                           f"{ai_validation['ai_confidence']:.0f}% "
                           f"(×{lot_multiplier:.2f})")
        if conflict_note:
            reasons.append(conflict_note)

        decision = {
            "action": direction,
            "lot_size": final_lot,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "source": source,
            "ai_validation": ai_validation,
            "reason": " | ".join(reasons),
            "signals_received": all_signals,
            "n_agreeing_strategies": n_agreeing,
        }

        self._log_decision(decision)
        return decision

    def _handle_ai_independent(self, features, session) -> dict:
        """
        Strategy E fires independently — no A-D signal present.
        Requires higher confidence threshold.
        """
        signal = self.strategy_e.generate_signal(features)

        if signal["direction"] == "WAIT":
            return self._make_skip_decision(
                reason=f"AI independent: no signal (prob={signal['raw_probability']:.4f})",
                signals=[],
                ai_validation={"raw_probability": signal["raw_probability"]},
            )

        if signal["confidence"] < E_INDEPENDENT_MIN_CONFIDENCE:
            return self._make_skip_decision(
                reason=(f"AI independent: {signal['direction']} but confidence "
                        f"{signal['confidence']:.0f}% < {E_INDEPENDENT_MIN_CONFIDENCE}% threshold"),
                signals=[],
                ai_validation=signal,
            )

        # AI fires independently — use smaller lot (more conservative)
        lot_size = LOT_SIZE_SINGLE  # Always 0.01 for AI-only trades

        # Risk-check
        max_lot = self._max_lot_for_sl(signal["sl_pips"])
        lot_size = min(lot_size, max_lot)

        decision = {
            "action": signal["direction"],
            "lot_size": lot_size,
            "sl_pips": signal["sl_pips"],
            "tp_pips": signal["tp_pips"],
            "source": "E_independent",
            "ai_validation": signal,
            "reason": (f"AI independent signal: {signal['direction']}@"
                       f"{signal['confidence']:.0f}% "
                       f"(prob={signal['raw_probability']:.4f})"),
            "signals_received": [],
            "n_agreeing_strategies": 0,
        }

        self._log_decision(decision)
        return decision

    # ─── Risk Management ──────────────────────────────────

    def _check_risk_limits(self, current_time, session) -> str:
        """
        Pre-flight risk checks. Returns a reason string if trade is blocked,
        or None if clear to proceed.
        """
        # Daily loss limit
        if self._daily_pnl <= -MAX_DAILY_LOSS:
            return f"Daily loss limit hit (${self._daily_pnl:.2f} <= -${MAX_DAILY_LOSS})"

        # Cooldown
        if current_time and self._last_trade_time:
            elapsed = (current_time - self._last_trade_time).total_seconds()
            if elapsed < COOLDOWN_SECONDS:
                remaining = COOLDOWN_SECONDS - elapsed
                return f"Cooldown active ({remaining:.0f}s remaining)"

        # Session limits
        if session and session in SESSION_LIMITS:
            limit = SESSION_LIMITS[session]
            current_count = self._session_trade_counts.get(session, 0)
            if current_count >= limit:
                return f"Session limit hit ({session}: {current_count}/{limit} trades)"

        # Off-session block
        if session == "off":
            return "No trading outside defined sessions"

        return None

    def _max_lot_for_sl(self, sl_pips: float) -> float:
        """
        Calculate maximum lot size that keeps risk within MAX_TRADE_RISK.
        risk = lot_size × XAUUSD_POINT_VALUE × sl_in_dollars
        """
        if sl_pips <= 0:
            return LOT_SIZE_MIN

        sl_dollars = sl_pips * PIP_VALUE  # Convert pips to dollar price movement
        # risk = lot_size × XAUUSD_POINT_VALUE × sl_dollars
        # lot_size = MAX_TRADE_RISK / (XAUUSD_POINT_VALUE × sl_dollars)
        max_lot = MAX_TRADE_RISK / (XAUUSD_POINT_VALUE * sl_dollars)
        max_lot = max(LOT_SIZE_MIN, min(max_lot, LOT_SIZE_MAX))
        return round(max_lot * 100) / 100

    def _reset_daily_state(self, today):
        """Reset daily counters for a new trading day."""
        self._current_date = today
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._session_trade_counts = {
            k: 0 for k in self._session_trade_counts
        }

    # ─── Trade Recording ──────────────────────────────────

    def record_trade_result(self, pnl: float, session: str = None,
                            trade_time: datetime = None):
        """
        Called after a trade closes to update risk tracking.

        Args:
            pnl: Profit/loss of the closed trade in dollars.
            session: Session the trade was in.
            trade_time: When the trade was opened.
        """
        self._daily_pnl += pnl
        self._daily_trades += 1

        if trade_time:
            self._last_trade_time = trade_time

        if session and session in self._session_trade_counts:
            self._session_trade_counts[session] += 1

    # ─── Utility ──────────────────────────────────────────

    def _make_skip_decision(self, reason, signals, ai_validation=None) -> dict:
        """Helper to create a SKIP decision."""
        return {
            "action": "SKIP",
            "lot_size": 0,
            "sl_pips": 0,
            "tp_pips": 0,
            "source": "none",
            "ai_validation": ai_validation,
            "reason": reason,
            "signals_received": signals,
            "n_agreeing_strategies": 0,
        }

    def _log_decision(self, decision):
        """Log a decision for debugging."""
        self._decision_log.append(decision)
        if len(self._decision_log) > 500:
            self._decision_log = self._decision_log[-250:]

    def get_status(self) -> dict:
        """Get current engine status for monitoring."""
        status = {
            "ai_enabled": self.enable_ai,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_trades": self._daily_trades,
            "session_counts": dict(self._session_trade_counts),
            "last_trade_time": str(self._last_trade_time) if self._last_trade_time else None,
            "decisions_logged": len(self._decision_log),
        }

        if self.enable_ai and self.strategy_e:
            status["strategy_e_diagnostics"] = self.strategy_e.get_diagnostics()

        return status

    def get_recent_decisions(self, n: int = 10) -> list:
        """Get the last N decisions for debugging."""
        return self._decision_log[-n:]


# ─── Standalone Demo ──────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("MIDAS v2 — Decision Engine Demo")
    print("=" * 70)

    # Initialize (will try to load Strategy E)
    try:
        engine = DecisionEngine(enable_ai=True)
    except Exception as e:
        print(f"  AI init failed: {e}")
        print("  Running in rule-based only mode...")
        engine = DecisionEngine(enable_ai=False)

    # ── Scenario 1: Strategy A fires BUY ──────────────────
    print("\n─── Scenario 1: Strategy A fires BUY @ 73% ───")
    dummy_features = {}
    signals_1 = [
        {"strategy": "A", "direction": "BUY", "confidence": 73, "sl_pips": 20, "tp_pips": 30}
    ]
    result_1 = engine.evaluate(dummy_features, signals_1, current_session="london")
    print(f"  Action: {result_1['action']} | Lot: {result_1['lot_size']} | "
          f"Source: {result_1['source']}")
    print(f"  Reason: {result_1['reason']}")

    # ── Scenario 2: A+B agree on BUY ─────────────────────
    print("\n─── Scenario 2: A+B both say BUY ───")
    signals_2 = [
        {"strategy": "A", "direction": "BUY", "confidence": 73, "sl_pips": 20, "tp_pips": 30},
        {"strategy": "B", "direction": "BUY", "confidence": 68, "sl_pips": 15, "tp_pips": 25},
    ]
    result_2 = engine.evaluate(dummy_features, signals_2, current_session="london")
    print(f"  Action: {result_2['action']} | Lot: {result_2['lot_size']} | "
          f"Source: {result_2['source']}")
    print(f"  Reason: {result_2['reason']}")

    # ── Scenario 3: A says BUY, C says SELL ───────────────
    print("\n─── Scenario 3: A=BUY vs C=SELL (conflict) ───")
    signals_3 = [
        {"strategy": "A", "direction": "BUY", "confidence": 73, "sl_pips": 20, "tp_pips": 30},
        {"strategy": "C", "direction": "SELL", "confidence": 65, "sl_pips": 18, "tp_pips": 27},
    ]
    result_3 = engine.evaluate(dummy_features, signals_3, current_session="london")
    print(f"  Action: {result_3['action']} | Lot: {result_3['lot_size']} | "
          f"Source: {result_3['source']}")
    print(f"  Reason: {result_3['reason']}")

    # ── Scenario 4: No strategy fires (AI independent) ────
    print("\n─── Scenario 4: No A-D signal, AI checks independently ───")
    result_4 = engine.evaluate(dummy_features, [], current_session="london")
    print(f"  Action: {result_4['action']} | Lot: {result_4['lot_size']} | "
          f"Source: {result_4['source']}")
    print(f"  Reason: {result_4['reason']}")

    # ── Engine status ─────────────────────────────────────
    print("\n─── Engine Status ───")
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n✅ Decision Engine demo complete!")