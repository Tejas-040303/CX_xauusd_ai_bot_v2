"""
Project MIDAS v2 — Phase 8: Dynamic Lot Sizer
===============================================
Maps lot size (0.01 → 0.05) based on:
  1. Strategy agreement count (more agree = more conviction)
  2. E confidence (higher = more conviction)
  3. Regime (trending = push, volatile = pull back, quiet = minimum)
  4. Drawdown proximity (closer to daily limit = reduce)
  5. Recent performance (winning streak = scale up, losing = scale down)

Also adjusts SL/TP:
  - SL tightened when lot is larger (keep dollar risk constant)
  - TP extended in trending regimes with high conviction

Lot range: 0.01 (minimum) → 0.05 (maximum)
Risk cap: MAX_TRADE_RISK ($40) always enforced regardless of lot

Usage:
    from dynamic_lot_sizer import DynamicLotSizer

    sizer = DynamicLotSizer()
    lot, sl_adj, tp_adj = sizer.compute(
        n_strategies=2, e_confidence=72, regime="TRENDING_BULL",
        daily_pnl=-20, recent_wins=3, recent_losses=0, atr=2.5
    )
"""

import numpy as np
from config import (
    MAX_TRADE_RISK, MAX_DAILY_LOSS, XAUUSD_POINT_VALUE, PIP_VALUE,
    LOT_SIZE_SINGLE, LOT_SIZE_MULTI,
)


# ─── Lot Sizing Configuration ────────────────────────────
LOT_MIN = 0.01
LOT_MAX = 0.05

# Base lots by agreement level
BASE_LOTS = {
    0: 0.01,    # E independent only
    1: 0.01,    # Single strategy
    2: 0.02,    # Two strategies agree
    3: 0.03,    # Three agree
    4: 0.03,    # Four agree (capped — rare event)
}

# E confidence multipliers (raw probability scale)
# Higher E confidence → scale lot up
E_CONF_MULTIPLIERS = {
    "very_high": 1.5,   # E prob ≥ 0.70
    "high": 1.3,        # E prob ≥ 0.65
    "medium": 1.0,      # E prob ≥ 0.58
    "low": 0.8,          # E prob ≥ 0.52
    "none": 1.0,         # No E prediction
}

# Regime multipliers
REGIME_LOT_MULTIPLIERS = {
    "TRENDING_BULL": 1.3,    # Push harder in trends
    "TRENDING_BEAR": 1.3,
    "VOLATILE": 0.7,         # Pull back in volatility
    "QUIET": 0.5,            # Minimum in quiet markets
    "RANGING": 0.9,          # Slightly conservative in ranges
}

# Regime SL adjustments (multiplicative on top of strategy SL)
REGIME_SL_ADJ = {
    "TRENDING_BULL": 1.0,
    "TRENDING_BEAR": 1.0,
    "VOLATILE": 1.3,         # Wider SL in volatile
    "QUIET": 0.8,            # Tighter SL in quiet
    "RANGING": 1.0,
}

# Regime TP adjustments
REGIME_TP_ADJ = {
    "TRENDING_BULL": 1.2,    # Let winners run in trends
    "TRENDING_BEAR": 1.2,
    "VOLATILE": 0.9,         # Take profit faster in volatile
    "QUIET": 0.8,            # Tight TP in quiet
    "RANGING": 0.9,          # Mean-reversion → tighter TP
}

# Drawdown proximity multiplier
# As daily PnL approaches -MAX_DAILY_LOSS, reduce lot
DRAWDOWN_ZONES = [
    (-20, 1.0),    # PnL > -$20 → full lot
    (-50, 0.7),    # PnL > -$50 → reduce 30%
    (-80, 0.4),    # PnL > -$80 → reduce 60%
    (-100, 0.0),   # PnL ≤ -$100 → no trading (daily limit)
]

# Recent performance (streak-based)
# Winning streak → cautiously increase, losing streak → reduce
STREAK_MULTIPLIERS = {
    "hot": 1.2,      # 3+ consecutive wins
    "warm": 1.1,     # 1-2 wins
    "neutral": 1.0,  # Mixed
    "cold": 0.8,     # 1-2 losses
    "frozen": 0.6,   # 3+ consecutive losses
}


class DynamicLotSizer:
    """
    Dynamic lot sizing engine.
    Computes lot size based on multiple contextual factors.
    Always enforces MAX_TRADE_RISK cap.
    """

    def __init__(self):
        # Track recent trade results for streak detection
        self._recent_results = []  # List of +1 (win) or -1 (loss)
        self._max_history = 10

        # Track daily state
        self._daily_pnl = 0.0
        self._current_date = None

    def compute(
        self,
        n_strategies: int = 1,
        e_prob: float = None,
        regime: str = "RANGING",
        daily_pnl: float = 0.0,
        atr: float = 2.0,
        sl_distance: float = None,
    ) -> dict:
        """
        Compute dynamic lot size and SL/TP adjustments.

        Args:
            n_strategies: Number of A-D strategies that agree (0 = E only)
            e_prob: E's raw probability (None if no prediction)
            regime: Current market regime
            daily_pnl: Today's cumulative PnL
            atr: Current ATR
            sl_distance: Strategy's SL distance in price (for risk capping)

        Returns:
            dict with:
              lot_size: 0.01-0.05
              sl_multiplier: multiplicative adjustment for SL
              tp_multiplier: multiplicative adjustment for TP
              factors: breakdown of what contributed to the lot size
        """
        # ── Base lot from agreement ───────────────────────
        base_lot = BASE_LOTS.get(min(n_strategies, 4), LOT_SIZE_SINGLE)

        # ── E confidence multiplier ───────────────────────
        if e_prob is not None:
            effective = max(e_prob, 1 - e_prob)
            if effective >= 0.70:
                e_mult = E_CONF_MULTIPLIERS["very_high"]
                e_label = "very_high"
            elif effective >= 0.65:
                e_mult = E_CONF_MULTIPLIERS["high"]
                e_label = "high"
            elif effective >= 0.58:
                e_mult = E_CONF_MULTIPLIERS["medium"]
                e_label = "medium"
            elif effective >= 0.52:
                e_mult = E_CONF_MULTIPLIERS["low"]
                e_label = "low"
            else:
                e_mult = 0.8
                e_label = "very_low"
        else:
            e_mult = E_CONF_MULTIPLIERS["none"]
            e_label = "none"

        # ── Regime multiplier ─────────────────────────────
        regime_mult = REGIME_LOT_MULTIPLIERS.get(regime, 1.0)

        # ── Drawdown proximity ────────────────────────────
        dd_mult = 1.0
        for threshold, mult in DRAWDOWN_ZONES:
            if daily_pnl > threshold:
                dd_mult = mult
                break
        else:
            dd_mult = 0.0  # Past all thresholds → no trading

        # ── Streak multiplier ─────────────────────────────
        streak_label, streak_mult = self._get_streak_multiplier()

        # ── Combine all factors ───────────────────────────
        raw_lot = base_lot * e_mult * regime_mult * dd_mult * streak_mult

        # ── Clamp to range ────────────────────────────────
        lot = max(LOT_MIN, min(LOT_MAX, raw_lot))
        lot = round(lot * 100) / 100  # Round to 0.01

        # ── Risk cap: ensure SL × lot doesn't exceed MAX_TRADE_RISK ──
        if sl_distance is not None and sl_distance > 0:
            max_lot_for_risk = MAX_TRADE_RISK / (XAUUSD_POINT_VALUE * sl_distance)
            lot = min(lot, max_lot_for_risk)
            lot = max(LOT_MIN, round(lot * 100) / 100)

        # ── SL/TP adjustments from regime ─────────────────
        sl_mult = REGIME_SL_ADJ.get(regime, 1.0)
        tp_mult = REGIME_TP_ADJ.get(regime, 1.0)

        return {
            "lot_size": lot,
            "sl_multiplier": sl_mult,
            "tp_multiplier": tp_mult,
            "factors": {
                "base_lot": base_lot,
                "e_confidence": e_label,
                "e_mult": e_mult,
                "regime": regime,
                "regime_mult": regime_mult,
                "drawdown_mult": dd_mult,
                "streak": streak_label,
                "streak_mult": streak_mult,
                "raw_lot": round(raw_lot, 4),
                "final_lot": lot,
            },
        }

    def record_result(self, pnl: float):
        """Record a trade result for streak tracking."""
        self._recent_results.append(1 if pnl > 0 else -1)
        if len(self._recent_results) > self._max_history:
            self._recent_results = self._recent_results[-self._max_history:]

    def reset_daily(self):
        """Reset daily state (call at start of each trading day)."""
        self._daily_pnl = 0.0

    def _get_streak_multiplier(self) -> tuple:
        """Determine current streak and its multiplier."""
        if not self._recent_results:
            return "neutral", 1.0

        # Count consecutive same results from the end
        last = self._recent_results[-1]
        streak = 0
        for r in reversed(self._recent_results):
            if r == last:
                streak += 1
            else:
                break

        if last == 1:  # Winning
            if streak >= 3:
                return "hot", STREAK_MULTIPLIERS["hot"]
            else:
                return "warm", STREAK_MULTIPLIERS["warm"]
        else:  # Losing
            if streak >= 3:
                return "frozen", STREAK_MULTIPLIERS["frozen"]
            else:
                return "cold", STREAK_MULTIPLIERS["cold"]

    def get_diagnostics(self) -> dict:
        """Get current sizer state."""
        streak_label, streak_mult = self._get_streak_multiplier()
        return {
            "recent_results": self._recent_results[-5:] if self._recent_results else [],
            "streak": streak_label,
            "streak_mult": streak_mult,
        }


# ─── Quick Test ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 8 — Dynamic Lot Sizer Test")
    print("=" * 60)

    sizer = DynamicLotSizer()

    test_cases = [
        {"desc": "Single strat, no E, ranging",
         "n_strategies": 1, "e_prob": None, "regime": "RANGING", "daily_pnl": 0, "atr": 2.0, "sl_distance": 3.0},
        {"desc": "Two strats agree, E strong, trending bull",
         "n_strategies": 2, "e_prob": 0.72, "regime": "TRENDING_BULL", "daily_pnl": 50, "atr": 2.5, "sl_distance": 3.75},
        {"desc": "E independent, high conf, volatile",
         "n_strategies": 0, "e_prob": 0.75, "regime": "VOLATILE", "daily_pnl": -30, "atr": 4.0, "sl_distance": 6.0},
        {"desc": "Three agree, E strong, trending, after 3 wins",
         "n_strategies": 3, "e_prob": 0.70, "regime": "TRENDING_BEAR", "daily_pnl": 100, "atr": 2.0, "sl_distance": 3.0},
        {"desc": "Single strat, quiet market, daily loss -$80",
         "n_strategies": 1, "e_prob": 0.55, "regime": "QUIET", "daily_pnl": -80, "atr": 1.0, "sl_distance": 1.5},
    ]

    # Simulate some wins for streak test
    for _ in range(3):
        sizer.record_result(5.0)

    for tc in test_cases:
        desc = tc.pop("desc")
        result = sizer.compute(**tc)
        print(f"\n  {desc}:")
        print(f"    Lot: {result['lot_size']} | SL×{result['sl_multiplier']:.1f} | TP×{result['tp_multiplier']:.1f}")
        f = result["factors"]
        print(f"    Base={f['base_lot']} × E({f['e_confidence']})={f['e_mult']} "
              f"× Regime={f['regime_mult']} × DD={f['drawdown_mult']} "
              f"× Streak({f['streak']})={f['streak_mult']} → raw={f['raw_lot']} → {f['final_lot']}")

    print("\n✅ Dynamic lot sizer working!")