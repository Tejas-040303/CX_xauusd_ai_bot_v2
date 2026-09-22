"""
Project MIDAS v2 — Strategy A: SMC Liquidity Sweep + Order Block
Phase 4 Implementation

Detects BOS + OB on M15, enters on M5 when price retraces into the OB zone.
Liquidity sweep is a bonus confirmation, not required.

Sub-Conditions (Required):
  1. BOS detected on M15 (swing break establishing direction)
  2. OB identified (last opposing candle body before BOS impulse, min 0.2x ATR)
  3. OB not invalidated (price hasn't closed through far edge)
  4. Price retraces into OB zone on M5
  5. M5 rejection candle (strong body > 60%, close near extreme)

Confidence Bonuses:
  Base 60% + sweep nearby +20% + OB in discount/premium +10% + trend aligned +10%
  Max = 100%

SL: Beyond OB far edge + 2x spread
TP: 1:1.5 RR base, extend to 1:2.5 if sweep confirmed
"""

import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import (
    ATR_PERIOD, SPREAD_SIMULATION_PIPS, PIP_VALUE,
    CONFIDENCE_THRESHOLD
)
from backtester import StrategyBase, TradeSignal


# ─── OB Zone Tracker ────────────────────────────────────

@dataclass
class OBZone:
    """Represents a single tracked Order Block zone."""
    creation_idx: int           # M15 candle index where BOS was detected
    creation_time: object       # datetime of BOS detection
    direction: str              # "bull" or "bear"
    ob_high: float              # Top of OB body
    ob_low: float               # Bottom of OB body
    bos_level: float            # The swing level that was broken
    impulse_size: float         # Size of move that caused BOS
    is_tested: bool = False     # Price has entered the zone (trade taken)
    is_invalidated: bool = False  # Price closed through far edge
    sweep_detected: bool = False  # Liquidity sweep found nearby
    sweep_count: int = 0        # How many equal highs/lows were swept

    @property
    def is_valid(self) -> bool:
        return not self.is_tested and not self.is_invalidated

    @property
    def midpoint(self) -> float:
        return (self.ob_high + self.ob_low) / 2


class OBTracker:
    """
    Tracks active Order Block zones from M15 BOS detections.
    Handles invalidation, sweep association, and zone management.
    """

    MAX_OBS_PER_DIRECTION = 5
    SWEEP_PROXIMITY_CANDLES = 45  # Sweep within 45 M15 candles of BOS
    SWEEP_MIN_PIPS = 2            # Minimum sweep distance (2 pips = $0.20)

    def __init__(self):
        self.active_bull_obs: List[OBZone] = []
        self.active_bear_obs: List[OBZone] = []
        self.last_m15_idx = -1

        # Sweep tracking: recent equal highs/lows and sweeps
        self.recent_equal_highs: List[Dict] = []  # [{idx, level, count}, ...]
        self.recent_equal_lows: List[Dict] = []
        self.recent_sweeps_above: List[Dict] = []  # [{idx, level}, ...]
        self.recent_sweeps_below: List[Dict] = []

    def update_from_m15(self, m15_row: pd.Series, m15_idx: int):
        """Process a new M15 candle for BOS, OBs, and sweeps."""
        if m15_idx <= self.last_m15_idx:
            return
        self.last_m15_idx = m15_idx

        # ── Check for new Bullish BOS + OB ──
        if m15_row.get("bos_bull", 0) == 1:
            ob_high = m15_row.get("ob_bull_high", np.nan)
            ob_low = m15_row.get("ob_bull_low", np.nan)
            bos_level = m15_row.get("bos_bull_level", np.nan)
            impulse = m15_row.get("ob_bull_impulse_size", np.nan)

            if not np.isnan(ob_high) and not np.isnan(ob_low) and ob_high > ob_low:
                ob = OBZone(
                    creation_idx=m15_idx,
                    creation_time=m15_row.get("datetime", None),
                    direction="bull",
                    ob_high=ob_high,
                    ob_low=ob_low,
                    bos_level=bos_level if not np.isnan(bos_level) else ob_high,
                    impulse_size=impulse if not np.isnan(impulse) else 0,
                )
                # Check for nearby sweep (equal lows swept)
                self._check_sweep_for_ob(ob, m15_row, m15_idx)
                self._add_ob(ob, self.active_bull_obs)

        # ── Check for new Bearish BOS + OB ──
        if m15_row.get("bos_bear", 0) == 1:
            ob_high = m15_row.get("ob_bear_high", np.nan)
            ob_low = m15_row.get("ob_bear_low", np.nan)
            bos_level = m15_row.get("bos_bear_level", np.nan)
            impulse = m15_row.get("ob_bear_impulse_size", np.nan)

            if not np.isnan(ob_high) and not np.isnan(ob_low) and ob_high > ob_low:
                ob = OBZone(
                    creation_idx=m15_idx,
                    creation_time=m15_row.get("datetime", None),
                    direction="bear",
                    ob_high=ob_high,
                    ob_low=ob_low,
                    bos_level=bos_level if not np.isnan(bos_level) else ob_low,
                    impulse_size=impulse if not np.isnan(impulse) else 0,
                )
                self._check_sweep_for_ob(ob, m15_row, m15_idx)
                self._add_ob(ob, self.active_bear_obs)

        # ── Track equal highs/lows for sweep detection ──
        if m15_row.get("equal_highs", 0) == 1:
            level = m15_row.get("equal_highs_level", np.nan)
            count = m15_row.get("equal_highs_count", 2)
            if not np.isnan(level):
                self.recent_equal_highs.append({
                    "idx": m15_idx, "level": level, "count": int(count)
                })
                # Keep only recent
                self.recent_equal_highs = [
                    e for e in self.recent_equal_highs
                    if m15_idx - e["idx"] <= self.SWEEP_PROXIMITY_CANDLES
                ]

        if m15_row.get("equal_lows", 0) == 1:
            level = m15_row.get("equal_lows_level", np.nan)
            count = m15_row.get("equal_lows_count", 2)
            if not np.isnan(level):
                self.recent_equal_lows.append({
                    "idx": m15_idx, "level": level, "count": int(count)
                })
                self.recent_equal_lows = [
                    e for e in self.recent_equal_lows
                    if m15_idx - e["idx"] <= self.SWEEP_PROXIMITY_CANDLES
                ]

        # ── Detect sweeps on this candle ──
        self._detect_sweeps(m15_row, m15_idx)

    def update_invalidations(self, m5_row: pd.Series):
        """Check if any active OBs were invalidated by M5 price action."""
        close = m5_row["close"]

        for ob in self.active_bull_obs:
            if not ob.is_valid:
                continue
            # Bullish OB invalidated if close breaks below OB low
            if close < ob.ob_low:
                ob.is_invalidated = True

        for ob in self.active_bear_obs:
            if not ob.is_valid:
                continue
            # Bearish OB invalidated if close breaks above OB high
            if close > ob.ob_high:
                ob.is_invalidated = True

    def get_best_bull_ob(self, current_price: float) -> Optional[OBZone]:
        """Get closest valid bullish OB that price could retrace into.
        For bullish OB: price is above OB, looking for pullback down into it."""
        valid = [ob for ob in self.active_bull_obs if ob.is_valid]
        if not valid:
            return None
        # OB must be below current price (price pulls back down into it)
        below = [ob for ob in valid if ob.ob_high < current_price]
        if not below:
            return None
        # Closest to current price
        return min(below, key=lambda ob: current_price - ob.ob_high)

    def get_best_bear_ob(self, current_price: float) -> Optional[OBZone]:
        """Get closest valid bearish OB that price could retrace into.
        For bearish OB: price is below OB, looking for pullback up into it."""
        valid = [ob for ob in self.active_bear_obs if ob.is_valid]
        if not valid:
            return None
        # OB must be above current price (price pulls back up into it)
        above = [ob for ob in valid if ob.ob_low > current_price]
        if not above:
            return None
        return min(above, key=lambda ob: ob.ob_low - current_price)

    def _check_sweep_for_ob(self, ob: OBZone, m15_row: pd.Series, m15_idx: int):
        """Check if there was a liquidity sweep near this BOS event."""
        sweep_min = self.SWEEP_MIN_PIPS * PIP_VALUE

        if ob.direction == "bull":
            # Bullish OB: look for sweep below equal lows (sell-side liquidity grabbed)
            for eq in self.recent_equal_lows:
                if abs(m15_idx - eq["idx"]) <= self.SWEEP_PROXIMITY_CANDLES:
                    # Check if price went below equal lows and came back
                    low = m15_row.get("low", np.nan) if not np.isnan(m15_row.get("low", np.nan)) else 0
                    # Also check recent sweeps
                    for sweep in self.recent_sweeps_below:
                        if (abs(m15_idx - sweep["idx"]) <= self.SWEEP_PROXIMITY_CANDLES and
                            abs(sweep["level"] - eq["level"]) <= sweep_min * 5):
                            ob.sweep_detected = True
                            ob.sweep_count = eq["count"]
                            return

        elif ob.direction == "bear":
            # Bearish OB: look for sweep above equal highs (buy-side liquidity grabbed)
            for eq in self.recent_equal_highs:
                if abs(m15_idx - eq["idx"]) <= self.SWEEP_PROXIMITY_CANDLES:
                    for sweep in self.recent_sweeps_above:
                        if (abs(m15_idx - sweep["idx"]) <= self.SWEEP_PROXIMITY_CANDLES and
                            abs(sweep["level"] - eq["level"]) <= sweep_min * 5):
                            ob.sweep_detected = True
                            ob.sweep_count = eq["count"]
                            return

    def _detect_sweeps(self, m15_row: pd.Series, m15_idx: int):
        """Detect if this candle swept past any equal highs/lows."""
        high = m15_row.get("high", np.nan)
        low = m15_row.get("low", np.nan)
        close = m15_row.get("close", np.nan)
        sweep_min = self.SWEEP_MIN_PIPS * PIP_VALUE

        if np.isnan(high) or np.isnan(low) or np.isnan(close):
            return

        # Sweep above equal highs: wick goes above but close is below
        for eq in self.recent_equal_highs:
            if high > eq["level"] + sweep_min and close < eq["level"]:
                self.recent_sweeps_above.append({"idx": m15_idx, "level": eq["level"]})

        # Sweep below equal lows: wick goes below but close is above
        for eq in self.recent_equal_lows:
            if low < eq["level"] - sweep_min and close > eq["level"]:
                self.recent_sweeps_below.append({"idx": m15_idx, "level": eq["level"]})

        # Cleanup old sweeps
        self.recent_sweeps_above = [
            s for s in self.recent_sweeps_above
            if m15_idx - s["idx"] <= self.SWEEP_PROXIMITY_CANDLES
        ]
        self.recent_sweeps_below = [
            s for s in self.recent_sweeps_below
            if m15_idx - s["idx"] <= self.SWEEP_PROXIMITY_CANDLES
        ]

    def _add_ob(self, ob: OBZone, ob_list: List[OBZone]):
        """Add OB, remove invalidated, keep max count."""
        ob_list.append(ob)
        ob_list[:] = [o for o in ob_list if o.is_valid]
        if len(ob_list) > self.MAX_OBS_PER_DIRECTION:
            ob_list.sort(key=lambda o: o.creation_idx, reverse=True)
            ob_list[:] = ob_list[:self.MAX_OBS_PER_DIRECTION]


# ─── Strategy A ──────────────────────────────────────────

class StrategyA(StrategyBase):
    """
    Strategy A: SMC Liquidity Sweep + Order Block.

    Highest conviction, lowest frequency strategy.
    Detects BOS + OB on M15, enters on M5 retrace + rejection.
    Sweep is bonus confirmation, not required.
    """

    name = "StrategyA_SMC_OB"

    def __init__(self):
        self.ob_tracker = OBTracker()
        self._last_m15_time = None
        self.diag = {
            "candles_processed": 0,
            "ob_found": 0,
            "cond3_not_invalidated": 0,
            "cond4_retrace_into_ob": 0,
            "cond5_rejection": 0,
            "all_required_pass": 0,
            "above_confidence": 0,
            "signals_generated": 0,
            "sweep_bonus_applied": 0,
        }

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        if idx < 50:
            return None

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        self.diag["candles_processed"] += 1

        # Update OB tracker from M15 data
        self._update_m15(row)

        # Update invalidations from M5 price
        self.ob_tracker.update_invalidations(row)

        # Check entries
        signal = self._check_bull_entry(df, idx, row, prev)
        if signal is not None:
            self.diag["signals_generated"] += 1
            return signal

        signal = self._check_bear_entry(df, idx, row, prev)
        if signal is not None:
            self.diag["signals_generated"] += 1
        return signal

    def _update_m15(self, m5_row: pd.Series):
        """Update OB tracker with latest M15 data."""
        m15_time = m5_row.get("m15_datetime")
        if m15_time is None or pd.isna(m15_time):
            return

        m15_time_str = str(m15_time)
        if m15_time_str == self._last_m15_time:
            return
        self._last_m15_time = m15_time_str

        # Build M15 row from merged columns
        m15_data = {}
        for col in m5_row.index:
            if col.startswith("m15_"):
                m15_data[col[4:]] = m5_row[col]

        if not m15_data:
            return

        m15_series = pd.Series(m15_data)
        raw_idx = m5_row.get("m15_idx", np.nan)
        if pd.isna(raw_idx):
            return
        m15_idx = int(raw_idx)

        self.ob_tracker.update_from_m15(m15_series, m15_idx)

    def _check_bull_entry(self, df, idx, row, prev) -> Optional[TradeSignal]:
        """Check for bullish OB entry (buy on retrace into bullish OB)."""
        atr = row.get(f"atr_{ATR_PERIOD}", np.nan)
        if np.isnan(atr) or atr <= 0:
            return None

        current_price = row["close"]
        ob = self.ob_tracker.get_best_bull_ob(current_price)
        if ob is None:
            return None

        self.diag["ob_found"] += 1
        sub_conditions = {}

        # Cond 1+2: BOS + OB (already confirmed by tracker)
        sub_conditions["bos_detected"] = True
        sub_conditions["ob_identified"] = True

        # Cond 3: OB not invalidated
        cond3 = ob.is_valid
        sub_conditions["ob_valid"] = cond3
        if cond3:
            self.diag["cond3_not_invalidated"] += 1

        # Cond 4: Price retraces into OB zone
        # Low must touch OB zone (enter from above)
        cond4 = (row["low"] <= ob.ob_high and current_price >= ob.ob_low)
        sub_conditions["retrace_into_ob"] = cond4
        if cond4:
            self.diag["cond4_retrace_into_ob"] += 1

        # Cond 5: M5 rejection candle
        body = abs(row["close"] - row["open"])
        candle_range = row["high"] - row["low"]
        cond5 = False
        if candle_range > 0:
            cond5 = (
                row["close"] > row["open"] and          # Bullish
                body > 0.6 * candle_range and            # Strong body
                row["close"] > (row["high"] - 0.25 * candle_range) and  # Close near high
                row["close"] > ob.ob_low                  # Close above OB low
            )
            # Also accept engulfing
            if not cond5 and row.get("is_engulfing_bull", 0) == 1:
                cond5 = (row["low"] <= ob.ob_high and row["close"] > ob.ob_low)
            # Also accept pin bar
            if not cond5 and row.get("is_pin_bar_bull", 0) == 1:
                cond5 = (row["low"] <= ob.ob_high and row["close"] > ob.ob_low)
        sub_conditions["m5_rejection"] = cond5
        if cond5:
            self.diag["cond5_rejection"] += 1

        # ── Check all required ──
        if not (cond3 and cond4 and cond5):
            return None

        self.diag["all_required_pass"] += 1

        # ── Confidence Scoring ──
        confidence = 60

        # Sweep bonus (+20%)
        if ob.sweep_detected:
            confidence += 20
            self.diag["sweep_bonus_applied"] += 1
        sub_conditions["sweep_nearby"] = ob.sweep_detected
        sub_conditions["sweep_count"] = ob.sweep_count

        # OB in discount zone (+10%)
        # Discount = below 50% of the impulse leg
        if ob.impulse_size > 0:
            impulse_midpoint = ob.ob_low + ob.impulse_size / 2
            in_discount = (current_price < impulse_midpoint)
        else:
            in_discount = False
        if in_discount:
            confidence += 10
        sub_conditions["ob_in_discount"] = in_discount

        # Trend aligned (+10%)
        m15_ema_slope = row.get("m15_ema_50_slope", np.nan)
        trend_aligned = (not np.isnan(m15_ema_slope) and m15_ema_slope > 0)
        if trend_aligned:
            confidence += 10
        sub_conditions["trend_aligned"] = trend_aligned

        if confidence < CONFIDENCE_THRESHOLD:
            return None

        self.diag["above_confidence"] += 1

        # ── Mark OB as tested ──
        ob.is_tested = True

        # ── SL: below OB low + buffer ──
        spread_buffer = SPREAD_SIMULATION_PIPS * PIP_VALUE * 2
        sl_price = ob.ob_low - spread_buffer

        # Min SL = 0.5x ATR
        min_sl = current_price - 0.5 * atr
        if sl_price > min_sl:
            sl_price = min_sl

        if sl_price >= current_price:
            return None

        # ── TP ──
        risk_distance = current_price - sl_price

        # Base 1:1.5, extend to 1:2.5 if sweep confirmed
        if ob.sweep_detected:
            tp_price = current_price + risk_distance * 2.5
        else:
            tp_price = current_price + risk_distance * 1.5

        return TradeSignal(
            datetime=row["datetime"],
            direction="buy",
            strategy_name=self.name,
            confidence=confidence,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_price=current_price,
            sub_conditions=sub_conditions,
        )

    def _check_bear_entry(self, df, idx, row, prev) -> Optional[TradeSignal]:
        """Check for bearish OB entry (sell on retrace into bearish OB)."""
        atr = row.get(f"atr_{ATR_PERIOD}", np.nan)
        if np.isnan(atr) or atr <= 0:
            return None

        current_price = row["close"]
        ob = self.ob_tracker.get_best_bear_ob(current_price)
        if ob is None:
            return None

        self.diag["ob_found"] += 1
        sub_conditions = {}

        sub_conditions["bos_detected"] = True
        sub_conditions["ob_identified"] = True

        # Cond 3: not invalidated
        cond3 = ob.is_valid
        sub_conditions["ob_valid"] = cond3
        if cond3:
            self.diag["cond3_not_invalidated"] += 1

        # Cond 4: retrace into OB (from below, price enters zone)
        cond4 = (row["high"] >= ob.ob_low and current_price <= ob.ob_high)
        sub_conditions["retrace_into_ob"] = cond4
        if cond4:
            self.diag["cond4_retrace_into_ob"] += 1

        # Cond 5: M5 rejection
        body = abs(row["close"] - row["open"])
        candle_range = row["high"] - row["low"]
        cond5 = False
        if candle_range > 0:
            cond5 = (
                row["close"] < row["open"] and
                body > 0.6 * candle_range and
                row["close"] < (row["low"] + 0.25 * candle_range) and
                row["close"] < ob.ob_high
            )
            if not cond5 and row.get("is_engulfing_bear", 0) == 1:
                cond5 = (row["high"] >= ob.ob_low and row["close"] < ob.ob_high)
            if not cond5 and row.get("is_pin_bar_bear", 0) == 1:
                cond5 = (row["high"] >= ob.ob_low and row["close"] < ob.ob_high)
        sub_conditions["m5_rejection"] = cond5
        if cond5:
            self.diag["cond5_rejection"] += 1

        if not (cond3 and cond4 and cond5):
            return None

        self.diag["all_required_pass"] += 1

        # ── Confidence ──
        confidence = 60

        if ob.sweep_detected:
            confidence += 20
            self.diag["sweep_bonus_applied"] += 1
        sub_conditions["sweep_nearby"] = ob.sweep_detected
        sub_conditions["sweep_count"] = ob.sweep_count

        # OB in premium zone (above 50% of impulse)
        if ob.impulse_size > 0:
            impulse_midpoint = ob.ob_high - ob.impulse_size / 2
            in_premium = (current_price > impulse_midpoint)
        else:
            in_premium = False
        if in_premium:
            confidence += 10
        sub_conditions["ob_in_premium"] = in_premium

        m15_ema_slope = row.get("m15_ema_50_slope", np.nan)
        trend_aligned = (not np.isnan(m15_ema_slope) and m15_ema_slope < 0)
        if trend_aligned:
            confidence += 10
        sub_conditions["trend_aligned"] = trend_aligned

        if confidence < CONFIDENCE_THRESHOLD:
            return None

        self.diag["above_confidence"] += 1

        ob.is_tested = True

        # ── SL ──
        spread_buffer = SPREAD_SIMULATION_PIPS * PIP_VALUE * 2
        sl_price = ob.ob_high + spread_buffer

        min_sl = current_price + 0.5 * atr
        if sl_price < min_sl:
            sl_price = min_sl

        if sl_price <= current_price:
            return None

        # ── TP ──
        risk_distance = sl_price - current_price

        if ob.sweep_detected:
            tp_price = current_price - risk_distance * 2.5
        else:
            tp_price = current_price - risk_distance * 1.5

        return TradeSignal(
            datetime=row["datetime"],
            direction="sell",
            strategy_name=self.name,
            confidence=confidence,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_price=current_price,
            sub_conditions=sub_conditions,
        )

    def print_diagnostics(self):
        d = self.diag
        print(f"\n  STRATEGY A DIAGNOSTICS:")
        print(f"  Candles processed:        {d['candles_processed']:>10,}")
        print(f"  OB candidates found:      {d['ob_found']:>10,}")
        print(f"  Cond3 OB valid:           {d['cond3_not_invalidated']:>10,}")
        print(f"  Cond4 Retrace into OB:    {d['cond4_retrace_into_ob']:>10,}")
        print(f"  Cond5 M5 rejection:       {d['cond5_rejection']:>10,}")
        print(f"  All required pass:        {d['all_required_pass']:>10,}")
        print(f"  Above confidence:         {d['above_confidence']:>10,}")
        print(f"  Sweep bonus applied:      {d['sweep_bonus_applied']:>10,}")
        print(f"  Signals generated:        {d['signals_generated']:>10,}")

        # OB tracker stats
        bull_obs = len(self.ob_tracker.active_bull_obs)
        bear_obs = len(self.ob_tracker.active_bear_obs)
        print(f"  Active bull OBs (final):  {bull_obs:>10,}")
        print(f"  Active bear OBs (final):  {bear_obs:>10,}")