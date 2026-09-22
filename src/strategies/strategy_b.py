"""
Project MIDAS v2 — Strategy B: Fair Value Gap (FVG) Rebalancing
Phase 1 Implementation

Detects FVGs on M15, enters on M5 when price retraces into the gap.

Sub-Conditions:
  1. FVG Present (required): Valid 3-candle gap on M15, 0.5-2.5x ATR, not expired
  2. Trend Aligned (required): FVG direction matches EMA(50) slope + price side
  3. First Touch (weighted): First time price enters this FVG = +20%, second = +0%
  4. Price Reaches CE (required): Price enters FVG and reaches 50% level
  5. M5 Rejection (required): M5 candle closes back in impulse direction
  6. Not Stacked (bonus): No more than 2 unfilled FVGs same direction nearby (+10%)
  7. Not News FVG (bonus): FVG not created near high-impact news (+10%)

Confidence: Required conditions must ALL be true (else 0%).
  Base 60% + First Touch 20% + Not Stacked 10% + Not News 10% = max 100%

Entry: at CE (50%) level of FVG
SL: beyond far edge of FVG + 2x spread buffer
TP: 1:1 RR base, extend to 1:2 if FVG created by strong impulse (>2x ATR move)
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import (
    ATR_PERIOD, SPREAD_SIMULATION_PIPS, PIP_VALUE,
    CONFIDENCE_THRESHOLD
)
from backtester import StrategyBase, TradeSignal


# ─── FVG Tracker ─────────────────────────────────────────

@dataclass
class FVGZone:
    """Represents a single tracked Fair Value Gap."""
    creation_time: datetime
    creation_idx: int           # M15 candle index where FVG was detected
    direction: str              # "bull" or "bear"
    top: float                  # Upper boundary of FVG zone
    bottom: float               # Lower boundary of FVG zone
    ce: float                   # Consequent Encroachment (50% level)
    size: float                 # Gap size in price
    size_atr: float             # Gap size normalized by ATR
    impulse_size: float         # Size of the move that created the FVG (candle 2 range)
    touch_count: int = 0        # How many times price has entered this zone
    is_filled: bool = False     # Fully filled (price closed through far edge)
    is_expired: bool = False    # Exceeded 100 M15 candles age
    is_news_fvg: bool = False   # Created near high-impact news (placeholder)

    @property
    def is_valid(self) -> bool:
        """FVG is still tradeable."""
        return not self.is_filled and not self.is_expired and self.touch_count < 3

    @property
    def age_candles(self) -> int:
        """Placeholder — set externally during tracking."""
        return 0


class FVGTracker:
    """
    Maintains a list of active FVGs detected from M15 data.
    Handles expiry, fill detection, touch counting, and stacking checks.
    """

    MAX_FVGS_PER_DIRECTION = 3
    MAX_AGE_M15_CANDLES = 100     # ~25 hours of trading
    STACK_WINDOW_M15_CANDLES = 20  # FVGs within 20 M15 candles = stacked

    def __init__(self):
        self.active_bull_fvgs: List[FVGZone] = []
        self.active_bear_fvgs: List[FVGZone] = []
        self.last_m15_idx_processed: int = -1

    def update_from_m15(self, m15_row: pd.Series, m15_idx: int):
        """
        Check if a new FVG was detected on this M15 candle.
        Called once per new M15 candle.
        """
        if m15_idx <= self.last_m15_idx_processed:
            return  # Already processed this M15 candle
        self.last_m15_idx_processed = m15_idx

        # Expire old FVGs
        self._expire_old(m15_idx)

        # Check for new bullish FVG
        if not np.isnan(m15_row.get("fvg_bull_top", np.nan)):
            fvg = FVGZone(
                creation_time=m15_row["datetime"],
                creation_idx=m15_idx,
                direction="bull",
                top=m15_row["fvg_bull_top"],
                bottom=m15_row["fvg_bull_bottom"],
                ce=m15_row["fvg_bull_ce"],
                size=m15_row["fvg_bull_size"],
                size_atr=m15_row["fvg_bull_size_atr"],
                impulse_size=m15_row.get("candle_range", m15_row.get("fvg_bull_size", 0)) ,
            )
            self._add_fvg(fvg, self.active_bull_fvgs)

        # Check for new bearish FVG
        if not np.isnan(m15_row.get("fvg_bear_top", np.nan)):
            fvg = FVGZone(
                creation_time=m15_row["datetime"],
                creation_idx=m15_idx,
                direction="bear",
                top=m15_row["fvg_bear_top"],
                bottom=m15_row["fvg_bear_bottom"],
                ce=m15_row["fvg_bear_ce"],
                size=m15_row["fvg_bear_size"],
                size_atr=m15_row["fvg_bear_size_atr"],
                impulse_size=m15_row.get("candle_range", m15_row.get("fvg_bear_size", 0)),
            )
            self._add_fvg(fvg, self.active_bear_fvgs)

    def update_fills(self, m5_row: pd.Series):
        """
        Check if any active FVGs were filled by this M5 candle.
        A close THROUGH the far edge = filled/invalidated.
        A wick into the zone = touch (but not filled).
        """
        close = m5_row["close"]
        low = m5_row["low"]
        high = m5_row["high"]

        # Check bullish FVGs
        for fvg in self.active_bull_fvgs:
            if not fvg.is_valid:
                continue
            # Price entered the zone (wick or body touched top boundary or lower)
            if low <= fvg.top:
                # Close BELOW the bottom = fully filled/invalidated
                if close < fvg.bottom:
                    fvg.is_filled = True

        # Check bearish FVGs
        for fvg in self.active_bear_fvgs:
            if not fvg.is_valid:
                continue
            # Price entered the zone (wick or body touched bottom boundary or higher)
            if high >= fvg.bottom:
                # Close ABOVE the top = fully filled/invalidated
                if close > fvg.top:
                    fvg.is_filled = True

    def get_best_bull_fvg(self, current_price: float) -> Optional[FVGZone]:
        """
        Get the closest valid bullish FVG that price could trade.
        For bullish FVG: price retraces DOWN into zone, so we need
        price to be within or near the zone (above the far edge/bottom).
        """
        valid = [f for f in self.active_bull_fvgs if f.is_valid]
        if not valid:
            return None
        # Price must be above the far edge (bottom) — hasn't crashed through
        # AND price must be near/in the zone (within top + 1x zone size buffer)
        candidates = [
            f for f in valid
            if current_price > f.bottom  # Not blown through far edge
            and current_price < f.top + (f.size * 2)  # Not too far above (within 2x gap)
        ]
        if not candidates:
            return None
        # Closest to CE level
        return min(candidates, key=lambda f: abs(current_price - f.ce))

    def get_best_bear_fvg(self, current_price: float) -> Optional[FVGZone]:
        """
        Get the closest valid bearish FVG that price could trade.
        For bearish FVG: price retraces UP into zone, so we need
        price to be within or near the zone (below the far edge/top).
        """
        valid = [f for f in self.active_bear_fvgs if f.is_valid]
        if not valid:
            return None
        # Price must be below the far edge (top) — hasn't blown through
        # AND price must be near/in the zone (within bottom - 1x zone size buffer)
        candidates = [
            f for f in valid
            if current_price < f.top  # Not blown through far edge
            and current_price > f.bottom - (f.size * 2)  # Not too far below (within 2x gap)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda f: abs(current_price - f.ce))

    def is_stacked(self, fvg: FVGZone) -> bool:
        """Check if there are 2+ FVGs in the same direction within STACK_WINDOW candles."""
        fvg_list = self.active_bull_fvgs if fvg.direction == "bull" else self.active_bear_fvgs
        nearby = [
            f for f in fvg_list
            if f.is_valid and f is not fvg
            and abs(f.creation_idx - fvg.creation_idx) <= self.STACK_WINDOW_M15_CANDLES
        ]
        return len(nearby) >= 2

    def _add_fvg(self, fvg: FVGZone, fvg_list: List[FVGZone]):
        """Add FVG to list, keeping only MAX_FVGS_PER_DIRECTION most recent."""
        fvg_list.append(fvg)
        # Remove invalid ones first
        fvg_list[:] = [f for f in fvg_list if f.is_valid]
        # Keep only most recent
        if len(fvg_list) > self.MAX_FVGS_PER_DIRECTION:
            fvg_list.sort(key=lambda f: f.creation_idx, reverse=True)
            fvg_list[:] = fvg_list[:self.MAX_FVGS_PER_DIRECTION]

    def _expire_old(self, current_m15_idx: int):
        """Mark FVGs older than MAX_AGE as expired."""
        for fvg in self.active_bull_fvgs + self.active_bear_fvgs:
            if current_m15_idx - fvg.creation_idx > self.MAX_AGE_M15_CANDLES:
                fvg.is_expired = True


# ─── Strategy B ──────────────────────────────────────────

class StrategyB(StrategyBase):
    """
    Strategy B: Fair Value Gap (FVG) Rebalancing.

    Detects FVGs on M15 data, enters trades on M5 when price
    retraces into the gap and shows rejection at the CE level.

    Requires merged DataFrame with M15 FVG columns prefixed with 'm15_'.
    """

    name = "StrategyB_FVG"

    def __init__(self):
        self.fvg_tracker = FVGTracker()
        self._last_m15_time = None
        # Diagnostic counters
        self.diag = {
            "candles_processed": 0,
            "fvg_found_for_candle": 0,      # FVG tracker returned a candidate
            "cond1_fvg_valid": 0,
            "cond2_trend_aligned": 0,
            "cond4_price_reaches_ce": 0,
            "cond5_rejection": 0,
            "all_required_pass": 0,
            "above_confidence": 0,
            "signals_generated": 0,
        }

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        """
        Check for FVG entry signal at M5 candle index `idx`.
        The DataFrame must have M15 features merged with prefix 'm15_'.
        """
        if idx < 10:
            return None

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]

        self.diag["candles_processed"] += 1

        # ─── Step 1: Update FVG tracker from M15 data ─────
        self._update_m15_fvgs(row)

        # ─── Step 2: Update fill status from M5 price action ──
        self.fvg_tracker.update_fills(row)

        # ─── Step 3: Check for entry signals ──────────────
        signal = self._check_bull_entry(df, idx, row, prev)
        if signal is not None:
            self.diag["signals_generated"] += 1
            return signal

        signal = self._check_bear_entry(df, idx, row, prev)
        if signal is not None:
            self.diag["signals_generated"] += 1
        return signal

    def print_diagnostics(self):
        """Print diagnostic counters to understand signal generation flow."""
        d = self.diag
        print(f"\n  {'STRATEGY B DIAGNOSTICS':}")
        print(f"  Candles processed:        {d['candles_processed']:>10,}")
        print(f"  FVG candidate found:      {d['fvg_found_for_candle']:>10,}")
        print(f"  Cond1 FVG valid:          {d['cond1_fvg_valid']:>10,}")
        print(f"  Cond2 Trend aligned:      {d['cond2_trend_aligned']:>10,}")
        print(f"  Cond4 Price reaches CE:   {d['cond4_price_reaches_ce']:>10,}")
        print(f"  Cond5 Rejection candle:   {d['cond5_rejection']:>10,}")
        print(f"  All required pass:        {d['all_required_pass']:>10,}")
        print(f"  Above confidence thresh:  {d['above_confidence']:>10,}")
        print(f"  Signals generated:        {d['signals_generated']:>10,}")

    def _update_m15_fvgs(self, m5_row: pd.Series):
        """
        Update FVG tracker with latest M15 data.
        Only processes when a new M15 candle appears (M15 time changes).
        """
        m15_time = m5_row.get("m15_datetime")
        if m15_time is None or pd.isna(m15_time):
            return

        # Robust comparison — convert to string to avoid Timestamp vs string mismatch
        m15_time_str = str(m15_time)
        if m15_time_str == self._last_m15_time:
            return  # Same M15 candle, already processed

        self._last_m15_time = m15_time_str

        # Build a pseudo M15 row for the tracker
        m15_data = {}
        for col in m5_row.index:
            if col.startswith("m15_"):
                clean_key = col[4:]  # Remove 'm15_' prefix
                m15_data[clean_key] = m5_row[col]

        if not m15_data:
            return

        m15_series = pd.Series(m15_data)
        # m15_idx might be float after asof merge, handle NaN
        raw_idx = m5_row.get("m15_idx", np.nan)
        if pd.isna(raw_idx):
            return
        m15_idx = int(raw_idx)
        self.fvg_tracker.update_from_m15(m15_series, m15_idx)

    def _check_bull_entry(self, df, idx, row, prev) -> Optional[TradeSignal]:
        """Check for bullish FVG entry (buy into a bullish FVG zone)."""
        current_price = row["close"]
        fvg = self.fvg_tracker.get_best_bull_fvg(current_price)

        if fvg is None:
            return None

        self.diag["fvg_found_for_candle"] += 1

        # ── Sub-Condition Evaluation ──
        sub_conditions = {}
        confidence = 0

        # Condition 1 (REQUIRED): FVG Present and valid
        cond1 = fvg.is_valid
        sub_conditions["fvg_present"] = cond1
        if cond1:
            self.diag["cond1_fvg_valid"] += 1

        # Condition 2 (REQUIRED): Trend Aligned
        ema50 = row.get("ema_50", None)
        ema50_slope = row.get("ema_50_slope", None)
        cond2 = False
        if ema50 is not None and not np.isnan(ema50):
            cond2 = (current_price > ema50) or (
                ema50_slope is not None and not np.isnan(ema50_slope) and ema50_slope > 0
            )
        sub_conditions["trend_aligned"] = cond2
        if cond1 and cond2:
            self.diag["cond2_trend_aligned"] += 1

        # Condition 4 (REQUIRED): Price reaches CE level
        cond4 = row["low"] <= fvg.ce
        sub_conditions["price_reaches_ce"] = cond4
        if cond1 and cond2 and cond4:
            self.diag["cond4_price_reaches_ce"] += 1

        # Condition 5 (REQUIRED): M5 Rejection candle
        # Current M5 candle shows rejection from FVG zone:
        # Option A: Bullish close while candle low was in/below FVG zone
        # Option B: Engulfing bull pattern while in zone
        # Option C: Pin bar bull while in zone
        # Note: prev candle check is a BONUS, not required
        candle_touched_zone = row["low"] <= fvg.top  # Current candle reached the zone

        cond5 = (
            candle_touched_zone and
            row["close"] > row["open"] and          # Bullish close
            row["close"] > fvg.bottom                # Closed above FVG far edge (not through it)
        )
        # Also accept engulfing or pin bar patterns
        if not cond5:
            cond5 = (
                row.get("is_engulfing_bull", 0) == 1 and
                candle_touched_zone and
                row["close"] > fvg.bottom
            )
        if not cond5:
            cond5 = (
                row.get("is_pin_bar_bull", 0) == 1 and
                candle_touched_zone and
                row["close"] > fvg.bottom
            )
        sub_conditions["m5_rejection"] = cond5
        if cond1 and cond2 and cond4 and cond5:
            self.diag["cond5_rejection"] += 1

        # ── Check all REQUIRED conditions ──
        if not (cond1 and cond2 and cond4 and cond5):
            return None

        self.diag["all_required_pass"] += 1

        # Base confidence = 60%
        confidence = 60

        # Condition 3 (WEIGHTED): First Touch
        # Track touch: if price entered zone, increment
        fvg.touch_count += 1
        if fvg.touch_count == 1:
            confidence += 20  # First touch bonus
            sub_conditions["first_touch"] = True
        elif fvg.touch_count == 2:
            confidence += 0   # Second touch, no bonus
            sub_conditions["first_touch"] = False
        else:
            return None       # Third touch, skip entirely

        # Condition 6 (BONUS): Not Stacked
        is_stacked = self.fvg_tracker.is_stacked(fvg)
        if not is_stacked:
            confidence += 10
        sub_conditions["not_stacked"] = not is_stacked

        # Condition 7 (BONUS): Not News FVG (placeholder — always True for now)
        is_news = fvg.is_news_fvg
        if not is_news:
            confidence += 10
        sub_conditions["not_news_fvg"] = not is_news

        # ── Build signal ──
        # entry_price = row["close"]  # Enter at current close (after rejection confirmed)
        entry_price = fvg.ce  # More aggressive: enter at CE level once touched and rejected
        sl_price = fvg.bottom - (SPREAD_SIMULATION_PIPS * PIP_VALUE * 2)  # Beyond far edge + buffer
        risk_distance = entry_price - sl_price

        # TP: 1:1 base, 1:2 if strong impulse
        atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
        if fvg.size_atr > 1.5:  # Large FVG = strong impulse
            tp_price = entry_price + risk_distance * 2.0  # 1:2 RR
        else:
            tp_price = entry_price + risk_distance * 1.0  # 1:1 RR

        # Safety: SL must be below entry
        if sl_price >= entry_price:
            return None

        if confidence >= CONFIDENCE_THRESHOLD:
            self.diag["above_confidence"] += 1

        return TradeSignal(
            datetime=row["datetime"],
            direction="buy",
            strategy_name=self.name,
            confidence=confidence,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_price=entry_price,
            sub_conditions=sub_conditions,
        )

    def _check_bear_entry(self, df, idx, row, prev) -> Optional[TradeSignal]:
        """Check for bearish FVG entry (sell into a bearish FVG zone)."""
        current_price = row["close"]
        fvg = self.fvg_tracker.get_best_bear_fvg(current_price)

        if fvg is None:
            return None

        self.diag["fvg_found_for_candle"] += 1

        # ── Sub-Condition Evaluation ──
        sub_conditions = {}
        confidence = 0

        # Condition 1 (REQUIRED): FVG Present and valid
        cond1 = fvg.is_valid
        sub_conditions["fvg_present"] = cond1
        if cond1:
            self.diag["cond1_fvg_valid"] += 1

        # Condition 2 (REQUIRED): Trend Aligned
        ema50 = row.get("ema_50", None)
        ema50_slope = row.get("ema_50_slope", None)
        cond2 = False
        if ema50 is not None and not np.isnan(ema50):
            cond2 = (current_price < ema50) or (
                ema50_slope is not None and not np.isnan(ema50_slope) and ema50_slope < 0
            )
        sub_conditions["trend_aligned"] = cond2
        if cond1 and cond2:
            self.diag["cond2_trend_aligned"] += 1

        # Condition 4 (REQUIRED): Price reaches CE level
        cond4 = row["high"] >= fvg.ce
        sub_conditions["price_reaches_ce"] = cond4
        if cond1 and cond2 and cond4:
            self.diag["cond4_price_reaches_ce"] += 1

        # Condition 5 (REQUIRED): M5 Rejection candle
        candle_touched_zone = row["high"] >= fvg.bottom

        cond5 = (
            candle_touched_zone and
            row["close"] < row["open"] and
            row["close"] < fvg.top
        )
        if not cond5:
            cond5 = (
                row.get("is_engulfing_bear", 0) == 1 and
                candle_touched_zone and
                row["close"] < fvg.top
            )
        if not cond5:
            cond5 = (
                row.get("is_pin_bar_bear", 0) == 1 and
                candle_touched_zone and
                row["close"] < fvg.top
            )
        sub_conditions["m5_rejection"] = cond5
        if cond1 and cond2 and cond4 and cond5:
            self.diag["cond5_rejection"] += 1

        # ── Check all REQUIRED conditions ──
        if not (cond1 and cond2 and cond4 and cond5):
            return None

        self.diag["all_required_pass"] += 1

        confidence = 60

        # Condition 3 (WEIGHTED): First Touch
        fvg.touch_count += 1
        if fvg.touch_count == 1:
            confidence += 20
            sub_conditions["first_touch"] = True
        elif fvg.touch_count == 2:
            confidence += 0
            sub_conditions["first_touch"] = False
        else:
            return None

        # Condition 6 (BONUS): Not Stacked
        is_stacked = self.fvg_tracker.is_stacked(fvg)
        if not is_stacked:
            confidence += 10
        sub_conditions["not_stacked"] = not is_stacked

        # Condition 7 (BONUS): Not News FVG
        is_news = fvg.is_news_fvg
        if not is_news:
            confidence += 10
        sub_conditions["not_news_fvg"] = not is_news

        # ── Build signal ──
        # entry_price = row["close"]
        entry_price = fvg.ce
        sl_price = fvg.top + (SPREAD_SIMULATION_PIPS * PIP_VALUE * 2)
        risk_distance = sl_price - entry_price

        if fvg.size_atr > 1.5:
            tp_price = entry_price - risk_distance * 2.0
        else:
            tp_price = entry_price - risk_distance * 1.0

        # Safety: SL must be above entry
        if sl_price <= entry_price:
            return None

        return TradeSignal(
            datetime=row["datetime"],
            direction="sell",
            strategy_name=self.name,
            confidence=confidence,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_price=entry_price,
            sub_conditions=sub_conditions,
        )