"""
Project MIDAS v2 — Strategy D: Trend Continuation (EMA Stack + RSI Pullback)
Phase 2 Implementation

Catches pullbacks within established trends on M5, using M15 for trend confirmation.

Sub-Conditions:
  1. EMA Alignment (required): M15 EMA(9) > EMA(21) > EMA(50) for buys
  2. EMA(200) Aligned (required): Price above EMA(200) on M15 for buys
  3. RSI Pullback (required): RSI(14) on M5 in 40-55 zone for buys (45-60 for sells)
  4. Price Touches EMA21 (required): M5 price within 1x ATR of EMA(21)
  5. Bounce Candle (required): M5 candle closes back in trend direction after touch

Confidence:
  All 5 required = base 60%
  + ADX > 25 (strong trend): +15%
  + First/second EMA21 touch (fresh pullback): +15%
  + Trend mature (>10 candles since last EMA cross): +10%
  Max = 100%

SL: Below EMA(50) on M5 + 2x spread. If >25 pips, use recent swing low + buffer.
TP: 1:1.5 RR base. Extend to 1:2.5 if ADX > 30 (strong trend).
"""

import sys
from typing import Optional, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import (
    ATR_PERIOD, SPREAD_SIMULATION_PIPS, PIP_VALUE,
    CONFIDENCE_THRESHOLD
)
from backtester import StrategyBase, TradeSignal


class StrategyD(StrategyBase):
    """
    Strategy D: Trend Continuation — EMA Stack + RSI Pullback.

    Uses M15 for trend direction (via merged m15_ columns)
    and M5 for pullback entry timing.

    Highest frequency strategy. Catches the "easy" moves.
    """

    name = "StrategyD_Trend"

    def __init__(self):
        self.diag = {
            "candles_processed": 0,
            "cond1_ema_alignment": 0,
            "cond2_ema200_aligned": 0,
            "cond3_rsi_pullback": 0,
            "cond4_ema21_touch": 0,
            "cond5_bounce_candle": 0,
            "all_required_pass": 0,
            "above_confidence": 0,
            "signals_generated": 0,
        }

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        """
        Check for trend continuation signal at M5 candle index `idx`.
        DataFrame must have M15 features merged with prefix 'm15_'.
        """
        if idx < 50:  # Need enough history for EMAs
            return None

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]

        self.diag["candles_processed"] += 1

        # Check both directions
        signal = self._check_bull_entry(df, idx, row, prev)
        if signal is not None:
            self.diag["signals_generated"] += 1
            return signal

        signal = self._check_bear_entry(df, idx, row, prev)
        if signal is not None:
            self.diag["signals_generated"] += 1
        return signal

    def _check_bull_entry(self, df, idx, row, prev) -> Optional[TradeSignal]:
        """Check for bullish trend continuation entry."""
        sub_conditions = {}
        atr = row.get(f"atr_{ATR_PERIOD}", np.nan)
        if np.isnan(atr) or atr <= 0:
            return None
        # Skip Asian session
        session = row.get("session", "off")
        if session == "asian":
            return None

        # ── Condition 1 (REQUIRED): EMA Alignment on M15 ──
        # M15 must show EMA(9) > EMA(21) > EMA(50)
        m15_ema_stack = row.get("m15_ema_stack", 0)
        m15_partial_bull = row.get("m15_ema_partial_bull", 0)

        # Accept full stack (9>21>50>200) or partial (9>21>50)
        cond1 = (m15_ema_stack == 1) or (m15_partial_bull == 1)
        sub_conditions["ema_alignment"] = cond1
        if cond1:
            self.diag["cond1_ema_alignment"] += 1

        # ── Condition 2 (REQUIRED): Price above EMA(200) on M15 ──
        m15_above_200 = row.get("m15_above_ema200", 0)
        cond2 = (m15_above_200 == 1)
        sub_conditions["ema200_aligned"] = cond2
        if cond2:
            self.diag["cond2_ema200_aligned"] += 1

        # ── Condition 3 (REQUIRED): RSI Pullback on M5 ──
        # RSI in 40-55 range (not oversold, just "resting")
        rsi = row.get(f"rsi_{ATR_PERIOD}", np.nan)
        cond3 = False
        if not np.isnan(rsi):
            cond3 = (30 <= rsi <= 50)
        sub_conditions["rsi_pullback"] = cond3
        if cond3:
            self.diag["cond3_rsi_pullback"] += 1

        # ── Condition 4 (REQUIRED): Price Touches EMA21 on M5 ──
        # Use the pre-computed feature from compute_trend()
        # cond4 = (row.get("ema21_touch_bull", 0) == 1)
        # # Fallback: manual check if feature not available
        # if not cond4 and "ema_21" in df.columns:
        #     ema21 = row.get("ema_21", np.nan)
        #     if not np.isnan(ema21):
        #         dist = abs(row["low"] - ema21)
        #         cond4 = (dist <= atr) and (row["close"] >= ema21)
        # sub_conditions["ema21_touch"] = cond4
        # if cond4:
        #     self.diag["cond4_ema21_touch"] += 1

        # ── Condition 5 (REQUIRED): Bounce Candle on M5 ──
        # cond5 = (row.get("bounce_candle_bull", 0) == 1)
        # # Fallback: manual check
        # if not cond5:
        #     prev_near_ema = False
        #     if "ema_21" in df.columns:
        #         prev_ema21 = prev.get("ema_21", np.nan)
        #         if not np.isnan(prev_ema21):
        #             prev_near_ema = abs(prev["low"] - prev_ema21) <= atr
        #     cond5 = (
        #         prev_near_ema and
        #         row["close"] > row["open"] and     # Bullish close
        #         row["close"] > row.get("ema_21", 0)  # Above EMA21
        #     )
        # sub_conditions["bounce_candle"] = cond5
        # if cond5:
        #     self.diag["cond5_bounce_candle"] += 1
        
        # ── Condition 4 (REQUIRED): Price Touches EMA21 OR EMA50 on M5 ──
        cond4 = False
        pullback_ema = "none"

        ema21 = row.get("ema_21", np.nan)
        ema50 = row.get("ema_50", np.nan)

        # Check EMA21 proximity
        if not np.isnan(ema21):
            near_ema21 = abs(row["low"] - ema21) <= atr and row["close"] >= ema21
            if near_ema21:
                cond4 = True
                pullback_ema = "ema21"

        # Check EMA50 proximity (deeper pullback, often higher conviction on gold)
        if not cond4 and not np.isnan(ema50):
            near_ema50 = abs(row["low"] - ema50) <= atr and row["close"] >= ema50
            if near_ema50:
                cond4 = True
                pullback_ema = "ema50"

        sub_conditions["ema21_touch"] = cond4
        sub_conditions["pullback_ema"] = pullback_ema
        if cond4:
            self.diag["cond4_ema21_touch"] += 1
        
        
        body = abs(row["close"] - row["open"])
        candle_range = row["high"] - row["low"]

        cond5 = False
        if candle_range > 0:
            prev_near_ema = False
            if "ema_21" in df.columns:
                prev_ema21 = prev.get("ema_21", np.nan)
                if not np.isnan(prev_ema21):
                    prev_near_ema = abs(prev["low"] - prev_ema21) <= atr
            cond5 = (
                prev_near_ema and
                row["close"] > row["open"] and
                body > 0.6 * candle_range and
                row["close"] > (row["high"] - 0.25 * candle_range) and
                row["close"] > row.get("ema_21", 0)
            )
        sub_conditions["bounce_candle"] = cond5
        if cond5:
            self.diag["cond5_bounce_candle"] += 1

        # ── Condition 6 (REQUIRED): Trend maturity ──
        candles_since_cross = row.get("candles_since_ema_cross", 0)
        cond6 = (candles_since_cross > 10)
        sub_conditions["mature_trend"] = cond6
        
        # ── Check all REQUIRED conditions ──
        if not (cond1 and cond2 and cond3 and cond4 and cond5 and cond6):
            return None

        self.diag["all_required_pass"] += 1

        # ── Confidence Scoring ──
        confidence = 60  # Base when all required met

        # ADX > 25 (strong trend) = +15%
        adx = row.get(f"adx_{ATR_PERIOD}", 0)
        strong_trend = (not np.isnan(adx)) and (adx > 20)  # Use 20 for buys, 25 for sells to be slightly more lenient on bullish signals
        if strong_trend:
            confidence += 15
        sub_conditions["adx_strong"] = strong_trend

        # Fresh pullback (1st or 2nd EMA21 touch) = +15%
        touch_count = row.get("ema21_touch_count", 0)
        fresh_pullback = (touch_count <= 2)
        if fresh_pullback:
            confidence += 15
        sub_conditions["fresh_pullback"] = fresh_pullback

        # Trend maturity (>10 candles since last EMA cross) = +10%
        # candles_since_cross = row.get("candles_since_ema_cross", 0)
        # mature_trend = (candles_since_cross > 10)
        # if mature_trend:
        #     confidence += 10
        # sub_conditions["mature_trend"] = mature_trend

        # EMA angle strength (strong upward slope) = +10%
        ema_angle = row.get("ema_21_angle", 0)
        strong_angle = (not np.isnan(ema_angle)) and (ema_angle > 15)
        if strong_angle:
            confidence += 10
        sub_conditions["strong_ema_angle"] = strong_angle

        if confidence < CONFIDENCE_THRESHOLD:
            return None

        self.diag["above_confidence"] += 1

        # ── SL Calculation ──
        ema50 = row.get("ema_50", np.nan)
        spread_buffer = SPREAD_SIMULATION_PIPS * PIP_VALUE * 2

        if not np.isnan(ema50):
            sl_price = ema50 - spread_buffer
        else:
            sl_price = row["close"] - 2 * atr
        # Minimum SL = 1x ATR (prevents noise stops)
        min_sl = row["close"] - atr
        if sl_price > min_sl:
            sl_price = min_sl
        # Cap: if SL is too wide (>25 pips), use recent swing low instead
        sl_distance_pips = (row["close"] - sl_price) / PIP_VALUE
        if sl_distance_pips > 25:
            swing_low = row.get("last_swing_low", np.nan)
            if not np.isnan(swing_low) and swing_low < row["close"]:
                sl_price = swing_low - spread_buffer
            else:
                # Hard cap at 25 pips
                sl_price = row["close"] - (25 * PIP_VALUE)

        # Safety: SL must be below entry
        if sl_price >= row["close"]:
            return None

        # ── TP Calculation ──
        risk_distance = row["close"] - sl_price

        # Base: 1:1.5 RR. Extend to 1:2.5 if ADX > 30
        very_strong = (not np.isnan(adx)) and (adx > 30)
        if very_strong:
            tp_price = row["close"] + risk_distance * 2.5
        else:
            tp_price = row["close"] + risk_distance * 1.5

        return TradeSignal(
            datetime=row["datetime"],
            direction="buy",
            strategy_name=self.name,
            confidence=confidence,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_price=row["close"],
            sub_conditions=sub_conditions,
        )

    def _check_bear_entry(self, df, idx, row, prev) -> Optional[TradeSignal]:
        """Check for bearish trend continuation entry."""
        sub_conditions = {}
        atr = row.get(f"atr_{ATR_PERIOD}", np.nan)
        if np.isnan(atr) or atr <= 0:
            return None
        # Skip Asian session
        session = row.get("session", "off")
        if session == "asian":
            return None

        # ── Condition 1 (REQUIRED): EMA Alignment on M15 ──
        m15_ema_stack = row.get("m15_ema_stack", 0)
        m15_partial_bear = row.get("m15_ema_partial_bear", 0)

        cond1 = (m15_ema_stack == -1) or (m15_partial_bear == 1)
        sub_conditions["ema_alignment"] = cond1
        if cond1:
            self.diag["cond1_ema_alignment"] += 1

        # ── Condition 2 (REQUIRED): Price below EMA(200) on M15 ──
        m15_above_200 = row.get("m15_above_ema200", 0)
        cond2 = (m15_above_200 == 0)
        sub_conditions["ema200_aligned"] = cond2
        if cond2:
            self.diag["cond2_ema200_aligned"] += 1

        # ── Condition 3 (REQUIRED): RSI Pullback on M5 ──
        rsi = row.get(f"rsi_{ATR_PERIOD}", np.nan)
        cond3 = False
        if not np.isnan(rsi):
            cond3 = (50 <= rsi <= 70)
        sub_conditions["rsi_pullback"] = cond3
        if cond3:
            self.diag["cond3_rsi_pullback"] += 1

        # ── Condition 4 (REQUIRED): Price Touches EMA21 on M5 ──
        # cond4 = (row.get("ema21_touch_bear", 0) == 1)
        # if not cond4 and "ema_21" in df.columns:
        #     ema21 = row.get("ema_21", np.nan)
        #     if not np.isnan(ema21):
        #         dist = abs(row["high"] - ema21)
        #         cond4 = (dist <= atr) and (row["close"] <= ema21)
        # sub_conditions["ema21_touch"] = cond4
        # if cond4:
        #     self.diag["cond4_ema21_touch"] += 1

        # ── Condition 5 (REQUIRED): Bounce Candle on M5 ──
        # cond5 = (row.get("bounce_candle_bear", 0) == 1)
        # if not cond5:
        #     prev_near_ema = False
        #     if "ema_21" in df.columns:
        #         prev_ema21 = prev.get("ema_21", np.nan)
        #         if not np.isnan(prev_ema21):
        #             prev_near_ema = abs(prev["high"] - prev_ema21) <= atr
        #     cond5 = (
        #         prev_near_ema and
        #         row["close"] < row["open"] and
        #         row["close"] < row.get("ema_21", float("inf"))
        #     )
        # sub_conditions["bounce_candle"] = cond5
        # if cond5:
        #     self.diag["cond5_bounce_candle"] += 1
        
        # ── Condition 4 (REQUIRED): Price Touches EMA21 OR EMA50 on M5 ──
        cond4 = False
        pullback_ema = "none"

        ema21 = row.get("ema_21", np.nan)
        ema50 = row.get("ema_50", np.nan)

        # Check EMA21 proximity
        if not np.isnan(ema21):
            near_ema21 = abs(row["high"] - ema21) <= atr and row["close"] <= ema21
            if near_ema21:
                cond4 = True
                pullback_ema = "ema21"

        # Check EMA50 proximity
        if not cond4 and not np.isnan(ema50):
            near_ema50 = abs(row["high"] - ema50) <= atr and row["close"] <= ema50
            if near_ema50:
                cond4 = True
                pullback_ema = "ema50"

        sub_conditions["ema21_touch"] = cond4
        sub_conditions["pullback_ema"] = pullback_ema
        if cond4:
            self.diag["cond4_ema21_touch"] += 1
        
        body = abs(row["close"] - row["open"])
        candle_range = row["high"] - row["low"]

        cond5 = False
        if candle_range > 0:
            prev_near_ema = False
            if "ema_21" in df.columns:
                prev_ema21 = prev.get("ema_21", np.nan)
                if not np.isnan(prev_ema21):
                    prev_near_ema = abs(prev["high"] - prev_ema21) <= atr
            cond5 = (
                prev_near_ema and
                row["close"] < row["open"] and
                body > 0.6 * candle_range and
                row["close"] < (row["low"] + 0.25 * candle_range) and
                row["close"] < row.get("ema_21", float("inf"))
            )
        sub_conditions["bounce_candle"] = cond5
        if cond5:
            self.diag["cond5_bounce_candle"] += 1
        
        # ── Condition 6 (REQUIRED): Trend maturity ──
        candles_since_cross = row.get("candles_since_ema_cross", 0)
        cond6 = (candles_since_cross > 10)
        sub_conditions["mature_trend"] = cond6
        
        # ── Check all REQUIRED conditions ──
        if not (cond1 and cond2 and cond3 and cond4 and cond5 and cond6):
            return None

        self.diag["all_required_pass"] += 1

        # ── Confidence Scoring ──
        confidence = 60

        adx = row.get(f"adx_{ATR_PERIOD}", 0)
        strong_trend = (not np.isnan(adx)) and (adx > 20)  # Use 20 for buys, 25 for sells to be slightly more lenient on bullish signals 
        if strong_trend:
            confidence += 15
        sub_conditions["adx_strong"] = strong_trend

        touch_count = row.get("ema21_touch_count", 0)
        fresh_pullback = (touch_count <= 2)
        if fresh_pullback:
            confidence += 15
        sub_conditions["fresh_pullback"] = fresh_pullback

        # candles_since_cross = row.get("candles_since_ema_cross", 0)
        # mature_trend = (candles_since_cross > 10)
        # if mature_trend:
        #     confidence += 10
        # sub_conditions["mature_trend"] = mature_trend
        
        # EMA angle strength (strong downward slope) = +10%
        ema_angle = row.get("ema_21_angle", 0)
        strong_angle = (not np.isnan(ema_angle)) and (ema_angle < -15)
        if strong_angle:
            confidence += 10
        sub_conditions["strong_ema_angle"] = strong_angle
        
        if confidence < CONFIDENCE_THRESHOLD:
            return None

        self.diag["above_confidence"] += 1

        # ── SL Calculation ──
        ema50 = row.get("ema_50", np.nan)
        spread_buffer = SPREAD_SIMULATION_PIPS * PIP_VALUE * 2

        if not np.isnan(ema50):
            sl_price = ema50 + spread_buffer
        else:
            sl_price = row["close"] + 2 * atr
        # Minimum SL = 1x ATR (prevents noise stops)
        min_sl = row["close"] + atr
        if sl_price < min_sl:
            sl_price = min_sl
        # Cap: if SL is too wide (>25 pips), use recent swing high
        sl_distance_pips = (sl_price - row["close"]) / PIP_VALUE
        if sl_distance_pips > 25:
            swing_high = row.get("last_swing_high", np.nan)
            if not np.isnan(swing_high) and swing_high > row["close"]:
                sl_price = swing_high + spread_buffer
            else:
                sl_price = row["close"] + (25 * PIP_VALUE)

        # Safety: SL must be above entry
        if sl_price <= row["close"]:
            return None

        # ── TP Calculation ──
        risk_distance = sl_price - row["close"]

        very_strong = (not np.isnan(adx)) and (adx > 30)
        if very_strong:
            tp_price = row["close"] - risk_distance * 2.5
        else:
            tp_price = row["close"] - risk_distance * 1.5

        return TradeSignal(
            datetime=row["datetime"],
            direction="sell",
            strategy_name=self.name,
            confidence=confidence,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_price=row["close"],
            sub_conditions=sub_conditions,
        )

    def print_diagnostics(self):
        """Print diagnostic counters."""
        d = self.diag
        print(f"\n  STRATEGY D DIAGNOSTICS:")
        print(f"  Candles processed:        {d['candles_processed']:>10,}")
        print(f"  Cond1 EMA alignment:      {d['cond1_ema_alignment']:>10,}")
        print(f"  Cond2 EMA200 aligned:     {d['cond2_ema200_aligned']:>10,}")
        print(f"  Cond3 RSI pullback:       {d['cond3_rsi_pullback']:>10,}")
        print(f"  Cond4 EMA21 touch:        {d['cond4_ema21_touch']:>10,}")
        print(f"  Cond5 Bounce candle:      {d['cond5_bounce_candle']:>10,}")
        print(f"  All required pass:        {d['all_required_pass']:>10,}")
        print(f"  Above confidence:         {d['above_confidence']:>10,}")
        print(f"  Signals generated:        {d['signals_generated']:>10,}")