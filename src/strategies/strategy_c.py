# """
# Project MIDAS v2 — Strategy C: Session Range Breakout
# Phase 3 Implementation

# Marks Asian session range, enters on London/NY breakout + retest.

# Sub-Conditions (Required):
#   1. Asian Range Formed: valid range from 02:00-08:00 broker time (UTC+2)
#   2. Range is Tight: width between 0.3x and 1.5x daily ATR
#   3. Session Active: 08:00-18:00 broker time, not Friday
#   4. Clean Breakout: M5 close beyond range high/low
#   5. Retest: price pulls back within 5 pips of broken edge, holds

# Confidence Bonuses:
#   Base 60% (all required met)
#   + Early breakout (before 14:00 broker): +15%
#   + No prior failed breakout opposite side: +10%
#   + Tight range (< 0.75x ATR): +15%
#   Max = 100%

# SL: Opposite side of range (tight) or middle (wide). +2x spread.
# TP: Range width (1:1). Extend to 1.5x if London + ATR expanding.
# """

# import sys
# from dataclasses import dataclass
# from typing import Optional, Dict

# import numpy as np
# import pandas as pd

# sys.path.insert(0, ".")
# from config import (
#     ATR_PERIOD, SPREAD_SIMULATION_PIPS, PIP_VALUE,
#     CONFIDENCE_THRESHOLD
# )
# from backtester import StrategyBase, TradeSignal


# @dataclass
# class DailyBreakoutState:
#     """Tracks breakout state for a single trading day."""
#     date: str
#     range_high: float = np.nan
#     range_low: float = np.nan
#     range_width: float = np.nan
#     range_valid: bool = False

#     bull_breakout_detected: bool = False
#     bull_breakout_failed: bool = False
#     bull_retest_done: bool = False
#     bull_signal_given: bool = False

#     bear_breakout_detected: bool = False
#     bear_breakout_failed: bool = False
#     bear_retest_done: bool = False
#     bear_signal_given: bool = False


# class StrategyC(StrategyBase):
#     """
#     Strategy C: Session Range Breakout.
#     Detects Asian session range, enters on London/NY breakout + retest.
#     """

#     name = "StrategyC_SessionBreakout"

#     ASIAN_END = 8
#     BREAKOUT_START = 8
#     BREAKOUT_END = 18
#     EARLY_CUTOFF = 14
#     RETEST_ATR_MULT = 0.5   # Retest within 0.5x M5 ATR of broken edge
#     FAILURE_ATR_MULT = 1.5  # Failed if close goes 1.5x ATR past edge back into range

#     def __init__(self):
#         self.daily_states: Dict[str, DailyBreakoutState] = {}
#         self.diag = {
#             "candles_processed": 0,
#             "range_valid": 0,
#             "in_breakout_window": 0,
#             "not_friday": 0,
#             "breakout_detected": 0,
#             "retest_detected": 0,
#             "failed_breakouts": 0,
#             "all_required_pass": 0,
#             "above_confidence": 0,
#             "signals_generated": 0,
#         }

#     def _get_daily_state(self, row) -> DailyBreakoutState:
#         """Get or create daily state."""
#         dt = row["datetime"]
#         if hasattr(dt, "date"):
#             date_str = str(dt.date())
#         else:
#             date_str = str(pd.to_datetime(dt).date())

#         if date_str not in self.daily_states:
#             state = DailyBreakoutState(date=date_str)
#             rh = row.get("asian_range_high", np.nan)
#             rl = row.get("asian_range_low", np.nan)
#             rv = row.get("asian_range_valid", 0)

#             if not np.isnan(rh) and not np.isnan(rl):
#                 state.range_high = rh
#                 state.range_low = rl
#                 state.range_width = rh - rl
#                 state.range_valid = (rv == 1)

#             self.daily_states[date_str] = state

#         state = self.daily_states[date_str]

#         # Update range if just became available
#         if not state.range_valid:
#             rh = row.get("asian_range_high", np.nan)
#             rl = row.get("asian_range_low", np.nan)
#             rv = row.get("asian_range_valid", 0)
#             if not np.isnan(rh) and not np.isnan(rl) and rv == 1:
#                 state.range_high = rh
#                 state.range_low = rl
#                 state.range_width = rh - rl
#                 state.range_valid = True

#         return state

#     def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
#         if idx < 10:
#             return None

#         row = df.iloc[idx]
#         self.diag["candles_processed"] += 1

#         dt = row["datetime"]
#         if hasattr(dt, "hour"):
#             hour = dt.hour
#             dow = dt.dayofweek
#         else:
#             dt = pd.to_datetime(dt)
#             hour = dt.hour
#             dow = dt.dayofweek

#         # Skip Asian session
#         if hour < self.ASIAN_END:
#             return None

#         state = self._get_daily_state(row)

#         if not state.range_valid:
#             return None
#         self.diag["range_valid"] += 1

#         if hour < self.BREAKOUT_START or hour >= self.BREAKOUT_END:
#             return None
#         self.diag["in_breakout_window"] += 1

#         # Skip Friday
#         if dow == 4:
#             return None
#         self.diag["not_friday"] += 1

#         # Update breakout/retest/failure state
#         self._update_state(state, row)

#         signal = self._check_bull_entry(df, idx, row, state, hour)
#         if signal is not None:
#             return signal

#         return self._check_bear_entry(df, idx, row, state, hour)

#     def _update_state(self, state: DailyBreakoutState, row):
#         """Track breakouts, retests, and failures."""
#         close = row["close"]
#         high = row["high"]
#         low = row["low"]
#         atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
#         if np.isnan(atr) or atr <= 0:
#             atr = 2.0

#         retest_dist = self.RETEST_ATR_MULT * atr
#         failure_dist = self.FAILURE_ATR_MULT * atr

#         # ── Bullish ──
#         if not state.bull_breakout_detected:
#             if close > state.range_high:
#                 state.bull_breakout_detected = True
#                 self.diag["breakout_detected"] += 1
#         else:
#             if not state.bull_breakout_failed and not state.bull_retest_done:
#                 # Failure: close goes significantly below range high (deep back into range)
#                 if close < state.range_high - failure_dist:
#                     state.bull_breakout_failed = True
#                     self.diag["failed_breakouts"] += 1

#                 # Retest: price comes back near range high but doesn't fail
#                 # Low touches near range_high (within retest_dist) AND close stays near/above
#                 elif low <= state.range_high + retest_dist and close >= state.range_high - retest_dist:
#                     state.bull_retest_done = True
#                     self.diag["retest_detected"] += 1

#         # ── Bearish ──
#         if not state.bear_breakout_detected:
#             if close < state.range_low:
#                 state.bear_breakout_detected = True
#                 self.diag["breakout_detected"] += 1
#         else:
#             if not state.bear_breakout_failed and not state.bear_retest_done:
#                 # Failure: close goes significantly above range low (deep back into range)
#                 if close > state.range_low + failure_dist:
#                     state.bear_breakout_failed = True
#                     self.diag["failed_breakouts"] += 1

#                 # Retest: price comes back near range low but doesn't fail
#                 elif high >= state.range_low - retest_dist and close <= state.range_low + retest_dist:
#                     state.bear_retest_done = True
#                     self.diag["retest_detected"] += 1

#     def _check_bull_entry(self, df, idx, row, state, hour) -> Optional[TradeSignal]:
#         if state.bull_signal_given:
#             return None

#         cond4 = state.bull_breakout_detected and not state.bull_breakout_failed
#         cond5 = state.bull_retest_done

#         if not (cond4 and cond5):
#             return None

#         self.diag["all_required_pass"] += 1
#         sub_conditions = {
#             "range_formed": True, "range_valid": True,
#             "session_active": True, "breakout_detected": True,
#             "retest_done": True,
#         }

#         # ── Confidence ──
#         confidence = 60

#         is_early = (hour < self.EARLY_CUTOFF)
#         if is_early:
#             confidence += 15
#         sub_conditions["early_breakout"] = is_early

#         no_opposite_fail = not state.bear_breakout_failed
#         if no_opposite_fail:
#             confidence += 10
#         sub_conditions["no_opposite_fail"] = no_opposite_fail

#         range_width_atr = row.get("asian_range_width_atr", 1.0)
#         if np.isnan(range_width_atr):
#             range_width_atr = 1.0
#         tight_range = (range_width_atr < 0.75)
#         if tight_range:
#             confidence += 15
#         sub_conditions["tight_range"] = tight_range

#         if confidence < CONFIDENCE_THRESHOLD:
#             return None
#         self.diag["above_confidence"] += 1

#         # ── SL ──
#         spread_buffer = SPREAD_SIMULATION_PIPS * PIP_VALUE * 2

#         if tight_range:
#             sl_price = state.range_low - spread_buffer
#         else:
#             sl_price = ((state.range_high + state.range_low) / 2) - spread_buffer

#         if sl_price >= row["close"]:
#             return None

#         # ── TP ──
#         risk_distance = row["close"] - sl_price

#         atr = row.get(f"atr_{ATR_PERIOD}", np.nan)
#         atr_expanding = False
#         if not np.isnan(atr):
#             prev_atr = df.iloc[max(0, idx - 12)].get(f"atr_{ATR_PERIOD}", atr)
#             atr_expanding = atr > prev_atr * 1.1

#         if is_early and atr_expanding:
#             tp_price = row["close"] + state.range_width * 1.5
#         else:
#             tp_price = row["close"] + state.range_width

#         # Minimum 1:1 RR
#         if (tp_price - row["close"]) < risk_distance:
#             tp_price = row["close"] + risk_distance

#         state.bull_signal_given = True
#         self.diag["signals_generated"] += 1

#         return TradeSignal(
#             datetime=row["datetime"],
#             direction="buy",
#             strategy_name=self.name,
#             confidence=confidence,
#             sl_price=sl_price,
#             tp_price=tp_price,
#             entry_price=row["close"],
#             sub_conditions=sub_conditions,
#         )

#     def _check_bear_entry(self, df, idx, row, state, hour) -> Optional[TradeSignal]:
#         if state.bear_signal_given:
#             return None

#         cond4 = state.bear_breakout_detected and not state.bear_breakout_failed
#         cond5 = state.bear_retest_done

#         if not (cond4 and cond5):
#             return None

#         self.diag["all_required_pass"] += 1
#         sub_conditions = {
#             "range_formed": True, "range_valid": True,
#             "session_active": True, "breakout_detected": True,
#             "retest_done": True,
#         }

#         confidence = 60

#         is_early = (hour < self.EARLY_CUTOFF)
#         if is_early:
#             confidence += 15
#         sub_conditions["early_breakout"] = is_early

#         no_opposite_fail = not state.bull_breakout_failed
#         if no_opposite_fail:
#             confidence += 10
#         sub_conditions["no_opposite_fail"] = no_opposite_fail

#         range_width_atr = row.get("asian_range_width_atr", 1.0)
#         if np.isnan(range_width_atr):
#             range_width_atr = 1.0
#         tight_range = (range_width_atr < 0.75)
#         if tight_range:
#             confidence += 15
#         sub_conditions["tight_range"] = tight_range

#         if confidence < CONFIDENCE_THRESHOLD:
#             return None
#         self.diag["above_confidence"] += 1

#         spread_buffer = SPREAD_SIMULATION_PIPS * PIP_VALUE * 2

#         if tight_range:
#             sl_price = state.range_high + spread_buffer
#         else:
#             sl_price = ((state.range_high + state.range_low) / 2) + spread_buffer

#         if sl_price <= row["close"]:
#             return None

#         risk_distance = sl_price - row["close"]

#         atr = row.get(f"atr_{ATR_PERIOD}", np.nan)
#         atr_expanding = False
#         if not np.isnan(atr):
#             prev_atr = df.iloc[max(0, idx - 12)].get(f"atr_{ATR_PERIOD}", atr)
#             atr_expanding = atr > prev_atr * 1.1

#         if is_early and atr_expanding:
#             tp_price = row["close"] - state.range_width * 1.5
#         else:
#             tp_price = row["close"] - state.range_width

#         if (row["close"] - tp_price) < risk_distance:
#             tp_price = row["close"] - risk_distance

#         state.bear_signal_given = True
#         self.diag["signals_generated"] += 1

#         return TradeSignal(
#             datetime=row["datetime"],
#             direction="sell",
#             strategy_name=self.name,
#             confidence=confidence,
#             sl_price=sl_price,
#             tp_price=tp_price,
#             entry_price=row["close"],
#             sub_conditions=sub_conditions,
#         )

#     def print_diagnostics(self):
#         d = self.diag
#         print(f"\n  STRATEGY C DIAGNOSTICS:")
#         print(f"  Candles processed:        {d['candles_processed']:>10,}")
#         print(f"  Range valid:              {d['range_valid']:>10,}")
#         print(f"  In breakout window:       {d['in_breakout_window']:>10,}")
#         print(f"  Not Friday:               {d['not_friday']:>10,}")
#         print(f"  Breakouts detected:       {d['breakout_detected']:>10,}")
#         print(f"  Retests detected:         {d['retest_detected']:>10,}")
#         print(f"  Failed breakouts:         {d['failed_breakouts']:>10,}")
#         print(f"  All required pass:        {d['all_required_pass']:>10,}")
#         print(f"  Above confidence:         {d['above_confidence']:>10,}")
#         print(f"  Signals generated:        {d['signals_generated']:>10,}")

#         total_days = len(self.daily_states)
#         valid_days = sum(1 for s in self.daily_states.values() if s.range_valid)
#         print(f"\n  DAILY SUMMARY:")
#         print(f"  Total days:               {total_days:>10,}")
#         print(f"  Valid range days:         {valid_days:>10,}")
#         print(f"  Bull breakout days:       {sum(1 for s in self.daily_states.values() if s.bull_breakout_detected):>10,}")
#         print(f"  Bear breakout days:       {sum(1 for s in self.daily_states.values() if s.bear_breakout_detected):>10,}")
#         print(f"  Bull retest days:         {sum(1 for s in self.daily_states.values() if s.bull_retest_done):>10,}")
#         print(f"  Bear retest days:         {sum(1 for s in self.daily_states.values() if s.bear_retest_done):>10,}")
#         print(f"  Bull failed days:         {sum(1 for s in self.daily_states.values() if s.bull_breakout_failed):>10,}")
#         print(f"  Bear failed days:         {sum(1 for s in self.daily_states.values() if s.bear_breakout_failed):>10,}")

"""
Project MIDAS v2 — Strategy C: Session Range Breakout
Phase 3 Implementation

Marks Asian session range, enters on London/NY breakout + retest.

Sub-Conditions (Required):
  1. Asian Range Formed: valid range from 02:00-08:00 broker time (UTC+2)
  2. Range is Tight: width between 0.3x and 1.5x daily ATR
  3. Session Active: 08:00-18:00 broker time, not Friday
  4. Clean Breakout: M5 close beyond range high/low
  5. Retest: price pulls back within 5 pips of broken edge, holds

Confidence Bonuses:
  Base 60% (all required met)
  + Early breakout (before 14:00 broker): +15%
  + No prior failed breakout opposite side: +10%
  + Tight range (< 0.75x ATR): +15%
  Max = 100%

SL: Opposite side of range (tight) or middle (wide). +2x spread.
TP: Range width (1:1). Extend to 1.5x if London + ATR expanding.
"""

import sys
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import (
    ATR_PERIOD, SPREAD_SIMULATION_PIPS, PIP_VALUE,
    CONFIDENCE_THRESHOLD
)
from backtester import StrategyBase, TradeSignal


@dataclass
class DailyBreakoutState:
    """Tracks breakout state for a single trading day."""
    date: str
    range_high: float = np.nan
    range_low: float = np.nan
    range_width: float = np.nan
    range_valid: bool = False

    bull_breakout_detected: bool = False
    bull_breakout_failed: bool = False
    bull_retest_done: bool = False
    bull_signal_given: bool = False
    bull_max_extension: float = 0.0  # How far price moved above range_high after breakout

    bear_breakout_detected: bool = False
    bear_breakout_failed: bool = False
    bear_retest_done: bool = False
    bear_signal_given: bool = False
    bear_max_extension: float = 0.0  # How far price moved below range_low after breakout


class StrategyC(StrategyBase):
    """
    Strategy C: Session Range Breakout.
    Detects Asian session range, enters on London/NY breakout + retest.
    """

    name = "StrategyC_SessionBreakout"

    ASIAN_END = 8
    BREAKOUT_START = 8
    BREAKOUT_END = 18
    EARLY_CUTOFF = 14
    RETEST_ATR_MULT = 0.5   # Retest within 0.5x M5 ATR of broken edge
    FAILURE_ATR_MULT = 1.5  # Failed if close goes 1.5x ATR past edge back into range

    def __init__(self):
        self.daily_states: Dict[str, DailyBreakoutState] = {}
        self.diag = {
            "candles_processed": 0,
            "range_valid": 0,
            "in_breakout_window": 0,
            "not_friday": 0,
            "breakout_detected": 0,
            "retest_detected": 0,
            "failed_breakouts": 0,
            "all_required_pass": 0,
            "above_confidence": 0,
            "signals_generated": 0,
        }

    def _get_daily_state(self, row) -> DailyBreakoutState:
        """Get or create daily state."""
        dt = row["datetime"]
        if hasattr(dt, "date"):
            date_str = str(dt.date())
        else:
            date_str = str(pd.to_datetime(dt).date())

        if date_str not in self.daily_states:
            state = DailyBreakoutState(date=date_str)
            rh = row.get("asian_range_high", np.nan)
            rl = row.get("asian_range_low", np.nan)
            rv = row.get("asian_range_valid", 0)

            if not np.isnan(rh) and not np.isnan(rl):
                state.range_high = rh
                state.range_low = rl
                state.range_width = rh - rl
                state.range_valid = (rv == 1)

            self.daily_states[date_str] = state

        state = self.daily_states[date_str]

        # Update range if just became available
        if not state.range_valid:
            rh = row.get("asian_range_high", np.nan)
            rl = row.get("asian_range_low", np.nan)
            rv = row.get("asian_range_valid", 0)
            if not np.isnan(rh) and not np.isnan(rl) and rv == 1:
                state.range_high = rh
                state.range_low = rl
                state.range_width = rh - rl
                state.range_valid = True

        return state

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        if idx < 10:
            return None

        row = df.iloc[idx]
        self.diag["candles_processed"] += 1

        dt = row["datetime"]
        if hasattr(dt, "hour"):
            hour = dt.hour
            dow = dt.dayofweek
        else:
            dt = pd.to_datetime(dt)
            hour = dt.hour
            dow = dt.dayofweek

        # Skip Asian session
        if hour < self.ASIAN_END:
            return None

        state = self._get_daily_state(row)

        if not state.range_valid:
            return None
        self.diag["range_valid"] += 1

        if hour < self.BREAKOUT_START or hour >= self.BREAKOUT_END:
            return None
        self.diag["in_breakout_window"] += 1

        # Skip Friday
        if dow == 4:
            return None
        self.diag["not_friday"] += 1

        # Update breakout/retest/failure state
        self._update_state(state, row)

        signal = self._check_bull_entry(df, idx, row, state, hour)
        if signal is not None:
            return signal

        return self._check_bear_entry(df, idx, row, state, hour)

    def _update_state(self, state: DailyBreakoutState, row):
        """Track breakouts, retests, and failures."""
        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
        if np.isnan(atr) or atr <= 0:
            atr = 2.0

        retest_dist = self.RETEST_ATR_MULT * atr
        failure_dist = self.FAILURE_ATR_MULT * atr
        min_extension = 0.5 * atr  # Must move at least 0.5x ATR away before retest counts

        # ── Bullish ──
        if not state.bull_breakout_detected:
            if close > state.range_high:
                state.bull_breakout_detected = True
                state.bull_max_extension = high - state.range_high
                self.diag["breakout_detected"] += 1
        else:
            # Track how far price has gone above range_high
            current_extension = high - state.range_high
            if current_extension > state.bull_max_extension:
                state.bull_max_extension = current_extension

            if not state.bull_breakout_failed and not state.bull_retest_done:
                # Failure: close goes significantly below range high (deep back into range)
                if close < state.range_high - failure_dist:
                    state.bull_breakout_failed = True
                    self.diag["failed_breakouts"] += 1

                # Retest: price must have first moved away, THEN come back
                elif (state.bull_max_extension >= min_extension and
                      low <= state.range_high + retest_dist and
                      close >= state.range_high - retest_dist):
                    state.bull_retest_done = True
                    self.diag["retest_detected"] += 1

        # ── Bearish ──
        if not state.bear_breakout_detected:
            if close < state.range_low:
                state.bear_breakout_detected = True
                state.bear_max_extension = state.range_low - low
                self.diag["breakout_detected"] += 1
        else:
            # Track how far price has gone below range_low
            current_extension = state.range_low - low
            if current_extension > state.bear_max_extension:
                state.bear_max_extension = current_extension

            if not state.bear_breakout_failed and not state.bear_retest_done:
                # Failure: close goes significantly above range low (deep back into range)
                if close > state.range_low + failure_dist:
                    state.bear_breakout_failed = True
                    self.diag["failed_breakouts"] += 1

                # Retest: price must have first moved away, THEN come back
                elif (state.bear_max_extension >= min_extension and
                      high >= state.range_low - retest_dist and
                      close <= state.range_low + retest_dist):
                    state.bear_retest_done = True
                    self.diag["retest_detected"] += 1

    def _check_bull_entry(self, df, idx, row, state, hour) -> Optional[TradeSignal]:
        if state.bull_signal_given:
            return None

        cond4 = state.bull_breakout_detected and not state.bull_breakout_failed
        cond5 = state.bull_retest_done

        if not (cond4 and cond5):
            return None
        # Trend filter: only take bullish breakout if M15 trend is up
        # Trend filter: skip strongly counter-trend breakouts
        m15_ema_slope = row.get("m15_ema_50_slope", np.nan)
        if not np.isnan(m15_ema_slope) and m15_ema_slope < -0.5:
            return None
        
        self.diag["all_required_pass"] += 1
        sub_conditions = {
            "range_formed": True, "range_valid": True,
            "session_active": True, "breakout_detected": True,
            "retest_done": True,
        }

        # ── Confidence ──
        confidence = 60

        is_early = (hour < self.EARLY_CUTOFF)
        if is_early:
            confidence += 15
        sub_conditions["early_breakout"] = is_early

        no_opposite_fail = not state.bear_breakout_failed
        if no_opposite_fail:
            confidence += 10
        sub_conditions["no_opposite_fail"] = no_opposite_fail

        range_width_atr = row.get("asian_range_width_atr", 1.0)
        if np.isnan(range_width_atr):
            range_width_atr = 1.0
        tight_range = (range_width_atr < 0.75)
        if tight_range:
            confidence += 15
        sub_conditions["tight_range"] = tight_range

        if confidence < CONFIDENCE_THRESHOLD:
            return None
        self.diag["above_confidence"] += 1

        # ── SL ──
        spread_buffer = SPREAD_SIMULATION_PIPS * PIP_VALUE * 2

        if tight_range:
            sl_price = state.range_low - spread_buffer
        else:
            sl_price = ((state.range_high + state.range_low) / 2) - spread_buffer

        if sl_price >= row["close"]:
            return None

        # ── TP ──
        risk_distance = row["close"] - sl_price

        atr = row.get(f"atr_{ATR_PERIOD}", np.nan)
        atr_expanding = False
        if not np.isnan(atr):
            prev_atr = df.iloc[max(0, idx - 12)].get(f"atr_{ATR_PERIOD}", atr)
            atr_expanding = atr > prev_atr * 1.1

        if is_early and atr_expanding:
            tp_price = row["close"] + state.range_width * 1.5
        else:
            tp_price = row["close"] + state.range_width

        # Minimum 1:1 RR
        if (tp_price - row["close"]) < risk_distance:
            tp_price = row["close"] + risk_distance

        state.bull_signal_given = True
        self.diag["signals_generated"] += 1

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

    def _check_bear_entry(self, df, idx, row, state, hour) -> Optional[TradeSignal]:
        if state.bear_signal_given:
            return None

        cond4 = state.bear_breakout_detected and not state.bear_breakout_failed
        cond5 = state.bear_retest_done

        if not (cond4 and cond5):
            return None

        # Trend filter: only take bearish breakout if M15 trend is down
        # Trend filter: skip strongly counter-trend breakouts
        m15_ema_slope = row.get("m15_ema_50_slope", np.nan)
        if not np.isnan(m15_ema_slope) and m15_ema_slope > 0.5:
            return None
        
        self.diag["all_required_pass"] += 1
        sub_conditions = {
            "range_formed": True, "range_valid": True,
            "session_active": True, "breakout_detected": True,
            "retest_done": True,
        }

        confidence = 60

        is_early = (hour < self.EARLY_CUTOFF)
        if is_early:
            confidence += 15
        sub_conditions["early_breakout"] = is_early

        no_opposite_fail = not state.bull_breakout_failed
        if no_opposite_fail:
            confidence += 10
        sub_conditions["no_opposite_fail"] = no_opposite_fail

        range_width_atr = row.get("asian_range_width_atr", 1.0)
        if np.isnan(range_width_atr):
            range_width_atr = 1.0
        tight_range = (range_width_atr < 0.75)
        if tight_range:
            confidence += 15
        sub_conditions["tight_range"] = tight_range

        if confidence < CONFIDENCE_THRESHOLD:
            return None
        self.diag["above_confidence"] += 1

        spread_buffer = SPREAD_SIMULATION_PIPS * PIP_VALUE * 2

        if tight_range:
            sl_price = state.range_high + spread_buffer
        else:
            sl_price = ((state.range_high + state.range_low) / 2) + spread_buffer

        if sl_price <= row["close"]:
            return None

        risk_distance = sl_price - row["close"]

        atr = row.get(f"atr_{ATR_PERIOD}", np.nan)
        atr_expanding = False
        if not np.isnan(atr):
            prev_atr = df.iloc[max(0, idx - 12)].get(f"atr_{ATR_PERIOD}", atr)
            atr_expanding = atr > prev_atr * 1.1

        if is_early and atr_expanding:
            tp_price = row["close"] - state.range_width * 1.5
        else:
            tp_price = row["close"] - state.range_width

        if (row["close"] - tp_price) < risk_distance:
            tp_price = row["close"] - risk_distance

        state.bear_signal_given = True
        self.diag["signals_generated"] += 1

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
        d = self.diag
        print(f"\n  STRATEGY C DIAGNOSTICS:")
        print(f"  Candles processed:        {d['candles_processed']:>10,}")
        print(f"  Range valid:              {d['range_valid']:>10,}")
        print(f"  In breakout window:       {d['in_breakout_window']:>10,}")
        print(f"  Not Friday:               {d['not_friday']:>10,}")
        print(f"  Breakouts detected:       {d['breakout_detected']:>10,}")
        print(f"  Retests detected:         {d['retest_detected']:>10,}")
        print(f"  Failed breakouts:         {d['failed_breakouts']:>10,}")
        print(f"  All required pass:        {d['all_required_pass']:>10,}")
        print(f"  Above confidence:         {d['above_confidence']:>10,}")
        print(f"  Signals generated:        {d['signals_generated']:>10,}")

        total_days = len(self.daily_states)
        valid_days = sum(1 for s in self.daily_states.values() if s.range_valid)
        print(f"\n  DAILY SUMMARY:")
        print(f"  Total days:               {total_days:>10,}")
        print(f"  Valid range days:         {valid_days:>10,}")
        print(f"  Bull breakout days:       {sum(1 for s in self.daily_states.values() if s.bull_breakout_detected):>10,}")
        print(f"  Bear breakout days:       {sum(1 for s in self.daily_states.values() if s.bear_breakout_detected):>10,}")
        print(f"  Bull retest days:         {sum(1 for s in self.daily_states.values() if s.bull_retest_done):>10,}")
        print(f"  Bear retest days:         {sum(1 for s in self.daily_states.values() if s.bear_retest_done):>10,}")
        print(f"  Bull failed days:         {sum(1 for s in self.daily_states.values() if s.bull_breakout_failed):>10,}")
        print(f"  Bear failed days:         {sum(1 for s in self.daily_states.values() if s.bear_breakout_failed):>10,}")