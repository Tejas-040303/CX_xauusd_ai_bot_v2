# """
# Project MIDAS v2 — Feature Engineering
# Phase 0: Shared features only (ATR, EMAs, RSI, ADX, sessions, candle patterns, swing points)
# Strategy-specific features will be added in Phases 1-4.

# Usage:
#     from feature_engineering import FeatureEngine
    
#     engine = FeatureEngine()
#     df = engine.compute_shared(df, timeframe="M5")
    
#     # Later phases:
#     # df = engine.compute_fvg(df)        # Phase 1
#     # df = engine.compute_trend(df)      # Phase 2
#     # df = engine.compute_session_range(df)  # Phase 3
#     # df = engine.compute_smc(df)        # Phase 4
# """

# import numpy as np
# import pandas as pd

# from config import (
#     EMA_PERIODS, ATR_PERIOD, RSI_PERIOD, ADX_PERIOD,
#     SWING_LOOKBACK, SESSIONS, PIP_VALUE
# )


# class FeatureEngine:
#     """
#     Central feature computation engine.
#     All features are added as new columns to the DataFrame.
#     Each method is idempotent — safe to call multiple times.
#     """

#     def compute_shared(self, df: pd.DataFrame, timeframe: str = "M5") -> pd.DataFrame:
#         """
#         Compute all shared features needed by every strategy.
#         Call this FIRST before any strategy-specific features.

#         Args:
#             df: DataFrame with columns [datetime, open, high, low, close, volume]
#             timeframe: Timeframe label for context-aware features

#         Returns:
#             DataFrame with shared features added
#         """
#         df = df.copy()

#         # Ensure datetime is proper type
#         if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
#             df["datetime"] = pd.to_datetime(df["datetime"])

#         # Sort by time
#         df = df.sort_values("datetime").reset_index(drop=True)

#         # ─── Core Price Features ─────────────────────
#         df = self._add_candle_features(df)

#         # ─── Moving Averages ─────────────────────────
#         df = self._add_emas(df)

#         # ─── ATR ─────────────────────────────────────
#         df = self._add_atr(df)

#         # ─── RSI ─────────────────────────────────────
#         df = self._add_rsi(df)

#         # ─── ADX ─────────────────────────────────────
#         df = self._add_adx(df)

#         # ─── Session Labels ──────────────────────────
#         df = self._add_sessions(df)

#         # ─── Time Features ───────────────────────────
#         df = self._add_time_features(df)

#         # ─── Swing Points ────────────────────────────
#         df = self._add_swing_points(df)

#         # ─── Volatility Regime ───────────────────────
#         df = self._add_volatility_regime(df)

#         # ─── EMA Distances ───────────────────────────
#         df = self._add_ema_distances(df)

#         # Store timeframe
#         df["timeframe"] = timeframe

#         return df

#     # ─── Phase 1: FVG Features ───────────────────────
#     def compute_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Phase 1: Detect Fair Value Gaps on this DataFrame.
#         Best used on M15 data for structure detection.

#         Adds columns:
#           - fvg_bull_top, fvg_bull_bottom: bullish FVG zone boundaries
#           - fvg_bear_top, fvg_bear_bottom: bearish FVG zone boundaries
#           - fvg_bull_ce, fvg_bear_ce: consequent encroachment (50% level)
#           - fvg_bull_size, fvg_bear_size: gap size in price
#           - fvg_bull_size_atr, fvg_bear_size_atr: gap size normalized by ATR
#           - fvg_detected: 1 if any FVG detected on this candle, 0 otherwise
#           - fvg_direction: "bull", "bear", or "none"

#         FVG rules:
#           - Bullish: candle_1_high < candle_3_low (gap up)
#           - Bearish: candle_1_low > candle_3_high (gap down)
#           - Min size: 0.5x ATR(14)
#           - Max size: 2.5x ATR(14)
#         """
#         df = df.copy()
#         atr_col = f"atr_{ATR_PERIOD}"

#         # Initialize columns
#         for prefix in ["bull", "bear"]:
#             for suffix in ["top", "bottom", "ce", "size", "size_atr"]:
#                 df[f"fvg_{prefix}_{suffix}"] = np.nan

#         df["fvg_detected"] = 0
#         df["fvg_direction"] = "none"

#         highs = df["high"].values
#         lows = df["low"].values
#         atr_vals = df[atr_col].values if atr_col in df.columns else np.full(len(df), 2.0)

#         for i in range(2, len(df)):
#             atr = atr_vals[i]
#             if np.isnan(atr) or atr <= 0:
#                 continue

#             min_size = 0.5 * atr
#             max_size = 2.5 * atr

#             # ── Bullish FVG: candle[i-2] high < candle[i] low ──
#             c1_high = highs[i - 2]
#             c3_low = lows[i]

#             if c3_low > c1_high:
#                 gap_size = c3_low - c1_high
#                 if min_size <= gap_size <= max_size:
#                     df.iloc[i, df.columns.get_loc("fvg_bull_bottom")] = c1_high
#                     df.iloc[i, df.columns.get_loc("fvg_bull_top")] = c3_low
#                     df.iloc[i, df.columns.get_loc("fvg_bull_ce")] = c1_high + gap_size / 2
#                     df.iloc[i, df.columns.get_loc("fvg_bull_size")] = gap_size
#                     df.iloc[i, df.columns.get_loc("fvg_bull_size_atr")] = gap_size / atr
#                     df.iloc[i, df.columns.get_loc("fvg_detected")] = 1
#                     df.iloc[i, df.columns.get_loc("fvg_direction")] = "bull"

#             # ── Bearish FVG: candle[i-2] low > candle[i] high ──
#             c1_low = lows[i - 2]
#             c3_high = highs[i]

#             if c1_low > c3_high:
#                 gap_size = c1_low - c3_high
#                 if min_size <= gap_size <= max_size:
#                     df.iloc[i, df.columns.get_loc("fvg_bear_top")] = c1_low
#                     df.iloc[i, df.columns.get_loc("fvg_bear_bottom")] = c3_high
#                     df.iloc[i, df.columns.get_loc("fvg_bear_ce")] = c3_high + gap_size / 2
#                     df.iloc[i, df.columns.get_loc("fvg_bear_size")] = gap_size
#                     df.iloc[i, df.columns.get_loc("fvg_bear_size_atr")] = gap_size / atr
#                     df.iloc[i, df.columns.get_loc("fvg_detected")] = 1
#                     # If both bull and bear detected same candle (very rare), bear overwrites
#                     if df.iloc[i, df.columns.get_loc("fvg_direction")] == "bull":
#                         df.iloc[i, df.columns.get_loc("fvg_direction")] = "both"
#                     else:
#                         df.iloc[i, df.columns.get_loc("fvg_direction")] = "bear"

#         # Count of FVGs detected (summary stat)
#         total_bull = (df["fvg_direction"].isin(["bull", "both"])).sum()
#         total_bear = (df["fvg_direction"].isin(["bear", "both"])).sum()
#         print(f"    FVG detected: {total_bull} bullish, {total_bear} bearish")

#         return df

#     # ─── Phase 2: Trend Features ─────────────────────
#     def compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Phase 2: Trend continuation features for Strategy D.
#         Most core indicators (EMAs, RSI, ADX) are already in shared features.
#         This adds Strategy D-specific derived features.

#         Adds columns:
#           - ema21_touch_bull: price came within 1x ATR of EMA21 from above (buy pullback)
#           - ema21_touch_bear: price came within 1x ATR of EMA21 from below (sell pullback)
#           - bounce_candle_bull: M5 closes bullish after touching EMA zone
#           - bounce_candle_bear: M5 closes bearish after touching EMA zone
#           - pullback_depth_atr: how deep the pullback is in ATR units
#           - trend_quality: composite score 0-4 counting alignment factors
#           - ema9_21_cross_bull: EMA9 just crossed above EMA21 (avoid, trend just starting)
#           - ema9_21_cross_bear: EMA9 just crossed below EMA21
#           - ema21_touch_count: rolling count of EMA21 touches in last 50 candles
#         """
#         df = df.copy()
#         atr_col = f"atr_{ATR_PERIOD}"

#         # ── EMA21 Touch Detection ────────────────────────
#         # Price within 1x ATR of EMA21
#         if "ema_21" in df.columns and atr_col in df.columns:
#             dist_to_ema21 = (df["close"] - df["ema_21"]).abs()
#             within_atr = dist_to_ema21 <= df[atr_col]

#             # Bull touch: price is above EMA21 but close to it (pulling back in uptrend)
#             df["ema21_touch_bull"] = (
#                 within_atr &
#                 (df["close"] >= df["ema_21"]) &
#                 (df["low"] <= df["ema_21"] + df[atr_col])
#             ).astype(int)

#             # Bear touch: price is below EMA21 but close to it (pulling back in downtrend)
#             df["ema21_touch_bear"] = (
#                 within_atr &
#                 (df["close"] <= df["ema_21"]) &
#                 (df["high"] >= df["ema_21"] - df[atr_col])
#             ).astype(int)

#             # Rolling touch count (how many times EMA21 was touched in last 50 candles)
#             # First/second touch = strong, third+ = trend weakening
#             df["ema21_touch_count"] = (
#                 (df["ema21_touch_bull"] | df["ema21_touch_bear"])
#                 .rolling(50, min_periods=1).sum()
#             )
#         else:
#             df["ema21_touch_bull"] = 0
#             df["ema21_touch_bear"] = 0
#             df["ema21_touch_count"] = 0

#         # ── Bounce Candle Detection ──────────────────────
#         # After touching EMA zone, candle closes back in trend direction
#         prev_touch_bull = df["ema21_touch_bull"].shift(1).fillna(0)
#         prev_touch_bear = df["ema21_touch_bear"].shift(1).fillna(0)

#         df["bounce_candle_bull"] = (
#             (prev_touch_bull == 1) &                # Previous candle touched EMA21 zone
#             (df["close"] > df["open"]) &            # Current candle is bullish
#             (df["close"] > df["ema_21"])             # Closed above EMA21
#         ).astype(int)

#         df["bounce_candle_bear"] = (
#             (prev_touch_bear == 1) &
#             (df["close"] < df["open"]) &
#             (df["close"] < df["ema_21"])
#         ).astype(int)

#         # Also accept engulfing as bounce
#         if "is_engulfing_bull" in df.columns:
#             df["bounce_candle_bull"] = df["bounce_candle_bull"] | (
#                 (prev_touch_bull == 1) & (df["is_engulfing_bull"] == 1)
#             ).astype(int)
#         if "is_engulfing_bear" in df.columns:
#             df["bounce_candle_bear"] = df["bounce_candle_bear"] | (
#                 (prev_touch_bear == 1) & (df["is_engulfing_bear"] == 1)
#             ).astype(int)

#         # ── Pullback Depth ───────────────────────────────
#         # How far price has pulled back from recent swing, measured in ATR
#         if "last_swing_high" in df.columns and atr_col in df.columns:
#             df["pullback_depth_bull_atr"] = (
#                 (df["last_swing_high"] - df["low"]) / df[atr_col].replace(0, np.nan)
#             )
#             df["pullback_depth_bear_atr"] = (
#                 (df["high"] - df["last_swing_low"]) / df[atr_col].replace(0, np.nan)
#             )
#         else:
#             df["pullback_depth_bull_atr"] = 0
#             df["pullback_depth_bear_atr"] = 0

#         # ── Trend Quality Score ──────────────────────────
#         # Composite 0-4: how many trend factors align
#         df["trend_quality_bull"] = 0
#         df["trend_quality_bear"] = 0

#         if "ema_partial_bull" in df.columns:
#             df["trend_quality_bull"] += df["ema_partial_bull"]  # EMA 9>21>50
#         if "above_ema200" in df.columns:
#             df["trend_quality_bull"] += df["above_ema200"]      # Above EMA200
#         if "trend_strong" in df.columns:
#             df["trend_quality_bull"] += df["trend_strong"]      # ADX > 25
#         if "rsi_pullback_buy" in df.columns:
#             df["trend_quality_bull"] += df["rsi_pullback_buy"]  # RSI in buy zone

#         if "ema_partial_bear" in df.columns:
#             df["trend_quality_bear"] += df["ema_partial_bear"]
#         if "above_ema200" in df.columns:
#             df["trend_quality_bear"] += (1 - df["above_ema200"])  # Below EMA200
#         if "trend_strong" in df.columns:
#             df["trend_quality_bear"] += df["trend_strong"]
#         if "rsi_pullback_sell" in df.columns:
#             df["trend_quality_bear"] += df["rsi_pullback_sell"]

#         # ── EMA9/21 Crossover Detection (avoid signal) ──
#         # If EMA9 just crossed EMA21, trend is brand new — risky
#         if "ema_9" in df.columns and "ema_21" in df.columns:
#             ema9_above = df["ema_9"] > df["ema_21"]
#             df["ema9_21_cross_bull"] = (ema9_above & ~ema9_above.shift(1).fillna(False)).astype(int)
#             df["ema9_21_cross_bear"] = (~ema9_above & ema9_above.shift(1).fillna(True)).astype(int)

#             # Rolling: how many candles since last cross (trend maturity)
#             cross_any = df["ema9_21_cross_bull"] | df["ema9_21_cross_bear"]
#             df["candles_since_ema_cross"] = cross_any.groupby(
#                 cross_any.cumsum()
#             ).cumcount()
#         else:
#             df["ema9_21_cross_bull"] = 0
#             df["ema9_21_cross_bear"] = 0
#             df["candles_since_ema_cross"] = 0

#         # Summary
#         bull_bounces = df["bounce_candle_bull"].sum()
#         bear_bounces = df["bounce_candle_bear"].sum()
#         print(f"    Trend features: {bull_bounces} bull bounces, {bear_bounces} bear bounces detected")

#         return df

#     # ─── Phase 3: Session Range Features ─────────────
#     def compute_session_range(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Phase 3: Session Range Breakout features.
#         Computes Asian session range per trading day and marks breakouts.

#         Broker timezone: UTC+2
#         Asian range hours: 02:00-08:00 broker time (= 00:00-06:00 UTC)
#         Breakout allowed: 08:00-18:00 broker time (= 06:00-16:00 UTC)
#         Early breakout: before 14:00 broker time (= before 12:00 UTC)

#         Adds columns:
#           - asian_range_high, asian_range_low, asian_range_width
#           - asian_range_mid: midpoint of range
#           - asian_range_width_atr: width normalized by ATR
#           - asian_range_valid: range meets size criteria (0.3x-1.5x ATR)
#           - in_breakout_window: current candle is in valid breakout hours
#           - is_early_breakout_window: before 14:00 broker time
#           - breakout_bull: M5 close above range high
#           - breakout_bear: M5 close below range low
#         """
#         df = df.copy()
#         atr_col = f"atr_{ATR_PERIOD}"

#         # Broker time hours (data timestamps are UTC+2)
#         ASIAN_START = 2   # 02:00 broker = 00:00 UTC
#         ASIAN_END = 8     # 08:00 broker = 06:00 UTC
#         BREAKOUT_START = 8   # 08:00 broker
#         BREAKOUT_END = 18    # 18:00 broker = 16:00 UTC
#         EARLY_CUTOFF = 14    # 14:00 broker = 12:00 UTC

#         hour = df["datetime"].dt.hour
#         date = df["datetime"].dt.date

#         # ── Compute Asian range per trading day ──────────
#         # Asian session mask
#         asian_mask = (hour >= ASIAN_START) & (hour < ASIAN_END)

#         # Group by date to get daily Asian range
#         df["_trade_date"] = date
#         asian_data = df[asian_mask].groupby("_trade_date").agg(
#             asian_high=("high", "max"),
#             asian_low=("low", "min"),
#             asian_candles=("high", "count"),
#         ).reset_index()

#         # Compute range metrics
#         asian_data["asian_range_width"] = asian_data["asian_high"] - asian_data["asian_low"]
#         asian_data["asian_range_mid"] = (asian_data["asian_high"] + asian_data["asian_low"]) / 2

#         # Merge back into main df
#         df = df.merge(
#             asian_data.rename(columns={
#                 "_trade_date": "_trade_date",
#                 "asian_high": "asian_range_high",
#                 "asian_low": "asian_range_low",
#                 "asian_candles": "asian_session_candles",
#             }),
#             on="_trade_date",
#             how="left",
#         )

#         # Forward fill for candles that don't have a range yet (early Asian)
#         # Actually the range should only be available AFTER Asian session ends
#         # So null it out during Asian session
#         df.loc[asian_mask, "asian_range_high"] = np.nan
#         df.loc[asian_mask, "asian_range_low"] = np.nan
#         df.loc[asian_mask, "asian_range_width"] = np.nan
#         df.loc[asian_mask, "asian_range_mid"] = np.nan

#         # ── Compute Daily ATR from M5 data ─────────────
#         # M5 ATR measures 5-minute volatility — wrong scale for 6-hour range comparison
#         # Daily ATR = rolling 14-day average of daily high-low range
#         daily_hl = df.groupby("_trade_date").agg(
#             day_high=("high", "max"),
#             day_low=("low", "min"),
#         )
#         daily_hl["daily_range"] = daily_hl["day_high"] - daily_hl["day_low"]
#         daily_hl["daily_atr_14"] = daily_hl["daily_range"].ewm(span=14, adjust=False).mean()
#         daily_hl = daily_hl[["daily_atr_14"]].reset_index()

#         df = df.merge(daily_hl, on="_trade_date", how="left")

#         # ── Range validity check ─────────────────────────
#         # Compare Asian range width to DAILY ATR (correct scale)
#         df["asian_range_width_atr"] = df["asian_range_width"] / df["daily_atr_14"].replace(0, np.nan)
#         df["asian_range_valid"] = (
#             (df["asian_range_width_atr"] >= 0.1) &
#             (df["asian_range_width_atr"] <= 0.7) &
#             (df["asian_session_candles"] >= 5)
#         ).astype(int)

#         # ── Time window flags ────────────────────────────
#         df["in_breakout_window"] = (
#             (hour >= BREAKOUT_START) & (hour < BREAKOUT_END)
#         ).astype(int)

#         df["is_early_breakout_window"] = (
#             (hour >= BREAKOUT_START) & (hour < EARLY_CUTOFF)
#         ).astype(int)

#         # ── Breakout detection ───────────────────────────
#         # M5 close above/below range
#         df["breakout_bull"] = (
#             df["in_breakout_window"].astype(bool) &
#             df["asian_range_valid"].astype(bool) &
#             (df["close"] > df["asian_range_high"])
#         ).astype(int)

#         df["breakout_bear"] = (
#             df["in_breakout_window"].astype(bool) &
#             df["asian_range_valid"].astype(bool) &
#             (df["close"] < df["asian_range_low"])
#         ).astype(int)

#         # ── Friday flag ──────────────────────────────────
#         df["is_friday"] = (df["datetime"].dt.dayofweek == 4).astype(int)

#         # ── Distance from range edges ────────────────────
#         df["dist_to_range_high"] = df["close"] - df["asian_range_high"]
#         df["dist_to_range_low"] = df["close"] - df["asian_range_low"]

#         # Cleanup
#         df = df.drop(columns=["_trade_date"], errors="ignore")

#         # Summary
#         bull_breakouts = df["breakout_bull"].sum()
#         bear_breakouts = df["breakout_bear"].sum()
#         valid_days = df[df["asian_range_valid"] == 1]["datetime"].dt.date.nunique()
#         print(f"    Session range: {valid_days} valid range days, {bull_breakouts} bull breakouts, {bear_breakouts} bear breakouts")

#         return df

#     # ─── Phase 4: SMC Features ───────────────────────
#     def compute_smc(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Phase 4: Smart Money Concepts features for Strategy A.
#         Detects Break of Structure (BOS), equal highs/lows (liquidity pools),
#         and Order Block zones on M15 data.

#         Adds columns:
#           - bos_bull: bullish BOS detected (close breaks above recent swing high)
#           - bos_bear: bearish BOS detected (close breaks below recent swing low)
#           - bos_bull_level: the swing high level that was broken
#           - bos_bear_level: the swing low level that was broken
#           - equal_highs: 1 if 2+ swing highs within tolerance at this level
#           - equal_lows: 1 if 2+ swing lows within tolerance at this level
#           - equal_highs_level: price level of the equal highs cluster
#           - equal_lows_level: price level of the equal lows cluster
#           - equal_highs_count: number of highs in the cluster
#           - equal_lows_count: number of lows in the cluster
#           - ob_bull_high, ob_bull_low: bullish OB zone (last bearish candle before bullish BOS)
#           - ob_bear_high, ob_bear_low: bearish OB zone (last bullish candle before bearish BOS)
#           - ob_bull_impulse_size: size of the impulse that caused bullish BOS
#           - ob_bear_impulse_size: size of the impulse that caused bearish BOS
#         """
#         df = df.copy()
#         atr_col = f"atr_{ATR_PERIOD}"
#         n = SWING_LOOKBACK

#         # Initialize all columns
#         for col in ["bos_bull", "bos_bear"]:
#             df[col] = 0
#         for col in ["bos_bull_level", "bos_bear_level",
#                      "ob_bull_high", "ob_bull_low", "ob_bear_high", "ob_bear_low",
#                      "ob_bull_impulse_size", "ob_bear_impulse_size",
#                      "equal_highs_level", "equal_lows_level"]:
#             df[col] = np.nan
#         df["equal_highs"] = 0
#         df["equal_lows"] = 0
#         df["equal_highs_count"] = 0
#         df["equal_lows_count"] = 0

#         highs = df["high"].values
#         lows = df["low"].values
#         closes = df["close"].values
#         opens = df["open"].values
#         atr_vals = df[atr_col].values if atr_col in df.columns else np.full(len(df), 2.0)
#         swing_high_prices = df["swing_high_price"].values if "swing_high_price" in df.columns else np.full(len(df), np.nan)
#         swing_low_prices = df["swing_low_price"].values if "swing_low_price" in df.columns else np.full(len(df), np.nan)

#         # Collect all swing highs and lows with their indices
#         swing_highs = []  # [(index, price), ...]
#         swing_lows = []

#         for i in range(len(df)):
#             if not np.isnan(swing_high_prices[i]):
#                 swing_highs.append((i, swing_high_prices[i]))
#             if not np.isnan(swing_low_prices[i]):
#                 swing_lows.append((i, swing_low_prices[i]))

#         # ── BOS Detection + OB Identification ────────────
#         # Track the most recent unbroken swing high/low
#         last_swing_high_price = np.nan
#         last_swing_high_idx = -1
#         last_swing_low_price = np.nan
#         last_swing_low_idx = -1

#         bos_bull_count = 0
#         bos_bear_count = 0
#         ob_bull_count = 0
#         ob_bear_count = 0

#         for i in range(n + 1, len(df)):
#             atr = atr_vals[i]
#             if np.isnan(atr) or atr <= 0:
#                 continue

#             # Update last swing high/low
#             if not np.isnan(swing_high_prices[i]):
#                 last_swing_high_price = swing_high_prices[i]
#                 last_swing_high_idx = i
#             if not np.isnan(swing_low_prices[i]):
#                 last_swing_low_price = swing_low_prices[i]
#                 last_swing_low_idx = i

#             # ── Bullish BOS: close breaks above last swing high ──
#             if (not np.isnan(last_swing_high_price) and
#                 closes[i] > last_swing_high_price and
#                 closes[i - 1] <= last_swing_high_price):

#                 df.iat[i, df.columns.get_loc("bos_bull")] = 1
#                 df.iat[i, df.columns.get_loc("bos_bull_level")] = last_swing_high_price
#                 bos_bull_count += 1

#                 # Find OB: last BEARISH candle body before this impulse
#                 # Walk backwards from current candle to find the impulse start
#                 ob_found = False
#                 for j in range(i - 1, max(i - 30, 0), -1):
#                     # Look for last bearish candle (close < open)
#                     if closes[j] < opens[j]:
#                         body_size = abs(opens[j] - closes[j])
#                         if body_size >= 0.2 * atr:  # Min body size
#                             # OB zone = candle body (for bearish: high=open, low=close)
#                             df.iat[i, df.columns.get_loc("ob_bull_high")] = opens[j]
#                             df.iat[i, df.columns.get_loc("ob_bull_low")] = closes[j]
#                             df.iat[i, df.columns.get_loc("ob_bull_impulse_size")] = closes[i] - closes[j]
#                             ob_bull_count += 1
#                             ob_found = True
#                             break
#                     # If we hit a bullish candle that's part of the impulse, keep going
#                     # But if we've gone past 30 candles, give up

#                 # Invalidate this swing high so it can't trigger again
#                 last_swing_high_price = closes[i]
#                 last_swing_high_idx = i

#             # ── Bearish BOS: close breaks below last swing low ──
#             if (not np.isnan(last_swing_low_price) and
#                 closes[i] < last_swing_low_price and
#                 closes[i - 1] >= last_swing_low_price):

#                 df.iat[i, df.columns.get_loc("bos_bear")] = 1
#                 df.iat[i, df.columns.get_loc("bos_bear_level")] = last_swing_low_price
#                 bos_bear_count += 1

#                 # Find OB: last BULLISH candle body before this impulse
#                 ob_found = False
#                 for j in range(i - 1, max(i - 30, 0), -1):
#                     if closes[j] > opens[j]:
#                         body_size = abs(closes[j] - opens[j])
#                         if body_size >= 0.2 * atr:
#                             # OB zone = candle body (for bullish: high=close, low=open)
#                             df.iat[i, df.columns.get_loc("ob_bear_high")] = closes[j]
#                             df.iat[i, df.columns.get_loc("ob_bear_low")] = opens[j]
#                             df.iat[i, df.columns.get_loc("ob_bear_impulse_size")] = closes[j] - closes[i]
#                             ob_bear_count += 1
#                             ob_found = True
#                             break

#                 last_swing_low_price = closes[i]
#                 last_swing_low_idx = i

#         # ── Equal Highs/Lows Detection (Liquidity Pools) ──
#         eq_high_count = 0
#         eq_low_count = 0
#         lookback_candles = 100  # Look within 100 M15 candles

#         for i in range(len(swing_highs)):
#             idx_i, price_i = swing_highs[i]
#             atr = atr_vals[idx_i] if not np.isnan(atr_vals[idx_i]) else 2.0
#             tolerance = 0.3 * atr
#             cluster = [(idx_i, price_i)]

#             for j in range(i + 1, len(swing_highs)):
#                 idx_j, price_j = swing_highs[j]
#                 if idx_j - idx_i > lookback_candles:
#                     break
#                 if abs(price_j - price_i) <= tolerance:
#                     cluster.append((idx_j, price_j))

#             if len(cluster) >= 2:
#                 # Mark the LAST swing in the cluster
#                 last_idx = cluster[-1][0]
#                 avg_price = np.mean([p for _, p in cluster])
#                 df.iat[last_idx, df.columns.get_loc("equal_highs")] = 1
#                 df.iat[last_idx, df.columns.get_loc("equal_highs_level")] = avg_price
#                 df.iat[last_idx, df.columns.get_loc("equal_highs_count")] = len(cluster)
#                 eq_high_count += 1

#         for i in range(len(swing_lows)):
#             idx_i, price_i = swing_lows[i]
#             atr = atr_vals[idx_i] if not np.isnan(atr_vals[idx_i]) else 2.0
#             tolerance = 0.3 * atr
#             cluster = [(idx_i, price_i)]

#             for j in range(i + 1, len(swing_lows)):
#                 idx_j, price_j = swing_lows[j]
#                 if idx_j - idx_i > lookback_candles:
#                     break
#                 if abs(price_j - price_i) <= tolerance:
#                     cluster.append((idx_j, price_j))

#             if len(cluster) >= 2:
#                 last_idx = cluster[-1][0]
#                 avg_price = np.mean([p for _, p in cluster])
#                 df.iat[last_idx, df.columns.get_loc("equal_lows")] = 1
#                 df.iat[last_idx, df.columns.get_loc("equal_lows_level")] = avg_price
#                 df.iat[last_idx, df.columns.get_loc("equal_lows_count")] = len(cluster)
#                 eq_low_count += 1

#         # Forward fill equal highs/lows levels so every candle knows the nearest pool
#         df["last_equal_highs_level"] = df["equal_highs_level"].ffill()
#         df["last_equal_lows_level"] = df["equal_lows_level"].ffill()

#         print(f"    SMC features: {bos_bull_count} bull BOS, {bos_bear_count} bear BOS, "
#               f"{ob_bull_count} bull OBs, {ob_bear_count} bear OBs, "
#               f"{eq_high_count} equal highs clusters, {eq_low_count} equal lows clusters")

#         return df

#     # ═════════════════════════════════════════════════
#     # PRIVATE METHODS — Shared Feature Implementations
#     # ═════════════════════════════════════════════════

#     def _add_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Basic candle-derived features."""
#         # Body and wick measurements
#         df["body"] = df["close"] - df["open"]
#         df["body_abs"] = df["body"].abs()
#         df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
#         df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
#         df["candle_range"] = df["high"] - df["low"]

#         # Direction
#         df["is_bullish"] = (df["close"] > df["open"]).astype(int)

#         # Body ratio (body size relative to total range)
#         df["body_ratio"] = np.where(
#             df["candle_range"] > 0,
#             df["body_abs"] / df["candle_range"],
#             0
#         )

#         # Candle patterns (simple)
#         # Doji: body < 10% of range
#         df["is_doji"] = (df["body_ratio"] < 0.1).astype(int)

#         # Pin bar: one wick > 60% of range, body < 30%
#         df["is_pin_bar_bull"] = (
#             (df["lower_wick"] > 0.6 * df["candle_range"]) &
#             (df["body_ratio"] < 0.3)
#         ).astype(int)

#         df["is_pin_bar_bear"] = (
#             (df["upper_wick"] > 0.6 * df["candle_range"]) &
#             (df["body_ratio"] < 0.3)
#         ).astype(int)

#         # Engulfing (current candle body fully covers previous candle body)
#         prev_open = df["open"].shift(1)
#         prev_close = df["close"].shift(1)
#         prev_body_high = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
#         prev_body_low = pd.concat([prev_open, prev_close], axis=1).min(axis=1)
#         curr_body_high = pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
#         curr_body_low = pd.concat([df["open"], df["close"]], axis=1).min(axis=1)

#         df["is_engulfing_bull"] = (
#             (df["is_bullish"] == 1) &
#             (df["is_bullish"].shift(1) == 0) &
#             (curr_body_high > prev_body_high) &
#             (curr_body_low < prev_body_low)
#         ).astype(int)

#         df["is_engulfing_bear"] = (
#             (df["is_bullish"] == 0) &
#             (df["is_bullish"].shift(1) == 1) &
#             (curr_body_high > prev_body_high) &
#             (curr_body_low < prev_body_low)
#         ).astype(int)

#         return df

#     def _add_emas(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Exponential Moving Averages."""
#         for period in EMA_PERIODS:
#             df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

#         # EMA slopes (rate of change over last 3 candles)
#         for period in EMA_PERIODS:
#             df[f"ema_{period}_slope"] = df[f"ema_{period}"].diff(3) / 3

#         # EMA stack alignment
#         # 1 = perfect bullish (9 > 21 > 50 > 200)
#         # -1 = perfect bearish (9 < 21 < 50 < 200)
#         # 0 = mixed
#         if all(p in EMA_PERIODS for p in [9, 21, 50, 200]):
#             bull_stack = (
#                 (df["ema_9"] > df["ema_21"]) &
#                 (df["ema_21"] > df["ema_50"]) &
#                 (df["ema_50"] > df["ema_200"])
#             )
#             bear_stack = (
#                 (df["ema_9"] < df["ema_21"]) &
#                 (df["ema_21"] < df["ema_50"]) &
#                 (df["ema_50"] < df["ema_200"])
#             )
#             df["ema_stack"] = 0
#             df.loc[bull_stack, "ema_stack"] = 1
#             df.loc[bear_stack, "ema_stack"] = -1

#             # Partial alignment (9 > 21 > 50 but not necessarily > 200)
#             df["ema_partial_bull"] = (
#                 (df["ema_9"] > df["ema_21"]) &
#                 (df["ema_21"] > df["ema_50"])
#             ).astype(int)

#             df["ema_partial_bear"] = (
#                 (df["ema_9"] < df["ema_21"]) &
#                 (df["ema_21"] < df["ema_50"])
#             ).astype(int)

#         # Price position relative to EMA 200
#         if 200 in EMA_PERIODS:
#             df["above_ema200"] = (df["close"] > df["ema_200"]).astype(int)

#         return df

#     def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Average True Range."""
#         high = df["high"]
#         low = df["low"]
#         prev_close = df["close"].shift(1)

#         tr1 = high - low
#         tr2 = (high - prev_close).abs()
#         tr3 = (low - prev_close).abs()

#         df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#         df[f"atr_{ATR_PERIOD}"] = df["true_range"].ewm(span=ATR_PERIOD, adjust=False).mean()

#         # ATR in pips (for gold: 1 pip = $0.10)
#         df[f"atr_{ATR_PERIOD}_pips"] = df[f"atr_{ATR_PERIOD}"] / PIP_VALUE

#         # Normalized ATR (ATR / close price — useful for regime detection)
#         df["atr_normalized"] = df[f"atr_{ATR_PERIOD}"] / df["close"]

#         return df

#     def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Relative Strength Index."""
#         delta = df["close"].diff()
#         gain = delta.clip(lower=0)
#         loss = (-delta).clip(lower=0)

#         avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
#         avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()

#         rs = avg_gain / avg_loss.replace(0, np.nan)
#         df[f"rsi_{RSI_PERIOD}"] = 100 - (100 / (1 + rs))

#         # RSI zones
#         df["rsi_overbought"] = (df[f"rsi_{RSI_PERIOD}"] > 70).astype(int)
#         df["rsi_oversold"] = (df[f"rsi_{RSI_PERIOD}"] < 30).astype(int)

#         # RSI in pullback zone (40-55 for buys, 45-60 for sells)
#         df["rsi_pullback_buy"] = (
#             (df[f"rsi_{RSI_PERIOD}"] >= 40) & (df[f"rsi_{RSI_PERIOD}"] <= 55)
#         ).astype(int)

#         df["rsi_pullback_sell"] = (
#             (df[f"rsi_{RSI_PERIOD}"] >= 45) & (df[f"rsi_{RSI_PERIOD}"] <= 60)
#         ).astype(int)

#         return df

#     def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Average Directional Index — trend strength indicator."""
#         high = df["high"]
#         low = df["low"]
#         close = df["close"]

#         # +DM and -DM
#         plus_dm = high.diff()
#         minus_dm = -low.diff()

#         plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
#         minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

#         # Smoothed with EMA
#         atr = df[f"atr_{ATR_PERIOD}"] if f"atr_{ATR_PERIOD}" in df.columns else df["true_range"].ewm(span=ADX_PERIOD, adjust=False).mean()

#         plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean()
#         minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean()

#         # +DI and -DI
#         plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
#         minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)

#         df["plus_di"] = plus_di
#         df["minus_di"] = minus_di

#         # DX and ADX
#         di_sum = plus_di + minus_di
#         di_diff = (plus_di - minus_di).abs()
#         dx = 100 * di_diff / di_sum.replace(0, np.nan)

#         df[f"adx_{ADX_PERIOD}"] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()

#         # Trend strength labels
#         df["trend_strong"] = (df[f"adx_{ADX_PERIOD}"] > 25).astype(int)
#         df["trend_very_strong"] = (df[f"adx_{ADX_PERIOD}"] > 40).astype(int)
#         df["trend_weak"] = (df[f"adx_{ADX_PERIOD}"] < 20).astype(int)

#         return df

#     def _add_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Label each candle with its trading session.
#         Uses UTC hours from datetime column.
        
#         NOTE: If your MT5 broker uses a different timezone (e.g., UTC+2),
#         adjust the datetime before calling this, or modify SESSIONS in config.
#         """
#         hour = df["datetime"].dt.hour

#         # Default to "off"
#         df["session"] = "off"

#         # Apply session labels (order matters — later overwrites earlier for overlaps)
#         for session_name, times in SESSIONS.items():
#             start_h = times["start"]
#             end_h = times["end"]

#             if start_h < end_h:
#                 mask = (hour >= start_h) & (hour < end_h)
#             else:  # Wraps around midnight
#                 mask = (hour >= start_h) | (hour < end_h)

#             df.loc[mask, "session"] = session_name

#         # Handle London/NY overlap (12:00-14:00 UTC) — label as "london_ny_overlap"
#         overlap_mask = (hour >= 12) & (hour < 14)
#         df.loc[overlap_mask, "session"] = "london_ny_overlap"

#         # One-hot encode sessions for model input
#         for session_name in list(SESSIONS.keys()) + ["off", "london_ny_overlap"]:
#             df[f"session_{session_name}"] = (df["session"] == session_name).astype(int)

#         return df

#     def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Time-based features for cyclical patterns."""
#         dt = df["datetime"]

#         df["hour"] = dt.dt.hour
#         df["minute"] = dt.dt.minute
#         df["day_of_week"] = dt.dt.dayofweek  # 0=Monday, 4=Friday
#         df["day_of_month"] = dt.dt.day
#         df["month"] = dt.dt.month

#         # Cyclical encoding (sin/cos) for hour — captures the circular nature
#         df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
#         df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

#         # Day of week cyclical
#         df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
#         df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)

#         # Is Friday (reduced follow-through, relevant for Strategy C)
#         df["is_friday"] = (df["day_of_week"] == 4).astype(int)

#         return df

#     def _add_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Detect swing highs and swing lows.
#         A swing high has a higher high than the N candles on each side.
#         A swing low has a lower low than the N candles on each side.
#         """
#         n = SWING_LOOKBACK

#         df["swing_high"] = 0
#         df["swing_low"] = 0
#         df["swing_high_price"] = np.nan
#         df["swing_low_price"] = np.nan

#         highs = df["high"].values
#         lows = df["low"].values

#         for i in range(n, len(df) - n):
#             # Swing high: current high is highest in window
#             left_highs = highs[i - n:i]
#             right_highs = highs[i + 1:i + n + 1]

#             if highs[i] > left_highs.max() and highs[i] > right_highs.max():
#                 df.iloc[i, df.columns.get_loc("swing_high")] = 1
#                 df.iloc[i, df.columns.get_loc("swing_high_price")] = highs[i]

#             # Swing low: current low is lowest in window
#             left_lows = lows[i - n:i]
#             right_lows = lows[i + 1:i + n + 1]

#             if lows[i] < left_lows.min() and lows[i] < right_lows.min():
#                 df.iloc[i, df.columns.get_loc("swing_low")] = 1
#                 df.iloc[i, df.columns.get_loc("swing_low_price")] = lows[i]

#         # Forward-fill the most recent swing high/low prices
#         # (so every candle knows where the last swing was)
#         df["last_swing_high"] = df["swing_high_price"].ffill()
#         df["last_swing_low"] = df["swing_low_price"].ffill()

#         # Distance from current price to last swing
#         df["dist_to_swing_high"] = df["close"] - df["last_swing_high"]
#         df["dist_to_swing_low"] = df["close"] - df["last_swing_low"]

#         return df

#     def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Classify current volatility regime based on ATR percentile.
#         Uses rolling 100-period percentile rank of ATR.
#         """
#         atr_col = f"atr_{ATR_PERIOD}"
#         if atr_col not in df.columns:
#             return df

#         # Rolling percentile rank (where does current ATR sit vs last 100 candles)
#         window = 100
#         df["atr_percentile"] = df[atr_col].rolling(window).apply(
#             lambda x: pd.Series(x).rank(pct=True).iloc[-1],
#             raw=False
#         )

#         # Regime labels
#         df["vol_regime"] = "normal"
#         df.loc[df["atr_percentile"] > 0.8, "vol_regime"] = "high_volatility"
#         df.loc[df["atr_percentile"] < 0.2, "vol_regime"] = "low_volatility"

#         # One-hot for model input
#         for regime in ["low_volatility", "normal", "high_volatility"]:
#             df[f"vol_{regime}"] = (df["vol_regime"] == regime).astype(int)

#         return df

#     def _add_ema_distances(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Distance of price from each EMA, normalized by ATR."""
#         atr_col = f"atr_{ATR_PERIOD}"
#         if atr_col not in df.columns:
#             return df

#         for period in EMA_PERIODS:
#             ema_col = f"ema_{period}"
#             if ema_col in df.columns:
#                 # Raw distance
#                 df[f"dist_ema_{period}"] = df["close"] - df[ema_col]

#                 # Normalized by ATR (how many ATRs away from EMA)
#                 df[f"dist_ema_{period}_atr"] = (
#                     df[f"dist_ema_{period}"] / df[atr_col].replace(0, np.nan)
#                 )

#         return df


# # ─── Utility Functions ───────────────────────────────────

# def load_and_compute(filepath: str, timeframe: str = "M5") -> pd.DataFrame:
#     """
#     Convenience: load a raw CSV and compute all shared features.

#     Args:
#         filepath: Path to raw XAUUSD CSV file
#         timeframe: Timeframe label

#     Returns:
#         DataFrame with all shared features computed
#     """
#     df = pd.read_csv(filepath, parse_dates=["datetime"])

#     engine = FeatureEngine()
#     df = engine.compute_shared(df, timeframe=timeframe)

#     return df


# def process_all_timeframes(raw_dir: str = None, processed_dir: str = None):
#     """
#     Process all raw timeframe CSVs and save with features.

#     Args:
#         raw_dir: Directory with raw CSVs (default: config DATA_RAW)
#         processed_dir: Output directory (default: config DATA_PROCESSED)
#     """
#     from config import DATA_RAW, DATA_PROCESSED, MT5_TIMEFRAMES

#     raw_dir = raw_dir or DATA_RAW
#     processed_dir = processed_dir or DATA_PROCESSED

#     engine = FeatureEngine()

#     for tf_name in MT5_TIMEFRAMES:
#         raw_file = Path(raw_dir) / f"XAUUSD_{tf_name}.csv"
#         if not raw_file.exists():
#             print(f"  {tf_name}: Raw file not found, skipping")
#             continue

#         print(f"  Processing {tf_name}...", end=" ")
#         df = pd.read_csv(raw_file, parse_dates=["datetime"])
#         df = engine.compute_shared(df, timeframe=tf_name)

#         output_file = Path(processed_dir) / f"XAUUSD_{tf_name}_features.csv"
#         df.to_csv(output_file, index=False)

#         print(f"{len(df):,} candles, {len(df.columns)} features → {output_file.name}")

#     print("\n✅ All timeframes processed!")


# if __name__ == "__main__":
#     print("=" * 60)
#     print("MIDAS v2 — Feature Engineering (Phase 0: Shared Features)")
#     print("=" * 60)
#     process_all_timeframes()

"""
Project MIDAS v2 — Feature Engineering
Phase 0: Shared features only (ATR, EMAs, RSI, ADX, sessions, candle patterns, swing points)
Strategy-specific features will be added in Phases 1-4.

Usage:
    from feature_engineering import FeatureEngine
    
    engine = FeatureEngine()
    df = engine.compute_shared(df, timeframe="M5")
    
    # Later phases:
    # df = engine.compute_fvg(df)        # Phase 1
    # df = engine.compute_trend(df)      # Phase 2
    # df = engine.compute_session_range(df)  # Phase 3
    # df = engine.compute_smc(df)        # Phase 4
"""

import numpy as np
import pandas as pd

from config import (
    EMA_PERIODS, ATR_PERIOD, RSI_PERIOD, ADX_PERIOD,
    SWING_LOOKBACK, SESSIONS, PIP_VALUE
)


class FeatureEngine:
    """
    Central feature computation engine.
    All features are added as new columns to the DataFrame.
    Each method is idempotent — safe to call multiple times.
    """

    def compute(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
        """
        Full live-trading pipeline: M5 + M15 → merged feature DataFrame.
        Mirrors the exact logic from run_combined_abcd.load_and_prepare_data().

        Args:
            df_m5:  Raw M5 OHLCV DataFrame from MT5 (columns: datetime,open,high,low,close,volume)
            df_m15: Raw M15 OHLCV DataFrame from MT5

        Returns:
            Merged DataFrame with all 167+ features, ready for strategies A-E.
        """
        # ── M15: shared + FVG + SMC ───────────────────────
        df_m15 = self.compute_shared(df_m15, timeframe="M15")
        df_m15 = self.compute_fvg(df_m15)
        df_m15 = self.compute_smc(df_m15)

        # ── M5: shared + trend + session range ───────────
        df_m5 = self.compute_shared(df_m5, timeframe="M5")
        df_m5 = self.compute_trend(df_m5)
        df_m5 = self.compute_session_range(df_m5)

        # ── Merge M15 features into M5 (asof backward) ───
        df_merged = self._merge_timeframes(df_m5, df_m15)

        return df_merged

    def _merge_timeframes(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
        """
        Merge M15 features into M5 using backward asof join.
        Each M5 candle gets the most recent M15 candle's features.
        Mirrors merge_timeframes() in run_combined_abcd.py.
        """
        import pandas as pd

        df_m5  = df_m5.sort_values("datetime").reset_index(drop=True)
        df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

        # Columns to bring from M15
        m15_cols = ["datetime"]

        fvg_cols = [c for c in df_m15.columns if c.startswith("fvg_")]
        m15_cols.extend(fvg_cols)

        smc_cols = [c for c in df_m15.columns
                    if c.startswith(("bos_", "ob_", "equal_", "last_equal_"))]
        m15_cols.extend(smc_cols)

        # Raw M15 OHLCV (needed by Strategy A sweep detection)
        m15_cols.extend(["open", "high", "low", "close"])

        # Key M15 indicators (EMA slopes, swing points, ATR)
        extra = [
            "ema_50_slope", "ema_200_slope", "ema_stack",
            f"atr_{ATR_PERIOD}",
            "swing_high", "swing_low", "swing_high_price", "swing_low_price",
            "last_swing_high", "last_swing_low",
        ]
        for col in extra:
            if col in df_m15.columns:
                m15_cols.append(col)

        m15_cols = list(dict.fromkeys(m15_cols))  # deduplicate, preserve order
        df_m15_sub = df_m15[m15_cols].copy()

        # Rename to avoid collision with M5 columns
        rename_map = {c: f"m15_{c}" for c in df_m15_sub.columns if c != "datetime"}
        df_m15_sub = df_m15_sub.rename(columns=rename_map)
        df_m15_sub = df_m15_sub.rename(columns={"datetime": "m15_datetime_key"})

        df_merged = pd.merge_asof(
            df_m5,
            df_m15_sub,
            left_on="datetime",
            right_on="m15_datetime_key",
            direction="backward",
        )
        df_merged = df_merged.rename(columns={"m15_datetime_key": "m15_datetime"})

        return df_merged

    def compute_shared(self, df: pd.DataFrame, timeframe: str = "M5") -> pd.DataFrame:
        """
        Compute all shared features needed by every strategy.
        Call this FIRST before any strategy-specific features.

        Args:
            df: DataFrame with columns [datetime, open, high, low, close, volume]
            timeframe: Timeframe label for context-aware features

        Returns:
            DataFrame with shared features added
        """
        df = df.copy()

        # Ensure datetime is proper type
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])

        # Sort by time
        df = df.sort_values("datetime").reset_index(drop=True)

        # ─── Core Price Features ─────────────────────
        df = self._add_candle_features(df)

        # ─── Moving Averages ─────────────────────────
        df = self._add_emas(df)

        # ─── ATR ─────────────────────────────────────
        df = self._add_atr(df)

        # ─── RSI ─────────────────────────────────────
        df = self._add_rsi(df)

        # ─── ADX ─────────────────────────────────────
        df = self._add_adx(df)

        # ─── Session Labels ──────────────────────────
        df = self._add_sessions(df)

        # ─── Time Features ───────────────────────────
        df = self._add_time_features(df)

        # ─── Swing Points ────────────────────────────
        df = self._add_swing_points(df)

        # ─── Volatility Regime ───────────────────────
        df = self._add_volatility_regime(df)

        # ─── EMA Distances ───────────────────────────
        df = self._add_ema_distances(df)

        # Store timeframe
        df["timeframe"] = timeframe

        return df

    # ─── Phase 1: FVG Features ───────────────────────
    def compute_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 1: Detect Fair Value Gaps on this DataFrame.
        Best used on M15 data for structure detection.

        Adds columns:
          - fvg_bull_top, fvg_bull_bottom: bullish FVG zone boundaries
          - fvg_bear_top, fvg_bear_bottom: bearish FVG zone boundaries
          - fvg_bull_ce, fvg_bear_ce: consequent encroachment (50% level)
          - fvg_bull_size, fvg_bear_size: gap size in price
          - fvg_bull_size_atr, fvg_bear_size_atr: gap size normalized by ATR
          - fvg_detected: 1 if any FVG detected on this candle, 0 otherwise
          - fvg_direction: "bull", "bear", or "none"

        FVG rules:
          - Bullish: candle_1_high < candle_3_low (gap up)
          - Bearish: candle_1_low > candle_3_high (gap down)
          - Min size: 0.5x ATR(14)
          - Max size: 2.5x ATR(14)
        """
        df = df.copy()
        atr_col = f"atr_{ATR_PERIOD}"

        # Initialize columns
        for prefix in ["bull", "bear"]:
            for suffix in ["top", "bottom", "ce", "size", "size_atr"]:
                df[f"fvg_{prefix}_{suffix}"] = np.nan

        df["fvg_detected"] = 0
        df["fvg_direction"] = "none"

        highs = df["high"].values
        lows = df["low"].values
        atr_vals = df[atr_col].values if atr_col in df.columns else np.full(len(df), 2.0)

        for i in range(2, len(df)):
            atr = atr_vals[i]
            if np.isnan(atr) or atr <= 0:
                continue

            min_size = 0.5 * atr
            max_size = 2.5 * atr

            # ── Bullish FVG: candle[i-2] high < candle[i] low ──
            c1_high = highs[i - 2]
            c3_low = lows[i]

            if c3_low > c1_high:
                gap_size = c3_low - c1_high
                if min_size <= gap_size <= max_size:
                    df.iloc[i, df.columns.get_loc("fvg_bull_bottom")] = c1_high
                    df.iloc[i, df.columns.get_loc("fvg_bull_top")] = c3_low
                    df.iloc[i, df.columns.get_loc("fvg_bull_ce")] = c1_high + gap_size / 2
                    df.iloc[i, df.columns.get_loc("fvg_bull_size")] = gap_size
                    df.iloc[i, df.columns.get_loc("fvg_bull_size_atr")] = gap_size / atr
                    df.iloc[i, df.columns.get_loc("fvg_detected")] = 1
                    df.iloc[i, df.columns.get_loc("fvg_direction")] = "bull"

            # ── Bearish FVG: candle[i-2] low > candle[i] high ──
            c1_low = lows[i - 2]
            c3_high = highs[i]

            if c1_low > c3_high:
                gap_size = c1_low - c3_high
                if min_size <= gap_size <= max_size:
                    df.iloc[i, df.columns.get_loc("fvg_bear_top")] = c1_low
                    df.iloc[i, df.columns.get_loc("fvg_bear_bottom")] = c3_high
                    df.iloc[i, df.columns.get_loc("fvg_bear_ce")] = c3_high + gap_size / 2
                    df.iloc[i, df.columns.get_loc("fvg_bear_size")] = gap_size
                    df.iloc[i, df.columns.get_loc("fvg_bear_size_atr")] = gap_size / atr
                    df.iloc[i, df.columns.get_loc("fvg_detected")] = 1
                    # If both bull and bear detected same candle (very rare), bear overwrites
                    if df.iloc[i, df.columns.get_loc("fvg_direction")] == "bull":
                        df.iloc[i, df.columns.get_loc("fvg_direction")] = "both"
                    else:
                        df.iloc[i, df.columns.get_loc("fvg_direction")] = "bear"

        # Count of FVGs detected (summary stat)
        total_bull = (df["fvg_direction"].isin(["bull", "both"])).sum()
        total_bear = (df["fvg_direction"].isin(["bear", "both"])).sum()
        print(f"    FVG detected: {total_bull} bullish, {total_bear} bearish")

        return df

    # ─── Phase 2: Trend Features ─────────────────────
    def compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 2: Trend continuation features for Strategy D.
        Most core indicators (EMAs, RSI, ADX) are already in shared features.
        This adds Strategy D-specific derived features.

        Adds columns:
          - ema21_touch_bull: price came within 1x ATR of EMA21 from above (buy pullback)
          - ema21_touch_bear: price came within 1x ATR of EMA21 from below (sell pullback)
          - bounce_candle_bull: M5 closes bullish after touching EMA zone
          - bounce_candle_bear: M5 closes bearish after touching EMA zone
          - pullback_depth_atr: how deep the pullback is in ATR units
          - trend_quality: composite score 0-4 counting alignment factors
          - ema9_21_cross_bull: EMA9 just crossed above EMA21 (avoid, trend just starting)
          - ema9_21_cross_bear: EMA9 just crossed below EMA21
          - ema21_touch_count: rolling count of EMA21 touches in last 50 candles
        """
        df = df.copy()
        atr_col = f"atr_{ATR_PERIOD}"

        # ── EMA21 Touch Detection ────────────────────────
        # Price within 1x ATR of EMA21
        if "ema_21" in df.columns and atr_col in df.columns:
            dist_to_ema21 = (df["close"] - df["ema_21"]).abs()
            within_atr = dist_to_ema21 <= df[atr_col]

            # Bull touch: price is above EMA21 but close to it (pulling back in uptrend)
            df["ema21_touch_bull"] = (
                within_atr &
                (df["close"] >= df["ema_21"]) &
                (df["low"] <= df["ema_21"] + df[atr_col])
            ).astype(int)

            # Bear touch: price is below EMA21 but close to it (pulling back in downtrend)
            df["ema21_touch_bear"] = (
                within_atr &
                (df["close"] <= df["ema_21"]) &
                (df["high"] >= df["ema_21"] - df[atr_col])
            ).astype(int)

            # Rolling touch count (how many times EMA21 was touched in last 50 candles)
            # First/second touch = strong, third+ = trend weakening
            df["ema21_touch_count"] = (
                (df["ema21_touch_bull"] | df["ema21_touch_bear"])
                .rolling(50, min_periods=1).sum()
            )
        else:
            df["ema21_touch_bull"] = 0
            df["ema21_touch_bear"] = 0
            df["ema21_touch_count"] = 0

        # ── Bounce Candle Detection ──────────────────────
        # After touching EMA zone, candle closes back in trend direction
        prev_touch_bull = df["ema21_touch_bull"].shift(1).fillna(0)
        prev_touch_bear = df["ema21_touch_bear"].shift(1).fillna(0)

        df["bounce_candle_bull"] = (
            (prev_touch_bull == 1) &                # Previous candle touched EMA21 zone
            (df["close"] > df["open"]) &            # Current candle is bullish
            (df["close"] > df["ema_21"])             # Closed above EMA21
        ).astype(int)

        df["bounce_candle_bear"] = (
            (prev_touch_bear == 1) &
            (df["close"] < df["open"]) &
            (df["close"] < df["ema_21"])
        ).astype(int)

        # Also accept engulfing as bounce
        if "is_engulfing_bull" in df.columns:
            df["bounce_candle_bull"] = df["bounce_candle_bull"] | (
                (prev_touch_bull == 1) & (df["is_engulfing_bull"] == 1)
            ).astype(int)
        if "is_engulfing_bear" in df.columns:
            df["bounce_candle_bear"] = df["bounce_candle_bear"] | (
                (prev_touch_bear == 1) & (df["is_engulfing_bear"] == 1)
            ).astype(int)

        # ── Pullback Depth ───────────────────────────────
        # How far price has pulled back from recent swing, measured in ATR
        if "last_swing_high" in df.columns and atr_col in df.columns:
            df["pullback_depth_bull_atr"] = (
                (df["last_swing_high"] - df["low"]) / df[atr_col].replace(0, np.nan)
            )
            df["pullback_depth_bear_atr"] = (
                (df["high"] - df["last_swing_low"]) / df[atr_col].replace(0, np.nan)
            )
        else:
            df["pullback_depth_bull_atr"] = 0
            df["pullback_depth_bear_atr"] = 0

        # ── Trend Quality Score ──────────────────────────
        # Composite 0-4: how many trend factors align
        df["trend_quality_bull"] = 0
        df["trend_quality_bear"] = 0

        if "ema_partial_bull" in df.columns:
            df["trend_quality_bull"] += df["ema_partial_bull"]  # EMA 9>21>50
        if "above_ema200" in df.columns:
            df["trend_quality_bull"] += df["above_ema200"]      # Above EMA200
        if "trend_strong" in df.columns:
            df["trend_quality_bull"] += df["trend_strong"]      # ADX > 25
        if "rsi_pullback_buy" in df.columns:
            df["trend_quality_bull"] += df["rsi_pullback_buy"]  # RSI in buy zone

        if "ema_partial_bear" in df.columns:
            df["trend_quality_bear"] += df["ema_partial_bear"]
        if "above_ema200" in df.columns:
            df["trend_quality_bear"] += (1 - df["above_ema200"])  # Below EMA200
        if "trend_strong" in df.columns:
            df["trend_quality_bear"] += df["trend_strong"]
        if "rsi_pullback_sell" in df.columns:
            df["trend_quality_bear"] += df["rsi_pullback_sell"]

        # ── EMA9/21 Crossover Detection (avoid signal) ──
        # If EMA9 just crossed EMA21, trend is brand new — risky
        if "ema_9" in df.columns and "ema_21" in df.columns:
            
            ema9_above = df["ema_9"] > df["ema_21"]
            prev_ema9_above = ema9_above.shift(1, fill_value=False)
            df["ema9_21_cross_bull"] = (ema9_above & ~prev_ema9_above).astype(int)
            df["ema9_21_cross_bear"] = (~ema9_above & prev_ema9_above).astype(int)
            # df["ema9_21_cross_bull"] = (ema9_above & ~ema9_above.shift(1).fillna(False)).infer_objects(copy=False).astype(int)
            # df["ema9_21_cross_bear"] = (~ema9_above & ema9_above.shift(1).fillna(True)).infer_objects(copy=False).astype(int)

            # Rolling: how many candles since last cross (trend maturity)
            cross_any = df["ema9_21_cross_bull"] | df["ema9_21_cross_bear"]
            df["candles_since_ema_cross"] = cross_any.groupby(
                cross_any.cumsum()
            ).cumcount()
        else:
            df["ema9_21_cross_bull"] = 0
            df["ema9_21_cross_bear"] = 0
            df["candles_since_ema_cross"] = 0

        # Summary
        bull_bounces = df["bounce_candle_bull"].sum()
        bear_bounces = df["bounce_candle_bear"].sum()
        print(f"    Trend features: {bull_bounces} bull bounces, {bear_bounces} bear bounces detected")

        return df

    # ─── Phase 3: Session Range Features ─────────────
    def compute_session_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 3: Session Range Breakout features.
        Computes Asian session range per trading day and marks breakouts.

        Broker timezone: UTC+2
        Asian range hours: 02:00-08:00 broker time (= 00:00-06:00 UTC)
        Breakout allowed: 08:00-18:00 broker time (= 06:00-16:00 UTC)
        Early breakout: before 14:00 broker time (= before 12:00 UTC)

        Adds columns:
          - asian_range_high, asian_range_low, asian_range_width
          - asian_range_mid: midpoint of range
          - asian_range_width_atr: width normalized by ATR
          - asian_range_valid: range meets size criteria (0.3x-1.5x ATR)
          - in_breakout_window: current candle is in valid breakout hours
          - is_early_breakout_window: before 14:00 broker time
          - breakout_bull: M5 close above range high
          - breakout_bear: M5 close below range low
        """
        df = df.copy()
        atr_col = f"atr_{ATR_PERIOD}"

        # Broker time hours (data timestamps are UTC+2)
        ASIAN_START = 2   # 02:00 broker = 00:00 UTC
        ASIAN_END = 8     # 08:00 broker = 06:00 UTC
        BREAKOUT_START = 8   # 08:00 broker
        BREAKOUT_END = 18    # 18:00 broker = 16:00 UTC
        EARLY_CUTOFF = 14    # 14:00 broker = 12:00 UTC

        hour = df["datetime"].dt.hour
        date = df["datetime"].dt.date

        # ── Compute Asian range per trading day ──────────
        # Asian session mask
        asian_mask = (hour >= ASIAN_START) & (hour < ASIAN_END)

        # Group by date to get daily Asian range
        df["_trade_date"] = date
        asian_data = df[asian_mask].groupby("_trade_date").agg(
            asian_high=("high", "max"),
            asian_low=("low", "min"),
            asian_candles=("high", "count"),
        ).reset_index()

        # Compute range metrics
        asian_data["asian_range_width"] = asian_data["asian_high"] - asian_data["asian_low"]
        asian_data["asian_range_mid"] = (asian_data["asian_high"] + asian_data["asian_low"]) / 2

        # Merge back into main df
        df = df.merge(
            asian_data.rename(columns={
                "_trade_date": "_trade_date",
                "asian_high": "asian_range_high",
                "asian_low": "asian_range_low",
                "asian_candles": "asian_session_candles",
            }),
            on="_trade_date",
            how="left",
        )

        # Forward fill for candles that don't have a range yet (early Asian)
        # Actually the range should only be available AFTER Asian session ends
        # So null it out during Asian session
        df.loc[asian_mask, "asian_range_high"] = np.nan
        df.loc[asian_mask, "asian_range_low"] = np.nan
        df.loc[asian_mask, "asian_range_width"] = np.nan
        df.loc[asian_mask, "asian_range_mid"] = np.nan

        # ── Compute Daily ATR from M5 data ─────────────
        # M5 ATR measures 5-minute volatility — wrong scale for 6-hour range comparison
        # Daily ATR = rolling 14-day average of daily high-low range
        daily_hl = df.groupby("_trade_date").agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
        )
        daily_hl["daily_range"] = daily_hl["day_high"] - daily_hl["day_low"]
        daily_hl["daily_atr_14"] = daily_hl["daily_range"].ewm(span=14, adjust=False).mean()
        daily_hl = daily_hl[["daily_atr_14"]].reset_index()

        df = df.merge(daily_hl, on="_trade_date", how="left")

        # ── Range validity check ─────────────────────────
        # Compare Asian range width to DAILY ATR (correct scale)
        df["asian_range_width_atr"] = df["asian_range_width"] / df["daily_atr_14"].replace(0, np.nan)
        df["asian_range_valid"] = (
            (df["asian_range_width_atr"] >= 0.1) &
            (df["asian_range_width_atr"] <= 0.7) &
            (df["asian_session_candles"] >= 5)
        ).astype(int)

        # ── Time window flags ────────────────────────────
        df["in_breakout_window"] = (
            (hour >= BREAKOUT_START) & (hour < BREAKOUT_END)
        ).astype(int)

        df["is_early_breakout_window"] = (
            (hour >= BREAKOUT_START) & (hour < EARLY_CUTOFF)
        ).astype(int)

        # ── Breakout detection ───────────────────────────
        # M5 close above/below range
        df["breakout_bull"] = (
            df["in_breakout_window"].astype(bool) &
            df["asian_range_valid"].astype(bool) &
            (df["close"] > df["asian_range_high"])
        ).astype(int)

        df["breakout_bear"] = (
            df["in_breakout_window"].astype(bool) &
            df["asian_range_valid"].astype(bool) &
            (df["close"] < df["asian_range_low"])
        ).astype(int)

        # ── Friday flag ──────────────────────────────────
        df["is_friday"] = (df["datetime"].dt.dayofweek == 4).astype(int)

        # ── Distance from range edges ────────────────────
        df["dist_to_range_high"] = df["close"] - df["asian_range_high"]
        df["dist_to_range_low"] = df["close"] - df["asian_range_low"]

        # Cleanup
        df = df.drop(columns=["_trade_date"], errors="ignore")

        # Summary
        bull_breakouts = df["breakout_bull"].sum()
        bear_breakouts = df["breakout_bear"].sum()
        valid_days = df[df["asian_range_valid"] == 1]["datetime"].dt.date.nunique()
        print(f"    Session range: {valid_days} valid range days, {bull_breakouts} bull breakouts, {bear_breakouts} bear breakouts")

        return df

    # ─── Phase 4: SMC Features ───────────────────────
    def compute_smc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 4: Smart Money Concepts features for Strategy A.
        Detects Break of Structure (BOS), equal highs/lows (liquidity pools),
        and Order Block zones on M15 data.

        Adds columns:
          - bos_bull: bullish BOS detected (close breaks above recent swing high)
          - bos_bear: bearish BOS detected (close breaks below recent swing low)
          - bos_bull_level: the swing high level that was broken
          - bos_bear_level: the swing low level that was broken
          - equal_highs: 1 if 2+ swing highs within tolerance at this level
          - equal_lows: 1 if 2+ swing lows within tolerance at this level
          - equal_highs_level: price level of the equal highs cluster
          - equal_lows_level: price level of the equal lows cluster
          - equal_highs_count: number of highs in the cluster
          - equal_lows_count: number of lows in the cluster
          - ob_bull_high, ob_bull_low: bullish OB zone (last bearish candle before bullish BOS)
          - ob_bear_high, ob_bear_low: bearish OB zone (last bullish candle before bearish BOS)
          - ob_bull_impulse_size: size of the impulse that caused bullish BOS
          - ob_bear_impulse_size: size of the impulse that caused bearish BOS
        """
        df = df.copy()
        atr_col = f"atr_{ATR_PERIOD}"
        n = SWING_LOOKBACK

        # Initialize all columns
        for col in ["bos_bull", "bos_bear"]:
            df[col] = 0
        for col in ["bos_bull_level", "bos_bear_level",
                     "ob_bull_high", "ob_bull_low", "ob_bear_high", "ob_bear_low",
                     "ob_bull_impulse_size", "ob_bear_impulse_size",
                     "equal_highs_level", "equal_lows_level"]:
            df[col] = np.nan
        df["equal_highs"] = 0
        df["equal_lows"] = 0
        df["equal_highs_count"] = 0
        df["equal_lows_count"] = 0

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        atr_vals = df[atr_col].values if atr_col in df.columns else np.full(len(df), 2.0)
        swing_high_prices = df["swing_high_price"].values if "swing_high_price" in df.columns else np.full(len(df), np.nan)
        swing_low_prices = df["swing_low_price"].values if "swing_low_price" in df.columns else np.full(len(df), np.nan)

        # Collect all swing highs and lows with their indices
        swing_highs = []  # [(index, price), ...]
        swing_lows = []

        for i in range(len(df)):
            if not np.isnan(swing_high_prices[i]):
                swing_highs.append((i, swing_high_prices[i]))
            if not np.isnan(swing_low_prices[i]):
                swing_lows.append((i, swing_low_prices[i]))

        # ── BOS Detection + OB Identification ────────────
        # Track the most recent unbroken swing high/low
        last_swing_high_price = np.nan
        last_swing_high_idx = -1
        last_swing_low_price = np.nan
        last_swing_low_idx = -1

        bos_bull_count = 0
        bos_bear_count = 0
        ob_bull_count = 0
        ob_bear_count = 0

        for i in range(n + 1, len(df)):
            atr = atr_vals[i]
            if np.isnan(atr) or atr <= 0:
                continue

            # Update last swing high/low
            if not np.isnan(swing_high_prices[i]):
                last_swing_high_price = swing_high_prices[i]
                last_swing_high_idx = i
            if not np.isnan(swing_low_prices[i]):
                last_swing_low_price = swing_low_prices[i]
                last_swing_low_idx = i

            # ── Bullish BOS: close breaks above last swing high ──
            if (not np.isnan(last_swing_high_price) and
                closes[i] > last_swing_high_price and
                closes[i - 1] <= last_swing_high_price):

                df.iat[i, df.columns.get_loc("bos_bull")] = 1
                df.iat[i, df.columns.get_loc("bos_bull_level")] = last_swing_high_price
                bos_bull_count += 1

                # Find OB: last BEARISH candle body before this impulse
                # Walk backwards from current candle to find the impulse start
                ob_found = False
                for j in range(i - 1, max(i - 30, 0), -1):
                    # Look for last bearish candle (close < open)
                    if closes[j] < opens[j]:
                        body_size = abs(opens[j] - closes[j])
                        if body_size >= 0.2 * atr:  # Min body size
                            # OB zone = candle body (for bearish: high=open, low=close)
                            df.iat[i, df.columns.get_loc("ob_bull_high")] = opens[j]
                            df.iat[i, df.columns.get_loc("ob_bull_low")] = closes[j]
                            df.iat[i, df.columns.get_loc("ob_bull_impulse_size")] = closes[i] - closes[j]
                            ob_bull_count += 1
                            ob_found = True
                            break
                    # If we hit a bullish candle that's part of the impulse, keep going
                    # But if we've gone past 30 candles, give up

                # Invalidate this swing high so it can't trigger again
                last_swing_high_price = closes[i]
                last_swing_high_idx = i

            # ── Bearish BOS: close breaks below last swing low ──
            if (not np.isnan(last_swing_low_price) and
                closes[i] < last_swing_low_price and
                closes[i - 1] >= last_swing_low_price):

                df.iat[i, df.columns.get_loc("bos_bear")] = 1
                df.iat[i, df.columns.get_loc("bos_bear_level")] = last_swing_low_price
                bos_bear_count += 1

                # Find OB: last BULLISH candle body before this impulse
                ob_found = False
                for j in range(i - 1, max(i - 30, 0), -1):
                    if closes[j] > opens[j]:
                        body_size = abs(closes[j] - opens[j])
                        if body_size >= 0.2 * atr:
                            # OB zone = candle body (for bullish: high=close, low=open)
                            df.iat[i, df.columns.get_loc("ob_bear_high")] = closes[j]
                            df.iat[i, df.columns.get_loc("ob_bear_low")] = opens[j]
                            df.iat[i, df.columns.get_loc("ob_bear_impulse_size")] = closes[j] - closes[i]
                            ob_bear_count += 1
                            ob_found = True
                            break

                last_swing_low_price = closes[i]
                last_swing_low_idx = i

        # ── Equal Highs/Lows Detection (Liquidity Pools) ──
        eq_high_count = 0
        eq_low_count = 0
        lookback_candles = 100  # Look within 100 M15 candles

        for i in range(len(swing_highs)):
            idx_i, price_i = swing_highs[i]
            atr = atr_vals[idx_i] if not np.isnan(atr_vals[idx_i]) else 2.0
            tolerance = 0.3 * atr
            cluster = [(idx_i, price_i)]

            for j in range(i + 1, len(swing_highs)):
                idx_j, price_j = swing_highs[j]
                if idx_j - idx_i > lookback_candles:
                    break
                if abs(price_j - price_i) <= tolerance:
                    cluster.append((idx_j, price_j))

            if len(cluster) >= 2:
                # Mark the LAST swing in the cluster
                last_idx = cluster[-1][0]
                avg_price = np.mean([p for _, p in cluster])
                df.iat[last_idx, df.columns.get_loc("equal_highs")] = 1
                df.iat[last_idx, df.columns.get_loc("equal_highs_level")] = avg_price
                df.iat[last_idx, df.columns.get_loc("equal_highs_count")] = len(cluster)
                eq_high_count += 1

        for i in range(len(swing_lows)):
            idx_i, price_i = swing_lows[i]
            atr = atr_vals[idx_i] if not np.isnan(atr_vals[idx_i]) else 2.0
            tolerance = 0.3 * atr
            cluster = [(idx_i, price_i)]

            for j in range(i + 1, len(swing_lows)):
                idx_j, price_j = swing_lows[j]
                if idx_j - idx_i > lookback_candles:
                    break
                if abs(price_j - price_i) <= tolerance:
                    cluster.append((idx_j, price_j))

            if len(cluster) >= 2:
                last_idx = cluster[-1][0]
                avg_price = np.mean([p for _, p in cluster])
                df.iat[last_idx, df.columns.get_loc("equal_lows")] = 1
                df.iat[last_idx, df.columns.get_loc("equal_lows_level")] = avg_price
                df.iat[last_idx, df.columns.get_loc("equal_lows_count")] = len(cluster)
                eq_low_count += 1

        # Forward fill equal highs/lows levels so every candle knows the nearest pool
        df["last_equal_highs_level"] = df["equal_highs_level"].ffill()
        df["last_equal_lows_level"] = df["equal_lows_level"].ffill()

        print(f"    SMC features: {bos_bull_count} bull BOS, {bos_bear_count} bear BOS, "
              f"{ob_bull_count} bull OBs, {ob_bear_count} bear OBs, "
              f"{eq_high_count} equal highs clusters, {eq_low_count} equal lows clusters")

        return df

    # ═════════════════════════════════════════════════
    # PRIVATE METHODS — Shared Feature Implementations
    # ═════════════════════════════════════════════════

    def _add_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic candle-derived features."""
        # Body and wick measurements
        df["body"] = df["close"] - df["open"]
        df["body_abs"] = df["body"].abs()
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["candle_range"] = df["high"] - df["low"]

        # Direction
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)

        # Body ratio (body size relative to total range)
        df["body_ratio"] = np.where(
            df["candle_range"] > 0,
            df["body_abs"] / df["candle_range"],
            0
        )

        # Candle patterns (simple)
        # Doji: body < 10% of range
        df["is_doji"] = (df["body_ratio"] < 0.1).astype(int)

        # Pin bar: one wick > 60% of range, body < 30%
        df["is_pin_bar_bull"] = (
            (df["lower_wick"] > 0.6 * df["candle_range"]) &
            (df["body_ratio"] < 0.3)
        ).astype(int)

        df["is_pin_bar_bear"] = (
            (df["upper_wick"] > 0.6 * df["candle_range"]) &
            (df["body_ratio"] < 0.3)
        ).astype(int)

        # Engulfing (current candle body fully covers previous candle body)
        prev_open = df["open"].shift(1)
        prev_close = df["close"].shift(1)
        prev_body_high = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
        prev_body_low = pd.concat([prev_open, prev_close], axis=1).min(axis=1)
        curr_body_high = pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
        curr_body_low = pd.concat([df["open"], df["close"]], axis=1).min(axis=1)

        df["is_engulfing_bull"] = (
            (df["is_bullish"] == 1) &
            (df["is_bullish"].shift(1) == 0) &
            (curr_body_high > prev_body_high) &
            (curr_body_low < prev_body_low)
        ).astype(int)

        df["is_engulfing_bear"] = (
            (df["is_bullish"] == 0) &
            (df["is_bullish"].shift(1) == 1) &
            (curr_body_high > prev_body_high) &
            (curr_body_low < prev_body_low)
        ).astype(int)

        return df

    def _add_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exponential Moving Averages."""
        for period in EMA_PERIODS:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # EMA slopes (rate of change over last 3 candles)
        for period in EMA_PERIODS:
            df[f"ema_{period}_slope"] = df[f"ema_{period}"].diff(3) / 3

        # EMA stack alignment
        # 1 = perfect bullish (9 > 21 > 50 > 200)
        # -1 = perfect bearish (9 < 21 < 50 < 200)
        # 0 = mixed
        if all(p in EMA_PERIODS for p in [9, 21, 50, 200]):
            bull_stack = (
                (df["ema_9"] > df["ema_21"]) &
                (df["ema_21"] > df["ema_50"]) &
                (df["ema_50"] > df["ema_200"])
            )
            bear_stack = (
                (df["ema_9"] < df["ema_21"]) &
                (df["ema_21"] < df["ema_50"]) &
                (df["ema_50"] < df["ema_200"])
            )
            df["ema_stack"] = 0
            df.loc[bull_stack, "ema_stack"] = 1
            df.loc[bear_stack, "ema_stack"] = -1

            # Partial alignment (9 > 21 > 50 but not necessarily > 200)
            df["ema_partial_bull"] = (
                (df["ema_9"] > df["ema_21"]) &
                (df["ema_21"] > df["ema_50"])
            ).astype(int)

            df["ema_partial_bear"] = (
                (df["ema_9"] < df["ema_21"]) &
                (df["ema_21"] < df["ema_50"])
            ).astype(int)

        # Price position relative to EMA 200
        if 200 in EMA_PERIODS:
            df["above_ema200"] = (df["close"] > df["ema_200"]).astype(int)

        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range."""
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f"atr_{ATR_PERIOD}"] = df["true_range"].ewm(span=ATR_PERIOD, adjust=False).mean()

        # ATR in pips (for gold: 1 pip = $0.10)
        df[f"atr_{ATR_PERIOD}_pips"] = df[f"atr_{ATR_PERIOD}"] / PIP_VALUE

        # Normalized ATR (ATR / close price — useful for regime detection)
        df["atr_normalized"] = df[f"atr_{ATR_PERIOD}"] / df["close"]

        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index."""
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{RSI_PERIOD}"] = 100 - (100 / (1 + rs))

        # RSI zones
        df["rsi_overbought"] = (df[f"rsi_{RSI_PERIOD}"] > 70).astype(int)
        df["rsi_oversold"] = (df[f"rsi_{RSI_PERIOD}"] < 30).astype(int)

        # RSI in pullback zone (40-55 for buys, 45-60 for sells)
        df["rsi_pullback_buy"] = (
            (df[f"rsi_{RSI_PERIOD}"] >= 40) & (df[f"rsi_{RSI_PERIOD}"] <= 55)
        ).astype(int)

        df["rsi_pullback_sell"] = (
            (df[f"rsi_{RSI_PERIOD}"] >= 45) & (df[f"rsi_{RSI_PERIOD}"] <= 60)
        ).astype(int)

        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average Directional Index — trend strength indicator."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        # Smoothed with EMA
        atr = df[f"atr_{ATR_PERIOD}"] if f"atr_{ATR_PERIOD}" in df.columns else df["true_range"].ewm(span=ADX_PERIOD, adjust=False).mean()

        plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean()

        # +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)

        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        # DX and ADX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100 * di_diff / di_sum.replace(0, np.nan)

        df[f"adx_{ADX_PERIOD}"] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()

        # Trend strength labels
        df["trend_strong"] = (df[f"adx_{ADX_PERIOD}"] > 25).astype(int)
        df["trend_very_strong"] = (df[f"adx_{ADX_PERIOD}"] > 40).astype(int)
        df["trend_weak"] = (df[f"adx_{ADX_PERIOD}"] < 20).astype(int)

        return df

    def _add_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each candle with its trading session.
        Uses UTC hours from datetime column.
        
        NOTE: If your MT5 broker uses a different timezone (e.g., UTC+2),
        adjust the datetime before calling this, or modify SESSIONS in config.
        """
        hour = df["datetime"].dt.hour

        # Default to "off"
        df["session"] = "off"

        # Apply session labels (order matters — later overwrites earlier for overlaps)
        for session_name, times in SESSIONS.items():
            start_h = times["start"]
            end_h = times["end"]

            if start_h < end_h:
                mask = (hour >= start_h) & (hour < end_h)
            else:  # Wraps around midnight
                mask = (hour >= start_h) | (hour < end_h)

            df.loc[mask, "session"] = session_name

        # Handle London/NY overlap (12:00-14:00 UTC) — label as "london_ny_overlap"
        overlap_mask = (hour >= 12) & (hour < 14)
        df.loc[overlap_mask, "session"] = "london_ny_overlap"

        # One-hot encode sessions for model input
        for session_name in list(SESSIONS.keys()) + ["off", "london_ny_overlap"]:
            df[f"session_{session_name}"] = (df["session"] == session_name).astype(int)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features for cyclical patterns."""
        dt = df["datetime"]

        df["hour"] = dt.dt.hour
        df["minute"] = dt.dt.minute
        df["day_of_week"] = dt.dt.dayofweek  # 0=Monday, 4=Friday
        df["day_of_month"] = dt.dt.day
        df["month"] = dt.dt.month

        # Cyclical encoding (sin/cos) for hour — captures the circular nature
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week cyclical
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)

        # Is Friday (reduced follow-through, relevant for Strategy C)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        return df

    def _add_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect swing highs and swing lows.
        A swing high has a higher high than the N candles on each side.
        A swing low has a lower low than the N candles on each side.
        """
        n = SWING_LOOKBACK

        df["swing_high"] = 0
        df["swing_low"] = 0
        df["swing_high_price"] = np.nan
        df["swing_low_price"] = np.nan

        highs = df["high"].values
        lows = df["low"].values

        for i in range(n, len(df) - n):
            # Swing high: current high is highest in window
            left_highs = highs[i - n:i]
            right_highs = highs[i + 1:i + n + 1]

            if highs[i] > left_highs.max() and highs[i] > right_highs.max():
                df.iloc[i, df.columns.get_loc("swing_high")] = 1
                df.iloc[i, df.columns.get_loc("swing_high_price")] = highs[i]

            # Swing low: current low is lowest in window
            left_lows = lows[i - n:i]
            right_lows = lows[i + 1:i + n + 1]

            if lows[i] < left_lows.min() and lows[i] < right_lows.min():
                df.iloc[i, df.columns.get_loc("swing_low")] = 1
                df.iloc[i, df.columns.get_loc("swing_low_price")] = lows[i]

        # Forward-fill the most recent swing high/low prices
        # (so every candle knows where the last swing was)
        df["last_swing_high"] = df["swing_high_price"].ffill()
        df["last_swing_low"] = df["swing_low_price"].ffill()

        # Distance from current price to last swing
        df["dist_to_swing_high"] = df["close"] - df["last_swing_high"]
        df["dist_to_swing_low"] = df["close"] - df["last_swing_low"]

        return df

    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify current volatility regime based on ATR percentile.
        Uses rolling 100-period percentile rank of ATR.
        """
        atr_col = f"atr_{ATR_PERIOD}"
        if atr_col not in df.columns:
            return df

        # Rolling percentile rank (where does current ATR sit vs last 100 candles)
        window = 100
        df["atr_percentile"] = df[atr_col].rolling(window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )

        # Regime labels
        df["vol_regime"] = "normal"
        df.loc[df["atr_percentile"] > 0.8, "vol_regime"] = "high_volatility"
        df.loc[df["atr_percentile"] < 0.2, "vol_regime"] = "low_volatility"

        # One-hot for model input
        for regime in ["low_volatility", "normal", "high_volatility"]:
            df[f"vol_{regime}"] = (df["vol_regime"] == regime).astype(int)

        return df

    def _add_ema_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distance of price from each EMA, normalized by ATR."""
        atr_col = f"atr_{ATR_PERIOD}"
        if atr_col not in df.columns:
            return df

        for period in EMA_PERIODS:
            ema_col = f"ema_{period}"
            if ema_col in df.columns:
                # Raw distance
                df[f"dist_ema_{period}"] = df["close"] - df[ema_col]

                # Normalized by ATR (how many ATRs away from EMA)
                df[f"dist_ema_{period}_atr"] = (
                    df[f"dist_ema_{period}"] / df[atr_col].replace(0, np.nan)
                )

        return df


# ─── Utility Functions ───────────────────────────────────

def load_and_compute(filepath: str, timeframe: str = "M5") -> pd.DataFrame:
    """
    Convenience: load a raw CSV and compute all shared features.

    Args:
        filepath: Path to raw XAUUSD CSV file
        timeframe: Timeframe label

    Returns:
        DataFrame with all shared features computed
    """
    df = pd.read_csv(filepath, parse_dates=["datetime"])

    engine = FeatureEngine()
    df = engine.compute_shared(df, timeframe=timeframe)

    return df


def process_all_timeframes(raw_dir: str = None, processed_dir: str = None):
    """
    Process all raw timeframe CSVs and save with features.

    Args:
        raw_dir: Directory with raw CSVs (default: config DATA_RAW)
        processed_dir: Output directory (default: config DATA_PROCESSED)
    """
    from config import DATA_RAW, DATA_PROCESSED, MT5_TIMEFRAMES

    raw_dir = raw_dir or DATA_RAW
    processed_dir = processed_dir or DATA_PROCESSED

    engine = FeatureEngine()

    for tf_name in MT5_TIMEFRAMES:
        raw_file = Path(raw_dir) / f"XAUUSD_{tf_name}.csv"
        if not raw_file.exists():
            print(f"  {tf_name}: Raw file not found, skipping")
            continue

        print(f"  Processing {tf_name}...", end=" ")
        df = pd.read_csv(raw_file, parse_dates=["datetime"])
        df = engine.compute_shared(df, timeframe=tf_name)

        output_file = Path(processed_dir) / f"XAUUSD_{tf_name}_features.csv"
        df.to_csv(output_file, index=False)

        print(f"{len(df):,} candles, {len(df.columns)} features → {output_file.name}")

    print("\n✅ All timeframes processed!")


if __name__ == "__main__":
    print("=" * 60)
    print("MIDAS v2 — Feature Engineering (Phase 0: Shared Features)")
    print("=" * 60)
    process_all_timeframes()
    
    
# files changed in this update:
# main.py
# feature_engineering.py
# excutor.py
# tsl_manager.py [new]
# notifier.py [new]
# logger.py
# daily_limit.py
# risk_engine.py
# 