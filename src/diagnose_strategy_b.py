"""
MIDAS v2 — Strategy B Diagnostic
Finds exactly where signals are being killed.
Run this INSTEAD of run_strategy_b.py to see the funnel.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_RAW, DATA_PROCESSED, ATR_PERIOD, SPREAD_SIMULATION_PIPS, PIP_VALUE
from feature_engineering import FeatureEngine
from run_strategy_b import load_and_prepare_data, merge_timeframes
from strategies.strategy_b import FVGTracker


def diagnose():
    # Load the same merged data
    df = load_and_prepare_data()

    print("\n" + "=" * 60)
    print("  STRATEGY B — DIAGNOSTIC")
    print("=" * 60)

    # ─── Check 1: Are M15 columns actually in the merged data? ───
    m15_cols = [c for c in df.columns if c.startswith("m15_")]
    print(f"\n[CHECK 1] M15 columns in merged data: {len(m15_cols)}")
    fvg_m15_cols = [c for c in m15_cols if "fvg" in c]
    print(f"  FVG-specific M15 columns: {fvg_m15_cols}")

    # Sample a row that has FVG data
    fvg_rows = df[df["m15_fvg_detected"] == 1] if "m15_fvg_detected" in df.columns else pd.DataFrame()
    print(f"  M5 rows with M15 FVG detected: {len(fvg_rows):,}")
    if len(fvg_rows) > 0:
        sample = fvg_rows.iloc[0]
        print(f"  Sample FVG row datetime: {sample['datetime']}")
        print(f"    m15_fvg_bull_top: {sample.get('m15_fvg_bull_top', 'MISSING')}")
        print(f"    m15_fvg_bull_bottom: {sample.get('m15_fvg_bull_bottom', 'MISSING')}")
        print(f"    m15_fvg_bull_ce: {sample.get('m15_fvg_bull_ce', 'MISSING')}")
        print(f"    m15_fvg_bear_top: {sample.get('m15_fvg_bear_top', 'MISSING')}")
        print(f"    m15_fvg_direction: {sample.get('m15_fvg_direction', 'MISSING')}")
    else:
        print("  ❌ NO M5 rows have M15 FVG data! Merge might be broken.")
        return

    # ─── Check 2: Simulate FVG tracker behavior ─────────────
    print(f"\n[CHECK 2] Simulating FVG tracker...")
    tracker = FVGTracker()
    last_m15_time = None
    fvgs_added = 0
    fvgs_expired = 0
    fvgs_filled = 0

    # Track how many M5 candles see an active FVG
    candles_with_active_bull_fvg = 0
    candles_with_active_bear_fvg = 0

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Update tracker from M15
        m15_time = row.get("m15_datetime")
        if m15_time is not None and not pd.isna(m15_time) and m15_time != last_m15_time:
            last_m15_time = m15_time

            # Build M15 data dict
            m15_data = {}
            for col in row.index:
                if col.startswith("m15_"):
                    m15_data[col[4:]] = row[col]

            m15_series = pd.Series(m15_data)
            m15_idx = int(row.get("m15_idx", 0))

            old_bull = len([f for f in tracker.active_bull_fvgs if f.is_valid])
            old_bear = len([f for f in tracker.active_bear_fvgs if f.is_valid])

            tracker.update_from_m15(m15_series, m15_idx)

            new_bull = len([f for f in tracker.active_bull_fvgs if f.is_valid])
            new_bear = len([f for f in tracker.active_bear_fvgs if f.is_valid])

            if new_bull > old_bull or new_bear > old_bear:
                fvgs_added += 1

        # Update fills
        tracker.update_fills(row)

        # Count active FVGs
        valid_bull = [f for f in tracker.active_bull_fvgs if f.is_valid]
        valid_bear = [f for f in tracker.active_bear_fvgs if f.is_valid]
        if valid_bull:
            candles_with_active_bull_fvg += 1
        if valid_bear:
            candles_with_active_bear_fvg += 1

        # Print progress every 100k candles
        if idx % 100000 == 0 and idx > 0:
            print(f"    ...processed {idx:,} candles, "
                  f"active FVGs: {len(valid_bull)} bull, {len(valid_bear)} bear")

    print(f"  FVGs added to tracker: {fvgs_added}")
    print(f"  M5 candles with active bull FVG: {candles_with_active_bull_fvg:,} "
          f"({candles_with_active_bull_fvg/len(df)*100:.1f}%)")
    print(f"  M5 candles with active bear FVG: {candles_with_active_bear_fvg:,} "
          f"({candles_with_active_bear_fvg/len(df)*100:.1f}%)")

    # ─── Check 3: Condition-by-condition funnel ──────────────
    print(f"\n[CHECK 3] Condition funnel (checking every M5 candle)...")

    tracker2 = FVGTracker()
    last_m15_time2 = None

    counts = {
        "total_candles": 0,
        "has_bull_fvg": 0,
        "has_bear_fvg": 0,
        "bull_trend_aligned": 0,
        "bear_trend_aligned": 0,
        "bull_price_near_fvg": 0,
        "bear_price_near_fvg": 0,
        "bull_price_reaches_ce": 0,
        "bear_price_reaches_ce": 0,
        "bull_m5_rejection": 0,
        "bear_m5_rejection": 0,
        "bull_all_required": 0,
        "bear_all_required": 0,
    }

    for idx in range(2, len(df)):
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        counts["total_candles"] += 1

        # Update tracker
        m15_time = row.get("m15_datetime")
        if m15_time is not None and not pd.isna(m15_time) and m15_time != last_m15_time2:
            last_m15_time2 = m15_time
            m15_data = {}
            for col in row.index:
                if col.startswith("m15_"):
                    m15_data[col[4:]] = row[col]
            m15_series = pd.Series(m15_data)
            m15_idx = int(row.get("m15_idx", 0))
            tracker2.update_from_m15(m15_series, m15_idx)

        tracker2.update_fills(row)

        current_price = row["close"]

        # ── Check BULLISH ──
        bull_fvg = tracker2.get_best_bull_fvg(current_price)
        if bull_fvg is not None:
            counts["has_bull_fvg"] += 1

            # Trend aligned
            ema50 = row.get("ema_50", np.nan)
            ema50_slope = row.get("ema_50_slope", np.nan)
            trend_ok = False
            if not np.isnan(ema50):
                trend_ok = (current_price > ema50) or (
                    not np.isnan(ema50_slope) and ema50_slope > 0
                )
            if trend_ok:
                counts["bull_trend_aligned"] += 1

            # Price near FVG (within 2x ATR of the zone)
            atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
            if current_price - bull_fvg.top < 2 * atr:
                counts["bull_price_near_fvg"] += 1

            # Price reaches CE
            if row["low"] <= bull_fvg.ce:
                counts["bull_price_reaches_ce"] += 1

                # M5 rejection
                rejection = (
                    row["close"] > row["open"] and
                    row["close"] > bull_fvg.bottom and
                    prev["low"] <= bull_fvg.top
                )
                engulfing = (
                    row.get("is_engulfing_bull", 0) == 1 and
                    row["low"] <= bull_fvg.top and
                    row["close"] > bull_fvg.bottom
                )
                pin_bar = (
                    row.get("is_pin_bar_bull", 0) == 1 and
                    row["low"] <= bull_fvg.top and
                    row["close"] > bull_fvg.bottom
                )
                if rejection or engulfing or pin_bar:
                    counts["bull_m5_rejection"] += 1

                    if trend_ok:
                        counts["bull_all_required"] += 1

        # ── Check BEARISH ──
        bear_fvg = tracker2.get_best_bear_fvg(current_price)
        if bear_fvg is not None:
            counts["has_bear_fvg"] += 1

            ema50 = row.get("ema_50", np.nan)
            ema50_slope = row.get("ema_50_slope", np.nan)
            trend_ok = False
            if not np.isnan(ema50):
                trend_ok = (current_price < ema50) or (
                    not np.isnan(ema50_slope) and ema50_slope < 0
                )
            if trend_ok:
                counts["bear_trend_aligned"] += 1

            atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
            if bear_fvg.bottom - current_price < 2 * atr:
                counts["bear_price_near_fvg"] += 1

            if row["high"] >= bear_fvg.ce:
                counts["bear_price_reaches_ce"] += 1

                rejection = (
                    row["close"] < row["open"] and
                    row["close"] < bear_fvg.top and
                    prev["high"] >= bear_fvg.bottom
                )
                engulfing = (
                    row.get("is_engulfing_bear", 0) == 1 and
                    row["high"] >= bear_fvg.bottom and
                    row["close"] < bear_fvg.top
                )
                pin_bar = (
                    row.get("is_pin_bar_bear", 0) == 1 and
                    row["high"] >= bear_fvg.bottom and
                    row["close"] < bear_fvg.top
                )
                if rejection or engulfing or pin_bar:
                    counts["bear_m5_rejection"] += 1

                    if trend_ok:
                        counts["bear_all_required"] += 1

        if idx % 100000 == 0 and idx > 0:
            print(f"    ...processed {idx:,} candles")

    # ─── Print Funnel ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONDITION FUNNEL (out of {counts['total_candles']:,} M5 candles)")
    print(f"{'='*60}")

    print(f"\n  BULLISH SIDE:")
    print(f"  {'Has active bull FVG':<35} {counts['has_bull_fvg']:>10,}")
    print(f"  {'+ Trend aligned':<35} {counts['bull_trend_aligned']:>10,}")
    print(f"  {'+ Price near FVG (<2 ATR)':<35} {counts['bull_price_near_fvg']:>10,}")
    print(f"  {'+ Price reaches CE (50%)':<35} {counts['bull_price_reaches_ce']:>10,}")
    print(f"  {'+ M5 rejection candle':<35} {counts['bull_m5_rejection']:>10,}")
    print(f"  {'= ALL required conditions met':<35} {counts['bull_all_required']:>10,}")

    print(f"\n  BEARISH SIDE:")
    print(f"  {'Has active bear FVG':<35} {counts['has_bear_fvg']:>10,}")
    print(f"  {'+ Trend aligned':<35} {counts['bear_trend_aligned']:>10,}")
    print(f"  {'+ Price near FVG (<2 ATR)':<35} {counts['bear_price_near_fvg']:>10,}")
    print(f"  {'+ Price reaches CE (50%)':<35} {counts['bear_price_reaches_ce']:>10,}")
    print(f"  {'+ M5 rejection candle':<35} {counts['bear_m5_rejection']:>10,}")
    print(f"  {'= ALL required conditions met':<35} {counts['bear_all_required']:>10,}")

    total_signals = counts['bull_all_required'] + counts['bear_all_required']
    print(f"\n  TOTAL potential signals: {total_signals}")
    print(f"  (These still need to pass confidence threshold, cooldown, session limits, etc.)")

    # ─── Identify the bottleneck ─────────────────────────
    print(f"\n{'='*60}")
    print(f"  BOTTLENECK ANALYSIS")
    print(f"{'='*60}")

    bull_funnel = [
        ("Active FVG", counts["has_bull_fvg"]),
        ("Trend aligned", counts["bull_trend_aligned"]),
        ("Price reaches CE", counts["bull_price_reaches_ce"]),
        ("M5 rejection", counts["bull_m5_rejection"]),
        ("All required", counts["bull_all_required"]),
    ]

    print("\n  Bullish drop-off:")
    for i in range(1, len(bull_funnel)):
        prev_name, prev_count = bull_funnel[i-1]
        curr_name, curr_count = bull_funnel[i]
        if prev_count > 0:
            drop = (1 - curr_count / prev_count) * 100
            print(f"    {prev_name} → {curr_name}: "
                  f"{prev_count:,} → {curr_count:,} "
                  f"({drop:.1f}% drop)")
        else:
            print(f"    {prev_name} → {curr_name}: ZERO (blocked here)")

    bear_funnel = [
        ("Active FVG", counts["has_bear_fvg"]),
        ("Trend aligned", counts["bear_trend_aligned"]),
        ("Price reaches CE", counts["bear_price_reaches_ce"]),
        ("M5 rejection", counts["bear_m5_rejection"]),
        ("All required", counts["bear_all_required"]),
    ]

    print("\n  Bearish drop-off:")
    for i in range(1, len(bear_funnel)):
        prev_name, prev_count = bear_funnel[i-1]
        curr_name, curr_count = bear_funnel[i]
        if prev_count > 0:
            drop = (1 - curr_count / prev_count) * 100
            print(f"    {prev_name} → {curr_name}: "
                  f"{prev_count:,} → {curr_count:,} "
                  f"({drop:.1f}% drop)")
        else:
            print(f"    {prev_name} → {curr_name}: ZERO (blocked here)")

    print()


if __name__ == "__main__":
    diagnose()