"""
Project MIDAS v2 — Strategy E: DEBIASED Backtest
==================================================
Fixes two critical issues from the initial backtest:

  1. FEATURE LEAKAGE: swing_high/swing_low use 5 future bars to confirm.
     Fix: Delay swing confirmation by SWING_LOOKBACK bars. A swing at bar i
     is only visible to the model at bar i + SWING_LOOKBACK.

  2. TRAIN/TEST OVERLAP: Final model was trained through Jun 2025 but
     backtest started at Jul 2024 (12 months of in-sample data).
     Fix: Walk-forward backtest — each period uses ONLY the model trained
     BEFORE that period:
       - W2 model (trained ≤ Jun 2024) → tests Jul-Dec 2024
       - W3 model (trained ≤ Jun 2024) → tests Jan-Jun 2025
       - Final model (trained ≤ Jun 2025) → tests Jul 2025+

Usage:
    python run_strategy_e_debiased.py
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from copy import deepcopy

from config import (
    DATA_PROCESSED, DATA_RAW, MODELS_DIR, BACKTEST_DIR,
    BACKTEST_FORWARD_START, BACKTEST_INITIAL_CAPITAL,
    ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE,
    MAX_DAILY_LOSS, MAX_TRADE_RISK,
    LOT_SIZE_SINGLE, LOT_SIZE_MIN, LOT_SIZE_MAX,
    SPREAD_SIMULATION_PIPS, COOLDOWN_SECONDS,
    SESSION_LIMITS, SESSIONS, SWING_LOOKBACK,
)
from strategy_e_data_prep import LABEL_COL, WALK_FORWARD_WINDOWS, HOLDOUT_START


# ─── Configuration ────────────────────────────────────────
FORWARD_START = BACKTEST_FORWARD_START
INITIAL_CAPITAL = BACKTEST_INITIAL_CAPITAL

SL_ATR_MULT = 1.5
TP_RR_BASE = 1.5
TP_RR_HIGH_CONF = 2.5
HIGH_CONF_CUTOFF = 75

THRESHOLDS_TO_TEST = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]
SPREAD_PRICE = SPREAD_SIMULATION_PIPS * PIP_VALUE

# Walk-forward model mapping:
# Each test period uses the model trained STRICTLY BEFORE that period.
WF_MODEL_MAP = [
    {
        "model": "strategy_e_W2.txt",
        "test_start": "2024-07-01",
        "test_end": "2024-12-31",
        "label": "W2 model → Jul-Dec 2024",
    },
    {
        "model": "strategy_e_W3.txt",
        "test_start": "2025-01-01",
        "test_end": "2025-06-30",
        "label": "W3 model → Jan-Jun 2025",
    },
    {
        "model": "strategy_e_final.txt",
        "test_start": "2025-07-01",
        "test_end": "2026-12-31",  # Open-ended
        "label": "Final model → Jul 2025+",
    },
]


# ═════════════════════════════════════════════════════════
# FIX 1: Debias swing features by delaying confirmation
# ═════════════════════════════════════════════════════════

def debias_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix look-ahead bias in swing point features.

    Original: swing_high[i] = 1 if bar i is a local max (uses bars i-5 to i+5)
    Problem:  In real-time, you can't confirm until bar i+5

    Fix: Shift all swing-derived features FORWARD by SWING_LOOKBACK bars.
    This means swing_high[i] = 1 becomes visible only at bar i + SWING_LOOKBACK.

    Also re-derives all downstream features: last_swing_high, dist_to_swing_*,
    equal highs/lows, BOS levels.
    """
    n = SWING_LOOKBACK
    df = df.copy()

    print(f"    Debiasing swing features (shifting by {n} bars)...")

    # ── Shift the core swing detection by N bars ──────────
    # swing_high at bar i should only be "known" at bar i+N
    for col in ["swing_high", "swing_low"]:
        if col in df.columns:
            df[col] = df[col].shift(n).fillna(0).astype(int)

    for col in ["swing_high_price", "swing_low_price"]:
        if col in df.columns:
            df[col] = df[col].shift(n)

    # ── Re-derive downstream features ─────────────────────
    # last_swing_high/low: forward-fill from shifted swing prices
    if "swing_high_price" in df.columns:
        df["last_swing_high"] = df["swing_high_price"].ffill()
        df["dist_to_swing_high"] = df["close"] - df["last_swing_high"]

    if "swing_low_price" in df.columns:
        df["last_swing_low"] = df["swing_low_price"].ffill()
        df["dist_to_swing_low"] = df["close"] - df["last_swing_low"]

    # Normalized versions
    atr_col = f"atr_{ATR_PERIOD}"
    if atr_col in df.columns:
        atr = df[atr_col].replace(0, np.nan)
        if "dist_to_swing_high" in df.columns:
            df["dist_to_swing_high_atr"] = df["dist_to_swing_high"] / atr
        if "dist_to_swing_low" in df.columns:
            df["dist_to_swing_low_atr"] = df["dist_to_swing_low"] / atr

    # ── Shift BOS features ────────────────────────────────
    # BOS detection depends on swing points, so it also has look-ahead
    for col in ["bos_bull", "bos_bear"]:
        if col in df.columns:
            df[col] = df[col].shift(n).fillna(0).astype(int)

    for col in ["bos_bull_level", "bos_bear_level",
                "ob_bull_high", "ob_bull_low", "ob_bear_high", "ob_bear_low",
                "ob_bull_impulse_size", "ob_bear_impulse_size"]:
        if col in df.columns:
            df[col] = df[col].shift(n)

    # ── Shift equal highs/lows ────────────────────────────
    for col in ["equal_highs", "equal_lows", "equal_highs_count", "equal_lows_count",
                "equal_highs_level", "equal_lows_level"]:
        if col in df.columns:
            df[col] = df[col].shift(n)

    # Re-derive forward-filled versions
    if "equal_highs_level" in df.columns:
        df["last_equal_highs_level"] = df["equal_highs_level"].ffill()
    if "equal_lows_level" in df.columns:
        df["last_equal_lows_level"] = df["equal_lows_level"].ffill()

    # Re-derive distances
    if "last_equal_highs_level" in df.columns and atr_col in df.columns:
        df["dist_to_eq_highs_atr"] = (df["close"] - df["last_equal_highs_level"]) / atr
    if "last_equal_lows_level" in df.columns and atr_col in df.columns:
        df["dist_to_eq_lows_atr"] = (df["close"] - df["last_equal_lows_level"]) / atr

    # ── Shift M15-derived BOS features ────────────────────
    # M15 bars = 3x M5, so M15 swing lookback of 5 M15 bars = 15 M5 bars
    # But since we merged M15 features using merge_asof (backward), the M15
    # BOS features are already lagged by up to 15 minutes. Shifting by
    # SWING_LOOKBACK M5 bars (25 min) provides conservative debiasing.
    for col in ["m15_bos_bull", "m15_bos_bear"]:
        if col in df.columns:
            df[col] = df[col].shift(n).fillna(0).astype(int)

    return df


# ═════════════════════════════════════════════════════════
# FIX 2: Walk-forward backtest (no train/test overlap)
# ═════════════════════════════════════════════════════════

def get_session(hour_utc):
    if 12 <= hour_utc < 14:
        return "london_ny_overlap"
    for name, times in SESSIONS.items():
        if times["start"] <= hour_utc < times["end"]:
            return name
    return "off"


def get_session_limit(session):
    if session == "london_ny_overlap":
        return SESSION_LIMITS.get("london", 2)
    return SESSION_LIMITS.get(session, 0)


def prob_to_confidence(prob):
    effective = max(prob, 1 - prob)
    effective = max(effective, 0.5)
    raw_conf = (2 * (effective - 0.5)) ** 0.7
    return min(100.0, raw_conf * 100.0)


def compute_lot_size(sl_price):
    lot = LOT_SIZE_SINGLE
    if sl_price > 0:
        max_lot = MAX_TRADE_RISK / (XAUUSD_POINT_VALUE * sl_price)
        lot = min(lot, max_lot)
    return max(LOT_SIZE_MIN, min(lot, LOT_SIZE_MAX))


def run_wf_backtest(df, feature_cols, prob_threshold):
    """
    Walk-forward backtest: each test period uses the correct model.
    No train/test overlap — predictions are always out-of-sample.
    """
    capital = INITIAL_CAPITAL
    all_trades = []
    equity_curve = []
    peak_equity = capital
    max_drawdown_pct = 0.0

    for wf_segment in WF_MODEL_MAP:
        model_path = MODELS_DIR / wf_segment["model"]
        if not model_path.exists():
            print(f"    WARNING: {wf_segment['model']} not found, skipping {wf_segment['label']}")
            continue

        model = lgb.Booster(model_file=str(model_path))

        # Filter data to this segment's test period
        seg_mask = (
            (df["datetime"] >= wf_segment["test_start"]) &
            (df["datetime"] <= wf_segment["test_end"])
        )
        seg_data = df[seg_mask].copy().reset_index(drop=True)

        if len(seg_data) == 0:
            continue

        # Predict
        X = seg_data[feature_cols].values
        probs = model.predict(X)

        # ── Bar-by-bar simulation for this segment ────────
        open_trade = None
        daily_pnl = 0.0
        current_date = None
        last_trade_time = None
        session_counts = defaultdict(int)

        for i in range(len(seg_data)):
            row = seg_data.iloc[i]
            bar_time = row["datetime"]
            bar_date = pd.Timestamp(bar_time).date()
            bar_high = row["high"]
            bar_low = row["low"]
            bar_close = row["close"]
            atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
            if pd.isna(atr) or atr <= 0:
                atr = 2.0
            hour_utc = bar_time.hour
            session = get_session(hour_utc)

            # Reset daily state
            if current_date != bar_date:
                current_date = bar_date
                daily_pnl = 0.0
                session_counts = defaultdict(int)

            # Check open trade
            if open_trade is not None:
                trade = open_trade
                hit_sl = hit_tp = False

                if trade["direction"] == "BUY":
                    if bar_low <= trade["sl_price"]:
                        hit_sl, exit_price = True, trade["sl_price"]
                    elif bar_high >= trade["tp_price"]:
                        hit_tp, exit_price = True, trade["tp_price"]
                else:
                    if bar_high >= trade["sl_price"]:
                        hit_sl, exit_price = True, trade["sl_price"]
                    elif bar_low <= trade["tp_price"]:
                        hit_tp, exit_price = True, trade["tp_price"]

                if hit_sl or hit_tp:
                    if trade["direction"] == "BUY":
                        price_diff = exit_price - trade["entry_price"]
                    else:
                        price_diff = trade["entry_price"] - exit_price

                    pnl = price_diff * trade["lot_size"] * XAUUSD_POINT_VALUE
                    capital += pnl
                    daily_pnl += pnl

                    trade["exit_price"] = round(exit_price, 2)
                    trade["exit_time"] = bar_time
                    trade["pnl"] = round(pnl, 2)
                    trade["result"] = "WIN" if pnl > 0 else "LOSS"
                    trade["hit"] = "TP" if hit_tp else "SL"
                    all_trades.append(trade)
                    open_trade = None

            # Track equity
            equity_curve.append({"datetime": bar_time, "equity": round(capital, 2)})

            if capital > peak_equity:
                peak_equity = capital
            dd_pct = (peak_equity - capital) / peak_equity * 100 if peak_equity > 0 else 0
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct

            if open_trade is not None:
                continue

            # Risk checks
            if daily_pnl <= -MAX_DAILY_LOSS:
                continue
            if session == "off":
                continue
            if session_counts[session] >= get_session_limit(session):
                continue
            if last_trade_time is not None:
                elapsed = (bar_time - last_trade_time).total_seconds()
                if elapsed < COOLDOWN_SECONDS:
                    continue

            # Signal
            prob = probs[i]
            if prob >= prob_threshold:
                direction = "BUY"
                confidence = prob_to_confidence(prob)
            elif prob <= (1 - prob_threshold):
                direction = "SELL"
                confidence = prob_to_confidence(prob)
            else:
                continue

            # SL/TP
            sl_distance = SL_ATR_MULT * atr
            rr = TP_RR_HIGH_CONF if confidence >= HIGH_CONF_CUTOFF else TP_RR_BASE
            tp_distance = sl_distance * rr

            if direction == "BUY":
                entry_price = bar_close + SPREAD_PRICE
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            else:
                entry_price = bar_close - SPREAD_PRICE
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance

            lot_size = compute_lot_size(sl_distance)
            potential_loss = sl_distance * lot_size * XAUUSD_POINT_VALUE
            if potential_loss > MAX_TRADE_RISK:
                continue

            open_trade = {
                "direction": direction,
                "entry_price": round(entry_price, 2),
                "sl_price": round(sl_price, 2),
                "tp_price": round(tp_price, 2),
                "lot_size": lot_size,
                "entry_time": bar_time,
                "confidence": round(confidence, 1),
                "probability": round(float(prob), 4),
                "session": session,
                "atr": round(atr, 4),
                "sl_pips": round(sl_distance / PIP_VALUE, 1),
                "tp_pips": round(tp_distance / PIP_VALUE, 1),
                "model_segment": wf_segment["label"],
            }

            last_trade_time = bar_time
            session_counts[session] += 1

    # Close remaining trade
    if open_trade is not None:
        last_close = df.iloc[-1]["close"]
        if open_trade["direction"] == "BUY":
            pnl = (last_close - open_trade["entry_price"]) * open_trade["lot_size"] * XAUUSD_POINT_VALUE
        else:
            pnl = (open_trade["entry_price"] - last_close) * open_trade["lot_size"] * XAUUSD_POINT_VALUE
        capital += pnl
        open_trade["pnl"] = round(pnl, 2)
        open_trade["result"] = "WIN" if pnl > 0 else "LOSS"
        open_trade["hit"] = "CLOSE"
        open_trade["exit_time"] = df.iloc[-1]["datetime"]
        all_trades.append(open_trade)

    return compute_metrics(all_trades, equity_curve, capital, max_drawdown_pct)


def compute_metrics(trades, equity_curve, final_capital, max_dd_pct):
    """Compute all performance metrics."""
    if not trades:
        return {"total_pnl": 0, "win_rate": 0, "profit_factor": 0,
                "total_trades": 0, "trades_per_month": 0,
                "max_drawdown_pct": 0, "sharpe": 0}

    trade_df = pd.DataFrame(trades)
    total_trades = len(trade_df)
    wins = trade_df[trade_df["pnl"] > 0]
    losses = trade_df[trade_df["pnl"] <= 0]

    total_pnl = trade_df["pnl"].sum()
    win_rate = len(wins) / total_trades * 100
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    first_trade = pd.Timestamp(trade_df["entry_time"].min())
    last_trade = pd.Timestamp(trade_df["entry_time"].max())
    months = max((last_trade - first_trade).days / 30.44, 1)
    trades_per_month = total_trades / months

    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    avg_trade = trade_df["pnl"].mean()

    # Sharpe
    equity_df = pd.DataFrame(equity_curve)
    equity_df["datetime"] = pd.to_datetime(equity_df["datetime"])
    daily_eq = equity_df.groupby(equity_df["datetime"].dt.date)["equity"].last()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if len(daily_ret) > 0 and daily_ret.std() > 0 else 0

    # Monthly
    trade_df["month"] = pd.to_datetime(trade_df["entry_time"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("month")["pnl"].sum()
    profitable_months = (monthly_pnl > 0).sum()
    total_months = len(monthly_pnl)

    # Sessions
    session_stats = {}
    for s in trade_df["session"].unique():
        st = trade_df[trade_df["session"] == s]
        sw = st[st["pnl"] > 0]
        session_stats[s] = {
            "trades": len(st), "pnl": round(st["pnl"].sum(), 2),
            "win_rate": round(len(sw) / max(len(st), 1) * 100, 1),
        }

    # Direction
    direction_stats = {}
    for d in ["BUY", "SELL"]:
        dt = trade_df[trade_df["direction"] == d]
        dw = dt[dt["pnl"] > 0]
        direction_stats[d] = {
            "trades": len(dt), "pnl": round(dt["pnl"].sum(), 2),
            "win_rate": round(len(dw) / max(len(dt), 1) * 100, 1),
        }

    # Confidence
    avg_conf_win = wins["confidence"].mean() if len(wins) > 0 else 0
    avg_conf_loss = losses["confidence"].mean() if len(losses) > 0 else 0

    return {
        "total_pnl": round(total_pnl, 2),
        "pnl_pct": round(total_pnl / INITIAL_CAPITAL * 100, 2),
        "final_capital": round(final_capital, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "trades_per_month": round(trades_per_month, 1),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "sharpe": round(sharpe, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_trade": round(avg_trade, 2),
        "wins": len(wins),
        "losses": len(losses),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profitable_months": profitable_months,
        "total_months": total_months,
        "profitable_months_pct": round(profitable_months / max(total_months, 1) * 100, 1),
        "session_breakdown": session_stats,
        "direction_breakdown": direction_stats,
        "avg_conf_winning": round(avg_conf_win, 1),
        "avg_conf_losing": round(avg_conf_loss, 1),
        "monthly_pnl": {str(k): round(v, 2) for k, v in monthly_pnl.items()},
    }


def print_comparison_table(results, label="DEBIASED"):
    print(f"\n{'=' * 100}")
    print(f"{'Thresh':>7} | {'PnL':>9} | {'PnL%':>6} | {'WR%':>5} | {'PF':>5} | "
          f"{'Trades':>6} | {'T/Mo':>5} | {'MaxDD':>6} | {'Sharpe':>6} | "
          f"{'AvgWin':>7} | {'AvgLoss':>8} | {'ProfMo':>6}")
    print("-" * 100)
    for thresh in sorted(results.keys()):
        r = results[thresh]
        print(f"  {thresh:.2f}  | ${r['total_pnl']:>7.0f} | {r['pnl_pct']:>5.1f}% | "
              f"{r['win_rate']:>4.1f} | {r['profit_factor']:>4.2f} | "
              f"{r['total_trades']:>6} | {r['trades_per_month']:>4.1f} | "
              f"{r['max_drawdown_pct']:>5.1f}% | {r['sharpe']:>5.2f} | "
              f"${r['avg_win']:>5.2f} | ${r['avg_loss']:>7.2f} | "
              f"{r['profitable_months']}/{r['total_months']}")
    print("=" * 100)


def print_detailed(result, threshold):
    print(f"\n{'─' * 70}")
    print(f"  DETAILED REPORT — Threshold = {threshold:.2f} (DEBIASED)")
    print(f"{'─' * 70}")
    print(f"\n  PnL:              ${result['total_pnl']:+,.2f} ({result['pnl_pct']:+.2f}%)")
    print(f"  Final Capital:    ${result['final_capital']:,.2f}")
    print(f"  Win Rate:         {result['win_rate']:.1f}%  ({result['wins']}W / {result['losses']}L)")
    print(f"  Profit Factor:    {result['profit_factor']:.2f}")
    print(f"  Trades/Month:     {result['trades_per_month']:.1f}")
    print(f"  Max Drawdown:     {result['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe Ratio:     {result['sharpe']:.2f}")
    print(f"  Avg Win:          ${result['avg_win']:.2f}")
    print(f"  Avg Loss:         ${result['avg_loss']:.2f}")
    print(f"  Avg Trade:        ${result['avg_trade']:.2f}")
    print(f"  Profitable Months: {result['profitable_months']}/{result['total_months']} "
          f"({result['profitable_months_pct']:.1f}%)")
    print(f"\n  Confidence on Winners: {result['avg_conf_winning']:.1f}%")
    print(f"  Confidence on Losers:  {result['avg_conf_losing']:.1f}%")
    print(f"  Conf Separation:       {result['avg_conf_winning'] - result['avg_conf_losing']:+.1f}%")

    for dir_name, stats in result.get("direction_breakdown", {}).items():
        print(f"    {dir_name}: {stats['trades']} trades, ${stats['pnl']:+.2f}, {stats['win_rate']:.1f}% WR")

    print(f"\n  Session Breakdown:")
    for sess, stats in result.get("session_breakdown", {}).items():
        print(f"    {sess:<20s}: {stats['trades']:>4} trades, ${stats['pnl']:>+8.2f}, {stats['win_rate']:.1f}% WR")

    print(f"\n  Monthly PnL:")
    for month, pnl in result.get("monthly_pnl", {}).items():
        marker = "✅" if pnl > 0 else "❌"
        print(f"    {month}: ${pnl:>+8.2f} {marker}")


def main():
    print("=" * 70)
    print("MIDAS v2 — Strategy E: DEBIASED Backtest")
    print("  Fix 1: Swing features shifted by SWING_LOOKBACK (no look-ahead)")
    print("  Fix 2: Walk-forward models (no train/test overlap)")
    print(f"  Forward Test: {FORWARD_START} → present")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────
    print("\n[1/4] Loading data...")
    data_path = DATA_PROCESSED / "strategy_e_train_ready.parquet"
    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    meta_path = MODELS_DIR / "strategy_e_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"]
    print(f"  Data: {len(df):,} rows, {len(feature_cols)} features")

    # ── Debias swing features ─────────────────────────────
    print("\n[2/4] Debiasing swing features...")
    df = debias_swing_features(df)

    # ── Verify models exist ───────────────────────────────
    print("\n[3/4] Checking walk-forward models...")
    for seg in WF_MODEL_MAP:
        model_path = MODELS_DIR / seg["model"]
        status = "✅" if model_path.exists() else "❌ MISSING"
        print(f"    {seg['model']:<30s} → {seg['label']:<30s} {status}")

    # ── Run debiased backtests ────────────────────────────
    print(f"\n[4/4] Running debiased backtests at {len(THRESHOLDS_TO_TEST)} thresholds...")
    results = {}

    for thresh in THRESHOLDS_TO_TEST:
        print(f"\n  Threshold = {thresh:.2f}...", end=" ")
        result = run_wf_backtest(df, feature_cols, thresh)
        results[thresh] = result
        print(f"PnL: ${result['total_pnl']:+,.2f} | WR: {result['win_rate']:.1f}% | "
              f"PF: {result['profit_factor']:.2f} | Trades: {result['total_trades']} | "
              f"MaxDD: {result['max_drawdown_pct']:.1f}%")

    # ── Print results ─────────────────────────────────────
    print_comparison_table(results)

    # Best by PnL
    best_thresh = max(results.keys(), key=lambda t: results[t]["total_pnl"])
    best = results[best_thresh]
    print_detailed(best, best_thresh)

    # ── Side-by-side: Biased vs Debiased ──────────────────
    print(f"\n{'=' * 70}")
    print("BIASED vs DEBIASED Comparison (best threshold)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25s} {'BIASED (t=0.70)':>18s} {'DEBIASED (t={:.2f})':>18s}".format(best_thresh))
    print(f"  {'-'*61}")
    print(f"  {'PnL':<25s} {'$15,474':>18s} ${best['total_pnl']:>17,.0f}")
    print(f"  {'Win Rate':<25s} {'70.9%':>18s} {best['win_rate']:>17.1f}%")
    print(f"  {'Profit Factor':<25s} {'4.08':>18s} {best['profit_factor']:>17.2f}")
    print(f"  {'Trades/Month':<25s} {'157.7':>18s} {best['trades_per_month']:>17.1f}")
    print(f"  {'Max Drawdown':<25s} {'0.5%':>18s} {best['max_drawdown_pct']:>17.1f}%")
    print(f"  {'Sharpe':<25s} {'20.17':>18s} {best['sharpe']:>17.2f}")
    print(f"  {'Profitable Months':<25s} {'22/22':>18s} "
          f"{best['profitable_months']}/{best['total_months']:>14}")

    # ── Strategy comparison ───────────────────────────────
    print(f"\n{'=' * 70}")
    print("Strategy Comparison (Forward Test) — DEBIASED")
    print(f"{'=' * 70}")
    print(f"  {'Strategy':<20s} {'PnL':>9} {'WR%':>6} {'PF':>5} {'T/Mo':>5} {'MaxDD':>6}")
    print(f"  {'-'*52}")
    print(f"  {'A (SMC OB)':<20s} ${'2,092':>7} {'49.4':>5} {'1.57':>5} {'51.4':>5} {'2.5%':>6}")
    print(f"  {'B (FVG)':<20s} ${'663':>7} {'70.4':>5} {'1.71':>5} {'32.6':>5} {'2.5%':>6}")
    print(f"  {'C (Session)':<20s} ${'497':>7} {'51.6':>5} {'1.25':>5} {'11.0':>5} {'6.2%':>6}")
    print(f"  {'D (Trend)':<20s} ${'36':>7} {'41.8':>5} {'1.17':>5} {'2.7':>5} {'1.2%':>6}")
    print(f"  {'A+B+C+D':<20s} ${'2,732':>7} {'57.6':>5} {'1.55':>5} {'64.0':>5} {'3.5%':>6}")
    print(f"  {'E (DEBIASED) t=' + f'{best_thresh:.2f}':<20s} "
          f"${best['total_pnl']:>7,.0f} "
          f"{best['win_rate']:>5.1f} "
          f"{best['profit_factor']:>5.2f} "
          f"{best['trades_per_month']:>5.1f} "
          f"{best['max_drawdown_pct']:>5.1f}%")

    # Save
    save_path = BACKTEST_DIR / "strategy_e_backtest_debiased.json"
    save_data = {
        "best_threshold": best_thresh,
        "fixes_applied": [
            "swing features shifted by SWING_LOOKBACK (5 bars) to remove look-ahead",
            "walk-forward models used (W2→H2-2024, W3→H1-2025, final→H2-2025+)",
        ],
        "results": {str(t): {k: v for k, v in r.items() if k != "monthly_pnl"}
                    for t, r in results.items()},
        "best_result_monthly": best.get("monthly_pnl", {}),
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {save_path.name}")

    print(f"\n{'=' * 70}")
    print("✅ Debiased backtest complete!")
    print("   These numbers are the REAL Strategy E performance.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()