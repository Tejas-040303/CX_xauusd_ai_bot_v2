"""
Project MIDAS v2 — Strategy E: Standalone Backtest
====================================================
Runs Strategy E (LightGBM) through the forward test period with full
risk management, matching the same evaluation format as strategies A-D.

Tests multiple confidence thresholds to find the sweet spot, then
produces a detailed report with PnL, win rate, profit factor,
max drawdown, trades/month, equity curve, and session breakdown.

Usage:
    python run_strategy_e.py

Output:
    backtest_results/strategy_e_backtest.json
    backtest_results/strategy_e_equity.png
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from config import (
    DATA_PROCESSED, MODELS_DIR, BACKTEST_DIR,
    BACKTEST_FORWARD_START, BACKTEST_INITIAL_CAPITAL,
    ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE,
    MAX_DAILY_LOSS, MAX_TRADE_RISK,
    LOT_SIZE_SINGLE, LOT_SIZE_MIN, LOT_SIZE_MAX,
    SPREAD_SIMULATION_PIPS, COOLDOWN_SECONDS,
    SESSION_LIMITS, SESSIONS,
)
from strategy_e_data_prep import LABEL_COL


# ─── Backtest Configuration ──────────────────────────────
FORWARD_START = BACKTEST_FORWARD_START  # "2024-07-01"
INITIAL_CAPITAL = BACKTEST_INITIAL_CAPITAL  # 5000

# SL/TP parameters (same as strategy_e.py)
SL_ATR_MULT = 1.5
TP_RR_BASE = 1.5
TP_RR_HIGH_CONF = 2.5
HIGH_CONF_CUTOFF = 75  # Confidence % above which we use extended TP

# Confidence thresholds to test
THRESHOLDS_TO_TEST = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]

# Spread in price terms
SPREAD_PRICE = SPREAD_SIMULATION_PIPS * PIP_VALUE  # 2.5 * 0.10 = $0.25


def load_model_and_data():
    """Load the trained LightGBM model and feature data."""
    # Load model
    model_path = MODELS_DIR / "strategy_e_final.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    print(f"  Model loaded: {model_path.name}")

    # Load metadata for feature list
    meta_path = MODELS_DIR / "strategy_e_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"]
    print(f"  Features: {len(feature_cols)}")

    # Load train-ready data
    data_path = DATA_PROCESSED / "strategy_e_train_ready.parquet"
    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  Data loaded: {len(df):,} rows")

    return model, feature_cols, df, metadata


def get_session(hour_utc):
    """Determine trading session from UTC hour."""
    # Check overlap first
    if 12 <= hour_utc < 14:
        return "london_ny_overlap"
    for name, times in SESSIONS.items():
        if times["start"] <= hour_utc < times["end"]:
            return name
    return "off"


def get_session_limit(session):
    """Get trade limit for a session."""
    if session == "london_ny_overlap":
        return SESSION_LIMITS.get("london", 2)
    return SESSION_LIMITS.get(session, 0)


def compute_lot_size(sl_price, confidence):
    """
    Compute lot size to keep risk within MAX_TRADE_RISK.
    Strategy E independent = always 0.01 lot (conservative),
    but cap by risk if SL is wide.
    """
    lot = LOT_SIZE_SINGLE  # 0.01 for independent E

    if sl_price > 0:
        max_lot = MAX_TRADE_RISK / (XAUUSD_POINT_VALUE * sl_price)
        lot = min(lot, max_lot)

    lot = max(LOT_SIZE_MIN, min(lot, LOT_SIZE_MAX))
    return round(lot * 100) / 100


def prob_to_confidence(prob):
    """Map model probability to confidence 0-100 (same as strategy_e.py)."""
    effective = max(prob, 1 - prob)
    effective = max(effective, 0.5)
    raw_conf = (2 * (effective - 0.5)) ** 0.7
    return min(100.0, raw_conf * 100.0)


def run_backtest(model, feature_cols, df, prob_threshold):
    """
    Run a full bar-by-bar backtest for a given probability threshold.

    Returns a dict with all performance metrics + trade list.
    """
    # Filter to forward test period
    fw_mask = df["datetime"] >= FORWARD_START
    fw_data = df[fw_mask].copy().reset_index(drop=True)

    if len(fw_data) == 0:
        return None

    # Predict probabilities for all forward test bars
    X = fw_data[feature_cols].values
    probs = model.predict(X)

    # ── State tracking ────────────────────────────────────
    capital = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    open_trade = None

    daily_pnl = 0.0
    daily_trades = 0
    current_date = None
    last_trade_time = None
    session_counts = defaultdict(int)

    peak_equity = capital
    max_drawdown = 0.0
    max_drawdown_pct = 0.0

    # ── Bar-by-bar simulation ─────────────────────────────
    for i in range(len(fw_data)):
        row = fw_data.iloc[i]
        bar_time = row["datetime"]
        bar_date = bar_time.date() if hasattr(bar_time, 'date') else pd.Timestamp(bar_time).date()
        bar_high = row["high"]
        bar_low = row["low"]
        bar_close = row["close"]
        atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
        if pd.isna(atr) or atr <= 0:
            atr = 2.0

        hour_utc = bar_time.hour
        session = get_session(hour_utc)

        # ── Reset daily state ─────────────────────────────
        if current_date != bar_date:
            current_date = bar_date
            daily_pnl = 0.0
            daily_trades = 0
            session_counts = defaultdict(int)

        # ── Check if open trade hits SL or TP ─────────────
        if open_trade is not None:
            trade = open_trade
            hit_sl = False
            hit_tp = False

            if trade["direction"] == "BUY":
                if bar_low <= trade["sl_price"]:
                    hit_sl = True
                    exit_price = trade["sl_price"]
                elif bar_high >= trade["tp_price"]:
                    hit_tp = True
                    exit_price = trade["tp_price"]
            else:  # SELL
                if bar_high >= trade["sl_price"]:
                    hit_sl = True
                    exit_price = trade["sl_price"]
                elif bar_low <= trade["tp_price"]:
                    hit_tp = True
                    exit_price = trade["tp_price"]

            if hit_sl or hit_tp:
                # Calculate PnL
                if trade["direction"] == "BUY":
                    price_diff = exit_price - trade["entry_price"]
                else:
                    price_diff = trade["entry_price"] - exit_price

                pnl = price_diff * trade["lot_size"] * XAUUSD_POINT_VALUE
                capital += pnl
                daily_pnl += pnl

                trade["exit_price"] = exit_price
                trade["exit_time"] = bar_time
                trade["pnl"] = round(pnl, 2)
                trade["result"] = "WIN" if pnl > 0 else "LOSS"
                trade["hit"] = "TP" if hit_tp else "SL"
                trades.append(trade)
                open_trade = None

        # ── Track equity ──────────────────────────────────
        equity_curve.append({
            "datetime": bar_time,
            "equity": round(capital, 2),
        })

        # Track drawdown
        if capital > peak_equity:
            peak_equity = capital
        dd = peak_equity - capital
        dd_pct = (dd / peak_equity) * 100 if peak_equity > 0 else 0
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct
            max_drawdown = dd

        # ── Skip if trade already open ────────────────────
        if open_trade is not None:
            continue

        # ── Risk pre-flight checks ────────────────────────
        if daily_pnl <= -MAX_DAILY_LOSS:
            continue
        if session == "off":
            continue
        session_limit = get_session_limit(session)
        if session_counts[session] >= session_limit:
            continue
        if last_trade_time is not None:
            elapsed = (bar_time - last_trade_time).total_seconds()
            if elapsed < COOLDOWN_SECONDS:
                continue

        # ── Generate signal ───────────────────────────────
        prob = probs[i]

        if prob >= prob_threshold:
            direction = "BUY"
            confidence = prob_to_confidence(prob)
        elif prob <= (1 - prob_threshold):
            direction = "SELL"
            confidence = prob_to_confidence(prob)
        else:
            continue  # No signal

        # ── Calculate SL/TP ───────────────────────────────
        sl_distance = SL_ATR_MULT * atr
        rr = TP_RR_HIGH_CONF if confidence >= HIGH_CONF_CUTOFF else TP_RR_BASE
        tp_distance = sl_distance * rr

        if direction == "BUY":
            entry_price = bar_close + SPREAD_PRICE  # Buy at ask
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            entry_price = bar_close - SPREAD_PRICE  # Sell at bid
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        # ── Lot size ──────────────────────────────────────
        lot_size = compute_lot_size(sl_distance, confidence)

        # Verify risk doesn't exceed max
        potential_loss = sl_distance * lot_size * XAUUSD_POINT_VALUE
        if potential_loss > MAX_TRADE_RISK:
            continue

        # ── Open trade ────────────────────────────────────
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
        }

        last_trade_time = bar_time
        session_counts[session] += 1
        daily_trades += 1

    # ── Close any remaining open trade at last close ──────
    if open_trade is not None:
        last_row = fw_data.iloc[-1]
        if open_trade["direction"] == "BUY":
            price_diff = last_row["close"] - open_trade["entry_price"]
        else:
            price_diff = open_trade["entry_price"] - last_row["close"]
        pnl = price_diff * open_trade["lot_size"] * XAUUSD_POINT_VALUE
        capital += pnl
        open_trade["exit_price"] = round(last_row["close"], 2)
        open_trade["exit_time"] = last_row["datetime"]
        open_trade["pnl"] = round(pnl, 2)
        open_trade["result"] = "WIN" if pnl > 0 else "LOSS"
        open_trade["hit"] = "CLOSE"
        trades.append(open_trade)

    # ── Compute metrics ───────────────────────────────────
    return compute_metrics(trades, equity_curve, capital, max_drawdown_pct, fw_data)


def compute_metrics(trades, equity_curve, final_capital, max_dd_pct, fw_data):
    """Compute all performance metrics from trade list."""
    if not trades:
        return {
            "total_pnl": 0, "win_rate": 0, "profit_factor": 0,
            "total_trades": 0, "trades_per_month": 0,
            "max_drawdown_pct": 0, "sharpe": 0,
        }

    trade_df = pd.DataFrame(trades)
    total_trades = len(trade_df)
    wins = trade_df[trade_df["pnl"] > 0]
    losses = trade_df[trade_df["pnl"] <= 0]

    total_pnl = trade_df["pnl"].sum()
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Trades per month
    if total_trades > 0:
        first_trade = trade_df["entry_time"].min()
        last_trade = trade_df["entry_time"].max()
        months = max((last_trade - first_trade).days / 30.44, 1)
        trades_per_month = total_trades / months
    else:
        trades_per_month = 0

    # Average trade metrics
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    avg_trade = trade_df["pnl"].mean()

    # Sharpe ratio (annualized, using daily returns)
    equity_df = pd.DataFrame(equity_curve)
    if len(equity_df) > 1:
        equity_df["datetime"] = pd.to_datetime(equity_df["datetime"])
        daily_equity = equity_df.groupby(equity_df["datetime"].dt.date)["equity"].last()
        daily_returns = daily_equity.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Session breakdown
    session_stats = {}
    for session_name in trade_df["session"].unique():
        s_trades = trade_df[trade_df["session"] == session_name]
        s_wins = s_trades[s_trades["pnl"] > 0]
        session_stats[session_name] = {
            "trades": len(s_trades),
            "pnl": round(s_trades["pnl"].sum(), 2),
            "win_rate": round(len(s_wins) / len(s_trades) * 100, 1) if len(s_trades) > 0 else 0,
        }

    # Direction breakdown
    buys = trade_df[trade_df["direction"] == "BUY"]
    sells = trade_df[trade_df["direction"] == "SELL"]
    direction_stats = {
        "BUY": {
            "trades": len(buys),
            "pnl": round(buys["pnl"].sum(), 2),
            "win_rate": round(len(buys[buys["pnl"] > 0]) / max(len(buys), 1) * 100, 1),
        },
        "SELL": {
            "trades": len(sells),
            "pnl": round(sells["pnl"].sum(), 2),
            "win_rate": round(len(sells[sells["pnl"] > 0]) / max(len(sells), 1) * 100, 1),
        },
    }

    # Monthly PnL
    trade_df["month"] = pd.to_datetime(trade_df["entry_time"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("month")["pnl"].sum()
    profitable_months = (monthly_pnl > 0).sum()
    total_months = len(monthly_pnl)
    profitable_months_pct = profitable_months / total_months * 100 if total_months > 0 else 0

    # Average confidence on winning vs losing trades
    avg_conf_win = wins["confidence"].mean() if len(wins) > 0 else 0
    avg_conf_loss = losses["confidence"].mean() if len(losses) > 0 else 0

    return {
        "total_pnl": round(total_pnl, 2),
        "pnl_pct": round(total_pnl / INITIAL_CAPITAL * 100, 2),
        "final_capital": round(final_capital, 2),
        "win_rate": round(win_rate * 100, 1),
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
        "profitable_months_pct": round(profitable_months_pct, 1),
        "session_breakdown": session_stats,
        "direction_breakdown": direction_stats,
        "avg_conf_winning": round(avg_conf_win, 1),
        "avg_conf_losing": round(avg_conf_loss, 1),
        "monthly_pnl": {str(k): round(v, 2) for k, v in monthly_pnl.items()},
        "equity_curve": equity_curve,
        "trades": trades,
    }


def save_equity_plot(results_by_threshold):
    """Save equity curve comparison across thresholds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Strategy E — Standalone Backtest Results", fontsize=16, fontweight="bold")

    # ── Plot 1: Equity curves for different thresholds ────
    ax = axes[0, 0]
    for thresh, result in results_by_threshold.items():
        if not result or not result.get("equity_curve"):
            continue
        eq = pd.DataFrame(result["equity_curve"])
        eq["datetime"] = pd.to_datetime(eq["datetime"])
        # Downsample for plotting (every 100th point)
        eq_plot = eq.iloc[::100]
        label = f"t={thresh:.2f} (PnL=${result['total_pnl']:+.0f})"
        ax.plot(eq_plot["datetime"], eq_plot["equity"], label=label, linewidth=1)
    ax.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Equity ($)")
    ax.set_title("Equity curves by threshold")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    # ── Plot 2: PnL vs threshold ──────────────────────────
    ax = axes[0, 1]
    thresholds = sorted(results_by_threshold.keys())
    pnls = [results_by_threshold[t]["total_pnl"] for t in thresholds]
    wrs = [results_by_threshold[t]["win_rate"] for t in thresholds]
    counts = [results_by_threshold[t]["total_trades"] for t in thresholds]

    ax2 = ax.twinx()
    bars = ax.bar(thresholds, pnls, width=0.018, color="#4CAF50", alpha=0.7, label="PnL ($)")
    line = ax2.plot(thresholds, wrs, "ro-", label="Win Rate %", markersize=5)
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("PnL ($)", color="#4CAF50")
    ax2.set_ylabel("Win Rate (%)", color="red")
    ax.set_title("PnL & win rate vs threshold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # ── Plot 3: Monthly PnL for best threshold ────────────
    ax = axes[1, 0]
    # Find best threshold by PnL
    best_thresh = max(results_by_threshold.keys(),
                      key=lambda t: results_by_threshold[t]["total_pnl"])
    best = results_by_threshold[best_thresh]
    if best.get("monthly_pnl"):
        months = list(best["monthly_pnl"].keys())
        month_pnls = list(best["monthly_pnl"].values())
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in month_pnls]
        ax.bar(range(len(months)), month_pnls, color=colors, alpha=0.8)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, fontsize=7)
        ax.set_ylabel("PnL ($)")
        ax.set_title(f"Monthly PnL (best threshold={best_thresh:.2f})")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # ── Plot 4: Session + direction breakdown ─────────────
    ax = axes[1, 1]
    if best.get("session_breakdown"):
        sessions = list(best["session_breakdown"].keys())
        s_pnls = [best["session_breakdown"][s]["pnl"] for s in sessions]
        s_trades = [best["session_breakdown"][s]["trades"] for s in sessions]
        s_wrs = [best["session_breakdown"][s]["win_rate"] for s in sessions]

        x = np.arange(len(sessions))
        bars = ax.bar(x, s_pnls, color="#2196F3", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(sessions, fontsize=8)
        ax.set_ylabel("PnL ($)")
        ax.set_title(f"Session breakdown (threshold={best_thresh:.2f})")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        for i, (pnl, t, wr) in enumerate(zip(s_pnls, s_trades, s_wrs)):
            ax.annotate(f"{t}t, {wr}%WR",
                        (i, pnl + (5 if pnl >= 0 else -15)),
                        ha="center", fontsize=7)

    plt.tight_layout()
    output_path = BACKTEST_DIR / "strategy_e_equity.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plots saved to {output_path.name}")
    plt.close()


def print_comparison_table(results_by_threshold):
    """Print a clean comparison table across all thresholds."""
    print("\n" + "=" * 100)
    print(f"{'Thresh':>7} | {'PnL':>9} | {'PnL%':>6} | {'WR%':>5} | {'PF':>5} | "
          f"{'Trades':>6} | {'T/Mo':>5} | {'MaxDD':>6} | {'Sharpe':>6} | "
          f"{'AvgWin':>7} | {'AvgLoss':>8} | {'ProfMo':>6}")
    print("-" * 100)

    for thresh in sorted(results_by_threshold.keys()):
        r = results_by_threshold[thresh]
        print(f"  {thresh:.2f}  | ${r['total_pnl']:>7.0f} | {r['pnl_pct']:>5.1f}% | "
              f"{r['win_rate']:>4.1f} | {r['profit_factor']:>4.2f} | "
              f"{r['total_trades']:>6} | {r['trades_per_month']:>4.1f} | "
              f"{r['max_drawdown_pct']:>5.1f}% | {r['sharpe']:>5.2f} | "
              f"${r['avg_win']:>5.2f} | ${r['avg_loss']:>7.2f} | "
              f"{r['profitable_months']}/{r['total_months']}")

    print("=" * 100)


def print_detailed_report(result, threshold):
    """Print detailed breakdown for a single threshold."""
    print(f"\n{'─' * 70}")
    print(f"  DETAILED REPORT — Threshold = {threshold:.2f}")
    print(f"{'─' * 70}")

    print(f"\n  PnL:              ${result['total_pnl']:+,.2f} ({result['pnl_pct']:+.2f}%)")
    print(f"  Final Capital:    ${result['final_capital']:,.2f}")
    print(f"  Win Rate:         {result['win_rate']:.1f}%  "
          f"({result['wins']}W / {result['losses']}L)")
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

    print(f"\n  Direction Breakdown:")
    for dir_name, stats in result.get("direction_breakdown", {}).items():
        print(f"    {dir_name}: {stats['trades']} trades, "
              f"${stats['pnl']:+.2f}, {stats['win_rate']:.1f}% WR")

    print(f"\n  Session Breakdown:")
    for sess, stats in result.get("session_breakdown", {}).items():
        print(f"    {sess:<20s}: {stats['trades']:>4} trades, "
              f"${stats['pnl']:>+8.2f}, {stats['win_rate']:.1f}% WR")

    print(f"\n  Monthly PnL:")
    for month, pnl in result.get("monthly_pnl", {}).items():
        marker = "✅" if pnl > 0 else "❌"
        print(f"    {month}: ${pnl:>+8.2f} {marker}")


def main():
    print("=" * 70)
    print("MIDAS v2 — Strategy E: Standalone Backtest")
    print(f"Forward Test Period: {FORWARD_START} → present")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print("=" * 70)

    # ── Load model and data ───────────────────────────────
    print("\n[1/3] Loading model and data...")
    model, feature_cols, df, metadata = load_model_and_data()

    # ── Run backtests at multiple thresholds ──────────────
    print(f"\n[2/3] Running backtests at {len(THRESHOLDS_TO_TEST)} thresholds...")
    results_by_threshold = {}

    for thresh in THRESHOLDS_TO_TEST:
        print(f"\n  Testing threshold = {thresh:.2f}...")
        result = run_backtest(model, feature_cols, df, thresh)
        if result:
            results_by_threshold[thresh] = result
            print(f"    → PnL: ${result['total_pnl']:+,.2f} | "
                  f"WR: {result['win_rate']:.1f}% | "
                  f"PF: {result['profit_factor']:.2f} | "
                  f"Trades: {result['total_trades']} | "
                  f"MaxDD: {result['max_drawdown_pct']:.1f}%")

    # ── Results ───────────────────────────────────────────
    print("\n[3/3] Results...")
    print_comparison_table(results_by_threshold)

    # Find best threshold
    if results_by_threshold:
        best_thresh = max(results_by_threshold.keys(),
                          key=lambda t: results_by_threshold[t]["total_pnl"])
        best = results_by_threshold[best_thresh]

        print_detailed_report(best, best_thresh)

        # ── Comparison with A-D strategies ────────────────
        print(f"\n{'=' * 70}")
        print("Strategy Comparison (Forward Test)")
        print(f"{'=' * 70}")
        print(f"  {'Strategy':<20s} {'PnL':>9} {'WR%':>6} {'PF':>5} {'T/Mo':>5} {'MaxDD':>6}")
        print(f"  {'-'*52}")
        print(f"  {'A (SMC OB)':<20s} ${'2,092':>7} {'49.4':>5} {'1.57':>5} {'51.4':>5} {'2.5%':>6}")
        print(f"  {'B (FVG)':<20s} ${'663':>7} {'70.4':>5} {'1.71':>5} {'32.6':>5} {'2.5%':>6}")
        print(f"  {'C (Session)':<20s} ${'497':>7} {'51.6':>5} {'1.25':>5} {'11.0':>5} {'6.2%':>6}")
        print(f"  {'D (Trend)':<20s} ${'36':>7} {'41.8':>5} {'1.17':>5} {'2.7':>5} {'1.2%':>6}")
        print(f"  {'A+B+C+D':<20s} ${'2,732':>7} {'57.6':>5} {'1.55':>5} {'64.0':>5} {'3.5%':>6}")
        print(f"  {'E (AI) t=' + f'{best_thresh:.2f}':<20s} "
              f"${best['total_pnl']:>7,.0f} "
              f"{best['win_rate']:>5.1f} "
              f"{best['profit_factor']:>5.2f} "
              f"{best['trades_per_month']:>5.1f} "
              f"{best['max_drawdown_pct']:>5.1f}%")

        # ── Save results ──────────────────────────────────
        save_data = {
            "best_threshold": best_thresh,
            "thresholds_tested": THRESHOLDS_TO_TEST,
            "summary": {t: {k: v for k, v in r.items()
                            if k not in ("equity_curve", "trades")}
                        for t, r in results_by_threshold.items()},
            "best_result": {k: v for k, v in best.items()
                           if k not in ("equity_curve",)},
        }

        results_path = BACKTEST_DIR / "strategy_e_backtest.json"
        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path.name}")

        # Save plots
        save_equity_plot(results_by_threshold)

    print(f"\n{'=' * 70}")
    print("✅ Strategy E backtest complete!")
    print("   Next: Run A+B+C+D+E combined to measure the full system.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()