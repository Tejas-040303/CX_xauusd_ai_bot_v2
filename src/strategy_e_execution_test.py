"""
Project MIDAS v2 — Strategy E: Realistic Execution Test
=========================================================
THE critical reality check: does Strategy E's edge survive
when we enter on bar i+1 OPEN instead of bar i CLOSE?

Why this matters:
  - Bar i close arrives → compute 134 features → run LightGBM → send order
  - By then bar i is over. Best case: fill on bar i+1 open
  - If the "edge" is just the price movement between bar i close and bar i+1 open,
    the model is predicting something it can't actually trade on

This script runs BOTH versions side-by-side:
  1. INSTANT:  enter at bar i close (current backtest — optimistic)
  2. DELAYED:  enter at bar i+1 open (realistic execution)

Same clean models, same clean data, same walk-forward structure.

Usage:
    python strategy_e_execution_test.py

Output:
    Prints side-by-side comparison at each threshold.
    Saves to backtest_results/strategy_e_execution_test.json
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict

from config import (
    DATA_PROCESSED, MODELS_DIR, BACKTEST_DIR,
    ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE,
    MAX_DAILY_LOSS, MAX_TRADE_RISK,
    LOT_SIZE_SINGLE, LOT_SIZE_MIN, LOT_SIZE_MAX,
    SPREAD_SIMULATION_PIPS, COOLDOWN_SECONDS,
    SESSION_LIMITS, SESSIONS, BACKTEST_INITIAL_CAPITAL,
)

# ─── Config ──────────────────────────────────────────────
INITIAL_CAPITAL = BACKTEST_INITIAL_CAPITAL
SL_ATR_MULT = 1.5
TP_RR_BASE = 1.5
TP_RR_HIGH_CONF = 2.5
HIGH_CONF_CUTOFF = 75
SPREAD_PRICE = SPREAD_SIMULATION_PIPS * PIP_VALUE
THRESHOLDS_TO_TEST = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]

WF_MODEL_MAP = [
    {"model": "strategy_e_clean_W2.txt",
     "test_start": "2024-07-01", "test_end": "2024-12-31"},
    {"model": "strategy_e_clean_W3.txt",
     "test_start": "2025-01-01", "test_end": "2025-06-30"},
    {"model": "strategy_e_clean_final.txt",
     "test_start": "2025-07-01", "test_end": "2026-12-31"},
]


def get_session(hour):
    if 12 <= hour < 14:
        return "london_ny_overlap"
    for name, times in SESSIONS.items():
        if times["start"] <= hour < times["end"]:
            return name
    return "off"


def get_session_limit(session):
    if session == "london_ny_overlap":
        return SESSION_LIMITS.get("london", 2)
    return SESSION_LIMITS.get(session, 0)


def prob_to_confidence(prob):
    effective = max(prob, 1 - prob)
    effective = max(effective, 0.5)
    return min(100.0, (2 * (effective - 0.5)) ** 0.7 * 100.0)


def run_backtest(df, feature_cols, prob_threshold, execution_mode="instant"):
    """
    Walk-forward backtest with configurable execution.

    execution_mode:
      "instant"  — enter at bar i close (current optimistic version)
      "delayed"  — signal on bar i, enter at bar i+1 open (realistic)
    """
    capital = INITIAL_CAPITAL
    trades = []
    equity = []
    peak = capital
    max_dd = 0.0

    for seg in WF_MODEL_MAP:
        model_path = MODELS_DIR / seg["model"]
        if not model_path.exists():
            continue

        model = lgb.Booster(model_file=str(model_path))
        seg_data = df[(df["datetime"] >= seg["test_start"]) &
                      (df["datetime"] <= seg["test_end"])].reset_index(drop=True)
        if len(seg_data) == 0:
            continue

        probs = model.predict(seg_data[feature_cols].values)

        open_trade = None
        pending_signal = None  # For delayed mode: signal waiting for next bar
        daily_pnl = 0.0
        current_date = None
        last_trade_time = None
        sess_counts = defaultdict(int)

        for i in range(len(seg_data)):
            row = seg_data.iloc[i]
            bt = row["datetime"]
            bd = pd.Timestamp(bt).date()
            bar_open = row["open"]
            bh, bl, bc = row["high"], row["low"], row["close"]
            atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
            if pd.isna(atr) or atr <= 0:
                atr = 2.0
            session = get_session(bt.hour)

            if current_date != bd:
                current_date = bd
                daily_pnl = 0.0
                sess_counts = defaultdict(int)

            # ── Execute pending signal (delayed mode) ─────
            if execution_mode == "delayed" and pending_signal is not None and open_trade is None:
                sig = pending_signal
                pending_signal = None

                # Enter at THIS bar's open (bar i+1 from signal)
                sl_dist = sig["sl_dist"]
                tp_dist = sig["tp_dist"]

                if sig["direction"] == "BUY":
                    entry = bar_open + SPREAD_PRICE
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    entry = bar_open - SPREAD_PRICE
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                lot = min(LOT_SIZE_SINGLE, MAX_TRADE_RISK / (XAUUSD_POINT_VALUE * sl_dist))
                lot = max(LOT_SIZE_MIN, min(lot, LOT_SIZE_MAX))
                lot = round(lot * 100) / 100

                if sl_dist * lot * XAUUSD_POINT_VALUE <= MAX_TRADE_RISK:
                    open_trade = {
                        "direction": sig["direction"],
                        "entry_price": round(entry, 2),
                        "sl_price": round(sl, 2),
                        "tp_price": round(tp, 2),
                        "lot_size": lot,
                        "entry_time": bt,
                        "confidence": sig["confidence"],
                        "probability": sig["probability"],
                        "session": sig["session"],
                        "atr": round(atr, 4),
                        "sl_pips": round(sl_dist / PIP_VALUE, 1),
                        "tp_pips": round(tp_dist / PIP_VALUE, 1),
                        "signal_time": sig["signal_time"],
                        "slippage": round(abs(bar_open - sig["signal_close"]), 2),
                    }
                    last_trade_time = bt
                    sess_counts[sig["session"]] += 1

                    # Check if THIS bar already hits SL or TP
                    # (could gap past our entry)
                    t = open_trade
                    hit_sl = hit_tp = False
                    if t["direction"] == "BUY":
                        if bl <= t["sl_price"]:
                            hit_sl, ep = True, t["sl_price"]
                        elif bh >= t["tp_price"]:
                            hit_tp, ep = True, t["tp_price"]
                    else:
                        if bh >= t["sl_price"]:
                            hit_sl, ep = True, t["sl_price"]
                        elif bl <= t["tp_price"]:
                            hit_tp, ep = True, t["tp_price"]

                    if hit_sl or hit_tp:
                        pd_diff = (ep - t["entry_price"]) if t["direction"] == "BUY" else (t["entry_price"] - ep)
                        pnl = pd_diff * t["lot_size"] * XAUUSD_POINT_VALUE
                        capital += pnl
                        daily_pnl += pnl
                        t.update({"exit_price": round(ep, 2), "exit_time": bt,
                                  "pnl": round(pnl, 2),
                                  "result": "WIN" if pnl > 0 else "LOSS",
                                  "hit": "TP" if hit_tp else "SL"})
                        trades.append(t)
                        open_trade = None

            # ── Check open trade SL/TP ────────────────────
            if open_trade is not None:
                t = open_trade
                hit_sl = hit_tp = False
                if t["direction"] == "BUY":
                    if bl <= t["sl_price"]:
                        hit_sl, ep = True, t["sl_price"]
                    elif bh >= t["tp_price"]:
                        hit_tp, ep = True, t["tp_price"]
                else:
                    if bh >= t["sl_price"]:
                        hit_sl, ep = True, t["sl_price"]
                    elif bl <= t["tp_price"]:
                        hit_tp, ep = True, t["tp_price"]

                if hit_sl or hit_tp:
                    pd_diff = (ep - t["entry_price"]) if t["direction"] == "BUY" else (t["entry_price"] - ep)
                    pnl = pd_diff * t["lot_size"] * XAUUSD_POINT_VALUE
                    capital += pnl
                    daily_pnl += pnl
                    t.update({"exit_price": round(ep, 2), "exit_time": bt,
                              "pnl": round(pnl, 2),
                              "result": "WIN" if pnl > 0 else "LOSS",
                              "hit": "TP" if hit_tp else "SL"})
                    trades.append(t)
                    open_trade = None

            # Track equity
            equity.append({"datetime": bt, "equity": round(capital, 2)})
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

            if open_trade is not None:
                continue
            if pending_signal is not None:
                continue  # Already have a pending signal waiting for next bar

            # ── Risk checks ───────────────────────────────
            if daily_pnl <= -MAX_DAILY_LOSS or session == "off":
                continue
            if sess_counts[session] >= get_session_limit(session):
                continue
            if last_trade_time and (bt - last_trade_time).total_seconds() < COOLDOWN_SECONDS:
                continue

            # ── Signal generation ─────────────────────────
            prob = probs[i]
            if prob >= prob_threshold:
                direction = "BUY"
                conf = prob_to_confidence(prob)
            elif prob <= (1 - prob_threshold):
                direction = "SELL"
                conf = prob_to_confidence(prob)
            else:
                continue

            sl_dist = SL_ATR_MULT * atr
            rr = TP_RR_HIGH_CONF if conf >= HIGH_CONF_CUTOFF else TP_RR_BASE
            tp_dist = sl_dist * rr

            if execution_mode == "instant":
                # ── INSTANT: enter at bar i close ─────────
                if direction == "BUY":
                    entry = bc + SPREAD_PRICE
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    entry = bc - SPREAD_PRICE
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                lot = min(LOT_SIZE_SINGLE, MAX_TRADE_RISK / (XAUUSD_POINT_VALUE * sl_dist))
                lot = max(LOT_SIZE_MIN, min(lot, LOT_SIZE_MAX))
                lot = round(lot * 100) / 100

                if sl_dist * lot * XAUUSD_POINT_VALUE > MAX_TRADE_RISK:
                    continue

                open_trade = {
                    "direction": direction, "entry_price": round(entry, 2),
                    "sl_price": round(sl, 2), "tp_price": round(tp, 2),
                    "lot_size": lot, "entry_time": bt,
                    "confidence": round(conf, 1),
                    "probability": round(float(prob), 4),
                    "session": session, "atr": round(atr, 4),
                    "sl_pips": round(sl_dist / PIP_VALUE, 1),
                    "tp_pips": round(tp_dist / PIP_VALUE, 1),
                }
                last_trade_time = bt
                sess_counts[session] += 1

            elif execution_mode == "delayed":
                # ── DELAYED: queue signal, execute on next bar open ──
                pending_signal = {
                    "direction": direction,
                    "confidence": round(conf, 1),
                    "probability": round(float(prob), 4),
                    "session": session,
                    "sl_dist": sl_dist,
                    "tp_dist": tp_dist,
                    "signal_time": bt,
                    "signal_close": bc,
                }

    # Close remaining
    if open_trade is not None:
        lc = df.iloc[-1]["close"]
        pd_diff = (lc - open_trade["entry_price"]) if open_trade["direction"] == "BUY" else (open_trade["entry_price"] - lc)
        pnl = pd_diff * open_trade["lot_size"] * XAUUSD_POINT_VALUE
        capital += pnl
        open_trade.update({"pnl": round(pnl, 2), "result": "WIN" if pnl > 0 else "LOSS",
                           "hit": "CLOSE", "exit_time": df.iloc[-1]["datetime"]})
        trades.append(open_trade)

    return compute_metrics(trades, equity, capital, max_dd)


def compute_metrics(trades, equity_curve, final_capital, max_dd_pct):
    if not trades:
        return {"total_pnl": 0, "win_rate": 0, "profit_factor": 0,
                "total_trades": 0, "trades_per_month": 0,
                "max_drawdown_pct": 0, "sharpe": 0,
                "avg_win": 0, "avg_loss": 0, "avg_trade": 0,
                "wins": 0, "losses": 0,
                "profitable_months": 0, "total_months": 0,
                "profitable_months_pct": 0,
                "avg_slippage": 0}

    tdf = pd.DataFrame(trades)
    total = len(tdf)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    total_pnl = tdf["pnl"].sum()
    wr = len(wins) / total * 100
    gp = wins["pnl"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")

    ft = pd.Timestamp(tdf["entry_time"].min())
    lt = pd.Timestamp(tdf["entry_time"].max())
    months = max((lt - ft).days / 30.44, 1)
    tpm = total / months

    # Sharpe
    edf = pd.DataFrame(equity_curve)
    edf["datetime"] = pd.to_datetime(edf["datetime"])
    daily_eq = edf.groupby(edf["datetime"].dt.date)["equity"].last()
    dr = daily_eq.pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if len(dr) > 0 and dr.std() > 0 else 0

    # Monthly
    tdf["month"] = pd.to_datetime(tdf["entry_time"]).dt.to_period("M")
    mpnl = tdf.groupby("month")["pnl"].sum()
    pm = (mpnl > 0).sum()
    tm = len(mpnl)

    # Average slippage (delayed mode only)
    avg_slip = tdf["slippage"].mean() if "slippage" in tdf.columns else 0

    # Confidence separation
    cw = wins["confidence"].mean() if len(wins) > 0 else 0
    cl = losses["confidence"].mean() if len(losses) > 0 else 0

    return {
        "total_pnl": round(total_pnl, 2),
        "pnl_pct": round(total_pnl / INITIAL_CAPITAL * 100, 2),
        "final_capital": round(final_capital, 2),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "total_trades": total,
        "trades_per_month": round(tpm, 1),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "sharpe": round(sharpe, 2),
        "avg_win": round(wins["pnl"].mean() if len(wins) > 0 else 0, 2),
        "avg_loss": round(losses["pnl"].mean() if len(losses) > 0 else 0, 2),
        "avg_trade": round(tdf["pnl"].mean(), 2),
        "wins": len(wins),
        "losses": len(losses),
        "profitable_months": pm,
        "total_months": tm,
        "profitable_months_pct": round(pm / max(tm, 1) * 100, 1),
        "avg_slippage": round(avg_slip, 4),
        "avg_conf_winning": round(cw, 1),
        "avg_conf_losing": round(cl, 1),
        "monthly_pnl": {str(k): round(v, 2) for k, v in mpnl.items()},
    }


def main():
    print("=" * 70)
    print("MIDAS v2 — Strategy E: EXECUTION REALITY TEST")
    print("  INSTANT = enter at bar i close (optimistic)")
    print("  DELAYED = signal at bar i, enter at bar i+1 open (realistic)")
    print("=" * 70)

    # Load clean data
    print("\n  Loading clean data...")
    data_path = DATA_PROCESSED / "strategy_e_train_ready_clean.parquet"
    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    meta_path = MODELS_DIR / "strategy_e_clean_metadata.json"
    with open(meta_path) as f:
        feature_cols = json.load(f)["feature_cols"]
    print(f"  Data: {len(df):,} rows, {len(feature_cols)} features")

    # Verify models
    for seg in WF_MODEL_MAP:
        p = MODELS_DIR / seg["model"]
        print(f"  {seg['model']}: {'✅' if p.exists() else '❌'}")

    # Run both modes
    instant_results = {}
    delayed_results = {}

    for thresh in THRESHOLDS_TO_TEST:
        print(f"\n  Threshold {thresh:.2f}:")

        print(f"    INSTANT...", end=" ")
        r_inst = run_backtest(df, feature_cols, thresh, "instant")
        instant_results[thresh] = r_inst
        print(f"PnL=${r_inst['total_pnl']:+,.0f}  WR={r_inst['win_rate']:.1f}%  "
              f"PF={r_inst['profit_factor']:.2f}  T={r_inst['total_trades']}")

        print(f"    DELAYED...", end=" ")
        r_del = run_backtest(df, feature_cols, thresh, "delayed")
        delayed_results[thresh] = r_del
        print(f"PnL=${r_del['total_pnl']:+,.0f}  WR={r_del['win_rate']:.1f}%  "
              f"PF={r_del['profit_factor']:.2f}  T={r_del['total_trades']}  "
              f"Slip=${r_del['avg_slippage']:.2f}")

    # ── Side-by-side comparison ───────────────────────────
    print(f"\n{'=' * 115}")
    print(f"{'':>7} | {'─── INSTANT (bar i close) ───':^38s} | {'─── DELAYED (bar i+1 open) ───':^38s} | {'─ DIFF ─':^12s}")
    print(f"{'Thresh':>7} | {'PnL':>9} {'WR%':>6} {'PF':>5} {'T':>5} {'Sharpe':>7} | "
          f"{'PnL':>9} {'WR%':>6} {'PF':>5} {'T':>5} {'Sharpe':>7} | {'PnL Δ':>7} {'%kept':>5}")
    print("-" * 115)

    for t in sorted(THRESHOLDS_TO_TEST):
        ri = instant_results[t]
        rd = delayed_results[t]
        pnl_diff = rd["total_pnl"] - ri["total_pnl"]
        pct_kept = (rd["total_pnl"] / ri["total_pnl"] * 100) if ri["total_pnl"] > 0 else 0

        print(f"  {t:.2f}  | "
              f"${ri['total_pnl']:>8,.0f} {ri['win_rate']:>5.1f} {ri['profit_factor']:>5.2f} "
              f"{ri['total_trades']:>5} {ri['sharpe']:>6.1f} | "
              f"${rd['total_pnl']:>8,.0f} {rd['win_rate']:>5.1f} {rd['profit_factor']:>5.2f} "
              f"{rd['total_trades']:>5} {rd['sharpe']:>6.1f} | "
              f"${pnl_diff:>+6,.0f} {pct_kept:>4.0f}%")

    print("=" * 115)

    # ── Best delayed threshold ────────────────────────────
    best_t = max(delayed_results.keys(), key=lambda t: delayed_results[t]["total_pnl"])
    best_d = delayed_results[best_t]
    best_i = instant_results[best_t]

    print(f"\n{'─' * 70}")
    print(f"  BEST DELAYED THRESHOLD = {best_t:.2f}")
    print(f"{'─' * 70}")
    print(f"  PnL:              ${best_d['total_pnl']:+,.2f} ({best_d['pnl_pct']:+.1f}%)")
    print(f"  Win Rate:         {best_d['win_rate']:.1f}%  ({best_d['wins']}W / {best_d['losses']}L)")
    print(f"  Profit Factor:    {best_d['profit_factor']:.2f}")
    print(f"  Trades/Month:     {best_d['trades_per_month']:.1f}")
    print(f"  Max Drawdown:     {best_d['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe:           {best_d['sharpe']:.2f}")
    print(f"  Profitable Months: {best_d['profitable_months']}/{best_d['total_months']}")
    print(f"  Avg Slippage:     ${best_d['avg_slippage']:.4f}")
    print(f"  Conf Separation:  {best_d['avg_conf_winning'] - best_d['avg_conf_losing']:+.1f}%")

    print(f"\n  vs Instant: PnL kept = "
          f"{best_d['total_pnl']/max(best_i['total_pnl'],1)*100:.0f}%")

    # Monthly PnL
    print(f"\n  Monthly PnL (DELAYED):")
    for m, p in best_d.get("monthly_pnl", {}).items():
        print(f"    {m}: ${p:>+8.2f} {'✅' if p > 0 else '❌'}")

    # ── Final verdict ─────────────────────────────────────
    pct_retained = best_d["total_pnl"] / max(best_i["total_pnl"], 1) * 100

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    if pct_retained >= 80:
        print(f"  ✅ STRONG: {pct_retained:.0f}% of PnL retained with realistic execution.")
        print("  The edge is REAL and survives execution delay.")
    elif pct_retained >= 50:
        print(f"  🟡 MODERATE: {pct_retained:.0f}% of PnL retained.")
        print("  Partial edge exists. Useful as validator, cautious as independent.")
    elif pct_retained >= 20:
        print(f"  🟠 WEAK: Only {pct_retained:.0f}% retained.")
        print("  Most edge is execution-dependent. Validator role only.")
    else:
        print(f"  ❌ FAILED: Only {pct_retained:.0f}% retained.")
        print("  Edge doesn't survive execution delay. Signal is stale.")

    # ── Strategy comparison ───────────────────────────────
    print(f"\n{'=' * 70}")
    print("Strategy Comparison (Forward Test) — REALISTIC EXECUTION")
    print(f"{'=' * 70}")
    print(f"  {'Strategy':<28s} {'PnL':>9} {'WR%':>6} {'PF':>5} {'T/Mo':>6} {'MaxDD':>6}")
    print(f"  {'-'*61}")
    print(f"  {'A (SMC OB)':<28s} ${'2,092':>7} {'49.4':>5} {'1.57':>5} {'51.4':>5} {'2.5%':>6}")
    print(f"  {'B (FVG)':<28s} ${'663':>7} {'70.4':>5} {'1.71':>5} {'32.6':>5} {'2.5%':>6}")
    print(f"  {'C (Session)':<28s} ${'497':>7} {'51.6':>5} {'1.25':>5} {'11.0':>5} {'6.2%':>6}")
    print(f"  {'D (Trend)':<28s} ${'36':>7} {'41.8':>5} {'1.17':>5} {'2.7':>5} {'1.2%':>6}")
    print(f"  {'A+B+C+D Combined':<28s} ${'2,732':>7} {'57.6':>5} {'1.55':>5} {'64.0':>5} {'3.5%':>6}")
    print(f"  {'E (DELAYED) t=' + f'{best_t:.2f}':<28s} "
          f"${best_d['total_pnl']:>7,.0f} "
          f"{best_d['win_rate']:>5.1f} "
          f"{best_d['profit_factor']:>5.2f} "
          f"{best_d['trades_per_month']:>5.1f} "
          f"{best_d['max_drawdown_pct']:>5.1f}%")

    # ── Save ──────────────────────────────────────────────
    save_data = {
        "test": "execution_reality_check",
        "instant": {str(t): {k: v for k, v in r.items() if k != "monthly_pnl"}
                    for t, r in instant_results.items()},
        "delayed": {str(t): {k: v for k, v in r.items() if k != "monthly_pnl"}
                    for t, r in delayed_results.items()},
        "best_delayed_threshold": best_t,
        "pnl_retained_pct": round(pct_retained, 1),
        "delayed_monthly_pnl": best_d.get("monthly_pnl", {}),
    }
    save_path = BACKTEST_DIR / "strategy_e_execution_test.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved: {save_path.name}")

    print(f"\n{'=' * 70}")
    print("✅ Execution test complete!")
    if pct_retained >= 50:
        print("   Edge survives! Ready for A+B+C+D+E combined test.")
    else:
        print("   Edge is execution-dependent. Rethink before combining.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()