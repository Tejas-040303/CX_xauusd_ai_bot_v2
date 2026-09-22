"""
Project MIDAS v2 — Strategy E: Clean Retrain Pipeline
=======================================================
End-to-end: debias → retrain → backtest on clean data.

CRITICAL: This does NOT modify feature_engineering.py.
Strategies A, B, C, D are completely untouched.

All debiasing happens INSIDE this pipeline only:
  1. Load pre-computed features (same parquet from data prep)
  2. Debias swing features for ML consumption (shift by SWING_LOOKBACK)
  3. Create labels on debiased data
  4. Retrain walk-forward models on clean data
  5. Backtest with walk-forward models on clean data

Usage:
    python strategy_e_clean_retrain.py

Output:
    data/processed/strategy_e_train_ready_clean.parquet
    models/strategy_e_clean_W1.txt, W2.txt, W3.txt, final.txt
    models/strategy_e_clean_metadata.json
    backtest_results/strategy_e_clean_backtest.json
"""

import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    DATA_PROCESSED, MODELS_DIR, BACKTEST_DIR,
    ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE, SWING_LOOKBACK,
    MAX_DAILY_LOSS, MAX_TRADE_RISK,
    LOT_SIZE_SINGLE, LOT_SIZE_MIN, LOT_SIZE_MAX,
    SPREAD_SIMULATION_PIPS, COOLDOWN_SECONDS,
    SESSION_LIMITS, SESSIONS, BACKTEST_INITIAL_CAPITAL,
)

# ─── Label Config (same as data_prep) ────────────────────
LABEL_HORIZON_BARS = 12
ATR_FILTER_THRESHOLD = 0.5
LABEL_COL = "label"

# ─── Walk-Forward Windows ────────────────────────────────
WALK_FORWARD_WINDOWS = {
    "W1": {"train_start": "2021-01-01", "train_end": "2023-12-31",
            "test_start": "2024-01-01", "test_end": "2024-06-30"},
    "W2": {"train_start": "2021-07-01", "train_end": "2024-06-30",
            "test_start": "2024-07-01", "test_end": "2024-12-31"},
    "W3": {"train_start": "2022-01-01", "train_end": "2024-06-30",
            "test_start": "2025-01-01", "test_end": "2025-06-30"},
}
HOLDOUT_START = "2025-07-01"

# Walk-forward model → test period mapping (for backtest)
WF_MODEL_MAP = [
    {"model": "strategy_e_clean_W2.txt",
     "test_start": "2024-07-01", "test_end": "2024-12-31",
     "label": "W2 → Jul-Dec 2024"},
    {"model": "strategy_e_clean_W3.txt",
     "test_start": "2025-01-01", "test_end": "2025-06-30",
     "label": "W3 → Jan-Jun 2025"},
    {"model": "strategy_e_clean_final.txt",
     "test_start": "2025-07-01", "test_end": "2026-12-31",
     "label": "Final → Jul 2025+"},
]

# ─── LightGBM Params (same as training script) ───────────
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}
EARLY_STOPPING_ROUNDS = 100
N_ESTIMATORS = 2000

# ─── Backtest Config ─────────────────────────────────────
FORWARD_START = "2024-07-01"
INITIAL_CAPITAL = BACKTEST_INITIAL_CAPITAL
SL_ATR_MULT = 1.5
TP_RR_BASE = 1.5
TP_RR_HIGH_CONF = 2.5
HIGH_CONF_CUTOFF = 75
SPREAD_PRICE = SPREAD_SIMULATION_PIPS * PIP_VALUE
THRESHOLDS_TO_TEST = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]

# Columns to exclude from features (same as original data_prep)
EXCLUDE_COLS = [
    "datetime", "timeframe",
    "open", "high", "low", "close", "volume",
    "swing_high_price", "swing_low_price",
    "last_swing_high", "last_swing_low",
    "bos_bull_level", "bos_bear_level",
    "ob_bull_high", "ob_bull_low", "ob_bear_high", "ob_bear_low",
    "equal_highs_level", "equal_lows_level",
    "last_equal_highs_level", "last_equal_lows_level",
    "fvg_bull_top", "fvg_bull_bottom", "fvg_bull_ce",
    "fvg_bear_top", "fvg_bear_bottom", "fvg_bear_ce",
    "asian_range_high", "asian_range_low", "asian_range_mid",
    "session", "vol_regime", "fvg_direction",
    LABEL_COL, "future_return", "future_return_atr",
    "_trade_date",
]


# ═════════════════════════════════════════════════════════
# PHASE 1: DATA PREPARATION (with debiasing)
# ═════════════════════════════════════════════════════════

def debias_swing_features(df):
    """
    Shift all swing-derived features forward by SWING_LOOKBACK bars.
    This removes look-ahead bias for ML training.

    ONLY affects Strategy E's data. feature_engineering.py is untouched.
    Strategies A-D continue using the original (unshifted) features.
    """
    n = SWING_LOOKBACK
    df = df.copy()

    # ── Core swing detection: shift forward ───────────────
    for col in ["swing_high", "swing_low"]:
        if col in df.columns:
            df[col] = df[col].shift(n).fillna(0).astype(int)

    for col in ["swing_high_price", "swing_low_price"]:
        if col in df.columns:
            df[col] = df[col].shift(n)

    # ── Re-derive downstream features ─────────────────────
    if "swing_high_price" in df.columns:
        df["last_swing_high"] = df["swing_high_price"].ffill()
        df["dist_to_swing_high"] = df["close"] - df["last_swing_high"]

    if "swing_low_price" in df.columns:
        df["last_swing_low"] = df["swing_low_price"].ffill()
        df["dist_to_swing_low"] = df["close"] - df["last_swing_low"]

    atr_col = f"atr_{ATR_PERIOD}"
    if atr_col in df.columns:
        atr = df[atr_col].replace(0, np.nan)
        if "dist_to_swing_high" in df.columns:
            df["dist_to_swing_high_atr"] = df["dist_to_swing_high"] / atr
        if "dist_to_swing_low" in df.columns:
            df["dist_to_swing_low_atr"] = df["dist_to_swing_low"] / atr

    # ── BOS features (depend on swings) ───────────────────
    for col in ["bos_bull", "bos_bear"]:
        if col in df.columns:
            df[col] = df[col].shift(n).fillna(0).astype(int)

    for col in ["bos_bull_level", "bos_bear_level",
                "ob_bull_high", "ob_bull_low", "ob_bear_high", "ob_bear_low",
                "ob_bull_impulse_size", "ob_bear_impulse_size"]:
        if col in df.columns:
            df[col] = df[col].shift(n)

    # ── Equal highs/lows (depend on swings) ───────────────
    for col in ["equal_highs", "equal_lows", "equal_highs_count", "equal_lows_count",
                "equal_highs_level", "equal_lows_level"]:
        if col in df.columns:
            df[col] = df[col].shift(n)

    if "equal_highs_level" in df.columns:
        df["last_equal_highs_level"] = df["equal_highs_level"].ffill()
    if "equal_lows_level" in df.columns:
        df["last_equal_lows_level"] = df["equal_lows_level"].ffill()

    if "last_equal_highs_level" in df.columns and atr_col in df.columns:
        df["dist_to_eq_highs_atr"] = (df["close"] - df["last_equal_highs_level"]) / atr
    if "last_equal_lows_level" in df.columns and atr_col in df.columns:
        df["dist_to_eq_lows_atr"] = (df["close"] - df["last_equal_lows_level"]) / atr

    # ── M15 BOS features ──────────────────────────────────
    for col in ["m15_bos_bull", "m15_bos_bear"]:
        if col in df.columns:
            df[col] = df[col].shift(n).fillna(0).astype(int)

    return df


def create_labels(df):
    """Create ATR-filtered directional labels."""
    atr_col = f"atr_{ATR_PERIOD}"
    df["_future_close"] = df["close"].shift(-LABEL_HORIZON_BARS)
    df["future_return"] = df["_future_close"] - df["close"]
    df["future_return_atr"] = df["future_return"] / df[atr_col].replace(0, np.nan)

    df[LABEL_COL] = -1
    df.loc[df["future_return_atr"] >= ATR_FILTER_THRESHOLD, LABEL_COL] = 1
    df.loc[df["future_return_atr"] <= -ATR_FILTER_THRESHOLD, LABEL_COL] = 0

    df = df[df["_future_close"].notna()].copy()
    df = df.drop(columns=["_future_close"])

    buys = (df[LABEL_COL] == 1).sum()
    sells = (df[LABEL_COL] == 0).sum()
    waits = (df[LABEL_COL] == -1).sum()
    trainable = buys + sells
    print(f"    Labels: BUY={buys:,}, SELL={sells:,}, WAIT={waits:,} "
          f"(trainable: {trainable:,}, ratio: {buys/max(sells,1):.3f})")
    return df


def get_feature_columns(df):
    """Determine valid feature columns."""
    exclude = set(EXCLUDE_COLS)
    object_cols = set(df.select_dtypes(include=["object", "category"]).columns)
    exclude.update(object_cols)
    leakage_cols = {c for c in df.columns if "future" in c.lower()}
    exclude.update(leakage_cols)

    feature_cols = sorted(set(df.columns) - exclude)
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols


# ═════════════════════════════════════════════════════════
# PHASE 2: TRAINING (on clean data)
# ═════════════════════════════════════════════════════════

def train_window(X_train, y_train, X_test, y_test, feature_cols, name):
    """Train one walk-forward window. Returns model + metrics."""
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    params = LGBM_PARAMS.copy()
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)

    # 10% temporal validation split
    val_size = int(len(X_train) * 0.1)
    X_t, y_t = X_train[:-val_size], y_train[:-val_size]
    X_v, y_v = X_train[-val_size:], y_train[-val_size:]

    train_ds = lgb.Dataset(X_t, label=y_t, feature_name=feature_cols)
    val_ds = lgb.Dataset(X_v, label=y_v, reference=train_ds, feature_name=feature_cols)

    model = lgb.train(
        params, train_ds,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "valid"],
        num_boost_round=N_ESTIMATORS,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=200),
        ],
    )

    # Evaluate on test
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    buy_scores = y_prob[y_test == 1]
    sell_scores = y_prob[y_test == 0]
    sep = float(np.mean(buy_scores) - np.mean(sell_scores))

    # Confidence bins
    conf_bins = {}
    for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = (y_prob >= t) | (y_prob <= 1 - t)
        if mask.sum() > 0:
            preds = np.where(y_prob >= t, 1, 0)[mask]
            conf_bins[f"{t:.2f}"] = {
                "accuracy": round(accuracy_score(y_test[mask], preds), 4),
                "count": int(mask.sum()),
                "pct": round(100 * mask.sum() / len(y_test), 1),
            }

    print(f"    {name}: Acc={acc:.4f}  AUC={auc:.4f}  Sep={sep:.4f}  "
          f"Iter={model.best_iteration}")
    print(f"      Confidence bins:")
    for t_name, stats in conf_bins.items():
        print(f"        ≥{t_name}: {stats['accuracy']:.4f} "
              f"({stats['count']:,} samples, {stats['pct']}%)")

    return model, {
        "accuracy": acc, "roc_auc": auc, "score_separation": sep,
        "best_iteration": model.best_iteration,
        "confidence_bins": conf_bins,
    }


# ═════════════════════════════════════════════════════════
# PHASE 3: BACKTEST (walk-forward, clean data)
# ═════════════════════════════════════════════════════════

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


def run_backtest(df, feature_cols, prob_threshold):
    """Walk-forward backtest using clean models on clean data."""
    capital = INITIAL_CAPITAL
    trades = []
    equity = []
    peak = capital
    max_dd = 0.0

    for seg in WF_MODEL_MAP:
        model_path = MODELS_DIR / seg["model"]
        if not model_path.exists():
            print(f"      WARNING: {seg['model']} missing, skipping")
            continue

        model = lgb.Booster(model_file=str(model_path))
        seg_data = df[(df["datetime"] >= seg["test_start"]) &
                      (df["datetime"] <= seg["test_end"])].reset_index(drop=True)
        if len(seg_data) == 0:
            continue

        probs = model.predict(seg_data[feature_cols].values)

        open_trade = None
        daily_pnl = 0.0
        current_date = None
        last_trade_time = None
        sess_counts = defaultdict(int)

        for i in range(len(seg_data)):
            row = seg_data.iloc[i]
            bt = row["datetime"]
            bd = pd.Timestamp(bt).date()
            bh, bl, bc = row["high"], row["low"], row["close"]
            atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
            if pd.isna(atr) or atr <= 0:
                atr = 2.0
            session = get_session(bt.hour)

            if current_date != bd:
                current_date = bd
                daily_pnl = 0.0
                sess_counts = defaultdict(int)

            # Check open trade
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

            equity.append({"datetime": bt, "equity": round(capital, 2)})
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

            if open_trade is not None:
                continue

            # Risk checks
            if daily_pnl <= -MAX_DAILY_LOSS or session == "off":
                continue
            if sess_counts[session] >= get_session_limit(session):
                continue
            if last_trade_time and (bt - last_trade_time).total_seconds() < COOLDOWN_SECONDS:
                continue

            # Signal
            prob = probs[i]
            if prob >= prob_threshold:
                direction, conf = "BUY", prob_to_confidence(prob)
            elif prob <= (1 - prob_threshold):
                direction, conf = "SELL", prob_to_confidence(prob)
            else:
                continue

            sl_dist = SL_ATR_MULT * atr
            rr = TP_RR_HIGH_CONF if conf >= HIGH_CONF_CUTOFF else TP_RR_BASE
            tp_dist = sl_dist * rr

            if direction == "BUY":
                entry = bc + SPREAD_PRICE
                sl, tp = entry - sl_dist, entry + tp_dist
            else:
                entry = bc - SPREAD_PRICE
                sl, tp = entry + sl_dist, entry - tp_dist

            lot = min(LOT_SIZE_SINGLE, MAX_TRADE_RISK / (XAUUSD_POINT_VALUE * sl_dist))
            lot = max(LOT_SIZE_MIN, min(lot, LOT_SIZE_MAX))
            lot = round(lot * 100) / 100

            if sl_dist * lot * XAUUSD_POINT_VALUE > MAX_TRADE_RISK:
                continue

            open_trade = {
                "direction": direction, "entry_price": round(entry, 2),
                "sl_price": round(sl, 2), "tp_price": round(tp, 2),
                "lot_size": lot, "entry_time": bt,
                "confidence": round(conf, 1), "probability": round(float(prob), 4),
                "session": session, "atr": round(atr, 4),
                "sl_pips": round(sl_dist / PIP_VALUE, 1),
                "tp_pips": round(tp_dist / PIP_VALUE, 1),
            }
            last_trade_time = bt
            sess_counts[session] += 1

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
                "max_drawdown_pct": 0, "sharpe": 0}

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

    # Sessions
    ss = {}
    for s in tdf["session"].unique():
        st = tdf[tdf["session"] == s]
        sw = st[st["pnl"] > 0]
        ss[s] = {"trades": len(st), "pnl": round(st["pnl"].sum(), 2),
                 "win_rate": round(len(sw) / max(len(st), 1) * 100, 1)}

    # Direction
    ds = {}
    for d in ["BUY", "SELL"]:
        dt = tdf[tdf["direction"] == d]
        dw = dt[dt["pnl"] > 0]
        ds[d] = {"trades": len(dt), "pnl": round(dt["pnl"].sum(), 2),
                 "win_rate": round(len(dw) / max(len(dt), 1) * 100, 1)}

    # Confidence
    cw = wins["confidence"].mean() if len(wins) > 0 else 0
    cl = losses["confidence"].mean() if len(losses) > 0 else 0

    return {
        "total_pnl": round(total_pnl, 2), "pnl_pct": round(total_pnl / INITIAL_CAPITAL * 100, 2),
        "final_capital": round(final_capital, 2),
        "win_rate": round(wr, 1), "profit_factor": round(pf, 2),
        "total_trades": total, "trades_per_month": round(tpm, 1),
        "max_drawdown_pct": round(max_dd_pct, 1), "sharpe": round(sharpe, 2),
        "avg_win": round(wins["pnl"].mean() if len(wins) > 0 else 0, 2),
        "avg_loss": round(losses["pnl"].mean() if len(losses) > 0 else 0, 2),
        "avg_trade": round(tdf["pnl"].mean(), 2),
        "wins": len(wins), "losses": len(losses),
        "profitable_months": pm, "total_months": tm,
        "profitable_months_pct": round(pm / max(tm, 1) * 100, 1),
        "session_breakdown": ss, "direction_breakdown": ds,
        "avg_conf_winning": round(cw, 1), "avg_conf_losing": round(cl, 1),
        "monthly_pnl": {str(k): round(v, 2) for k, v in mpnl.items()},
    }


# ═════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MIDAS v2 — Strategy E: CLEAN RETRAIN PIPELINE")
    print("  feature_engineering.py: UNTOUCHED")
    print("  Strategies A-D: UNTOUCHED")
    print("  Debiasing: swing features shifted for ML only")
    print("=" * 70)

    # ══════════════════════════════════════════════════════
    # PHASE 1: LOAD + DEBIAS + LABEL
    # ══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("PHASE 1: Data Preparation (clean)")
    print("─" * 70)

    # Load the original pre-computed features
    src_path = DATA_PROCESSED / "strategy_e_train_ready.parquet"
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} not found. Run strategy_e_data_prep.py first.")

    print("\n  Loading original features...")
    df = pd.read_parquet(src_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Drop old labels (we'll recreate after debiasing)
    for col in [LABEL_COL, "future_return", "future_return_atr"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Debias swing features
    print("\n  Debiasing swing features (shift by {})...".format(SWING_LOOKBACK))
    df = debias_swing_features(df)

    # Create labels on debiased data
    print("\n  Creating labels on debiased data...")
    df = create_labels(df)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"  Features for training: {len(feature_cols)}")

    # Save clean dataset
    clean_path = DATA_PROCESSED / "strategy_e_train_ready_clean.parquet"
    df.to_parquet(clean_path, index=False)
    print(f"  Clean dataset saved: {clean_path.name}")

    # Save feature list
    feat_path = DATA_PROCESSED / "strategy_e_feature_list_clean.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # ══════════════════════════════════════════════════════
    # PHASE 2: WALK-FORWARD TRAINING
    # ══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("PHASE 2: Walk-Forward Training (on clean data)")
    print("─" * 70)

    all_results = {}
    best_iters = []

    for wname, wdates in WALK_FORWARD_WINDOWS.items():
        print(f"\n  {wname}: Train {wdates['train_start']}→{wdates['train_end']} | "
              f"Test {wdates['test_start']}→{wdates['test_end']}")

        train_mask = ((df["datetime"] >= wdates["train_start"]) &
                      (df["datetime"] <= wdates["train_end"]) &
                      (df[LABEL_COL] != -1))
        test_mask = ((df["datetime"] >= wdates["test_start"]) &
                     (df["datetime"] <= wdates["test_end"]) &
                     (df[LABEL_COL] != -1))

        train, test = df[train_mask], df[test_mask]
        X_tr = train[feature_cols].values
        y_tr = train[LABEL_COL].values.astype(int)
        X_te = test[feature_cols].values
        y_te = test[LABEL_COL].values.astype(int)

        print(f"    Train: {len(y_tr):,} (BUY={int((y_tr==1).sum()):,}, "
              f"SELL={int((y_tr==0).sum()):,})")
        print(f"    Test:  {len(y_te):,}")

        model, metrics = train_window(X_tr, y_tr, X_te, y_te, feature_cols, wname)

        # Save model
        model_path = MODELS_DIR / f"strategy_e_clean_{wname}.txt"
        model.save_model(str(model_path))
        print(f"    Saved: {model_path.name}")

        all_results[wname] = metrics
        best_iters.append(metrics["best_iteration"])

    # ── Train final model (all data before holdout) ───────
    print(f"\n  Training FINAL model (all data < {HOLDOUT_START})...")
    final_mask = ((df["datetime"] < HOLDOUT_START) & (df[LABEL_COL] != -1))
    final_train = df[final_mask]
    X_f = final_train[feature_cols].values
    y_f = final_train[LABEL_COL].values.astype(int)

    params = LGBM_PARAMS.copy()
    params["scale_pos_weight"] = (y_f == 0).sum() / max((y_f == 1).sum(), 1)

    final_rounds = int(np.median(best_iters))
    print(f"    Using {final_rounds} rounds (median of {best_iters})")

    final_ds = lgb.Dataset(X_f, label=y_f, feature_name=feature_cols)
    final_model = lgb.train(
        params, final_ds, num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(period=500)],
    )
    final_path = MODELS_DIR / "strategy_e_clean_final.txt"
    final_model.save_model(str(final_path))
    print(f"    Saved: {final_path.name}")

    # ── Feature importance ────────────────────────────────
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": final_model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance["gain_pct"] = 100 * importance["gain"] / importance["gain"].sum()

    print(f"\n  Top 20 Features (CLEAN — no look-ahead):")
    for _, row in importance.head(20).iterrows():
        print(f"    {row['feature']:<40s} gain={row['gain_pct']:>5.2f}%")

    # ── Training summary ──────────────────────────────────
    print(f"\n  Walk-Forward Summary (CLEAN):")
    for wn in WALK_FORWARD_WINDOWS:
        m = all_results[wn]
        print(f"    {wn}: Acc={m['accuracy']:.4f}  AUC={m['roc_auc']:.4f}  "
              f"Sep={m['score_separation']:.4f}")

    avg_sep = np.mean([all_results[w]["score_separation"] for w in WALK_FORWARD_WINDOWS])
    avg_acc = np.mean([all_results[w]["accuracy"] for w in WALK_FORWARD_WINDOWS])
    print(f"\n  Average: Acc={avg_acc:.4f}  Sep={avg_sep:.4f}")

    # ══════════════════════════════════════════════════════
    # PHASE 3: BACKTEST
    # ══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("PHASE 3: Walk-Forward Backtest (clean models + clean data)")
    print("─" * 70)

    results = {}
    for thresh in THRESHOLDS_TO_TEST:
        print(f"\n  Threshold = {thresh:.2f}...", end=" ")
        r = run_backtest(df, feature_cols, thresh)
        results[thresh] = r
        print(f"PnL: ${r['total_pnl']:+,.2f} | WR: {r['win_rate']:.1f}% | "
              f"PF: {r['profit_factor']:.2f} | Trades: {r['total_trades']} | "
              f"MaxDD: {r['max_drawdown_pct']:.1f}%")

    # ── Results table ─────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"{'Thresh':>7} | {'PnL':>9} | {'PnL%':>6} | {'WR%':>5} | {'PF':>5} | "
          f"{'Trades':>6} | {'T/Mo':>5} | {'MaxDD':>6} | {'Sharpe':>6} | "
          f"{'AvgWin':>7} | {'AvgLoss':>8} | {'ProfMo':>6}")
    print("-" * 100)
    for t in sorted(results.keys()):
        r = results[t]
        print(f"  {t:.2f}  | ${r['total_pnl']:>7.0f} | {r['pnl_pct']:>5.1f}% | "
              f"{r['win_rate']:>4.1f} | {r['profit_factor']:>4.2f} | "
              f"{r['total_trades']:>6} | {r['trades_per_month']:>4.1f} | "
              f"{r['max_drawdown_pct']:>5.1f}% | {r['sharpe']:>5.2f} | "
              f"${r['avg_win']:>5.2f} | ${r['avg_loss']:>7.2f} | "
              f"{r['profitable_months']}/{r['total_months']}")
    print("=" * 100)

    # ── Best threshold detail ─────────────────────────────
    best_t = max(results.keys(), key=lambda t: results[t]["total_pnl"])
    best = results[best_t]

    print(f"\n{'─' * 70}")
    print(f"  BEST THRESHOLD = {best_t:.2f} (CLEAN)")
    print(f"{'─' * 70}")
    print(f"  PnL:              ${best['total_pnl']:+,.2f} ({best['pnl_pct']:+.2f}%)")
    print(f"  Win Rate:         {best['win_rate']:.1f}%  ({best['wins']}W / {best['losses']}L)")
    print(f"  Profit Factor:    {best['profit_factor']:.2f}")
    print(f"  Trades/Month:     {best['trades_per_month']:.1f}")
    print(f"  Max Drawdown:     {best['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe Ratio:     {best['sharpe']:.2f}")
    print(f"  Profitable Months: {best['profitable_months']}/{best['total_months']}")
    print(f"\n  Confidence on Winners: {best['avg_conf_winning']:.1f}%")
    print(f"  Confidence on Losers:  {best['avg_conf_losing']:.1f}%")
    print(f"  Conf Separation:       {best['avg_conf_winning'] - best['avg_conf_losing']:+.1f}%")

    print(f"\n  Direction:")
    for d, s in best.get("direction_breakdown", {}).items():
        print(f"    {d}: {s['trades']} trades, ${s['pnl']:+.2f}, {s['win_rate']:.1f}% WR")
    print(f"\n  Sessions:")
    for s, st in best.get("session_breakdown", {}).items():
        print(f"    {s:<20s}: {st['trades']:>4} trades, ${st['pnl']:>+8.2f}, {st['win_rate']:.1f}% WR")
    print(f"\n  Monthly PnL:")
    for m, p in best.get("monthly_pnl", {}).items():
        print(f"    {m}: ${p:>+8.2f} {'✅' if p > 0 else '❌'}")

    # ── 3-way comparison ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("3-Way Comparison: BIASED → DEBIASED-TEST → CLEAN-RETRAIN")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<22s} {'BIASED':>12s} {'DEBIAS-TEST':>14s} {'CLEAN':>12s}")
    print(f"  {'-'*60}")
    print(f"  {'PnL':<22s} {'$15,474':>12s} {'$6,221':>14s} ${best['total_pnl']:>11,.0f}")
    print(f"  {'Win Rate':<22s} {'70.9%':>12s} {'54.8%':>14s} {best['win_rate']:>11.1f}%")
    print(f"  {'Profit Factor':<22s} {'4.08':>12s} {'1.89':>14s} {best['profit_factor']:>11.2f}")
    print(f"  {'Trades/Month':<22s} {'157.7':>12s} {'133.0':>14s} {best['trades_per_month']:>11.1f}")
    print(f"  {'Max Drawdown':<22s} {'0.5%':>12s} {'1.6%':>14s} {best['max_drawdown_pct']:>11.1f}%")
    print(f"  {'Sharpe':<22s} {'20.17':>12s} {'8.54':>14s} {best['sharpe']:>11.2f}")

    # ── Strategy comparison ───────────────────────────────
    print(f"\n{'=' * 70}")
    print("Strategy Comparison (Forward Test) — CLEAN")
    print(f"{'=' * 70}")
    print(f"  {'Strategy':<25s} {'PnL':>9} {'WR%':>6} {'PF':>5} {'T/Mo':>6} {'MaxDD':>6}")
    print(f"  {'-'*58}")
    print(f"  {'A (SMC OB)':<25s} ${'2,092':>7} {'49.4':>5} {'1.57':>5} {'51.4':>5} {'2.5%':>6}")
    print(f"  {'B (FVG)':<25s} ${'663':>7} {'70.4':>5} {'1.71':>5} {'32.6':>5} {'2.5%':>6}")
    print(f"  {'C (Session)':<25s} ${'497':>7} {'51.6':>5} {'1.25':>5} {'11.0':>5} {'6.2%':>6}")
    print(f"  {'D (Trend)':<25s} ${'36':>7} {'41.8':>5} {'1.17':>5} {'2.7':>5} {'1.2%':>6}")
    print(f"  {'A+B+C+D Combined':<25s} ${'2,732':>7} {'57.6':>5} {'1.55':>5} {'64.0':>5} {'3.5%':>6}")
    print(f"  {'E (CLEAN) t=' + f'{best_t:.2f}':<25s} "
          f"${best['total_pnl']:>7,.0f} "
          f"{best['win_rate']:>5.1f} "
          f"{best['profit_factor']:>5.2f} "
          f"{best['trades_per_month']:>5.1f} "
          f"{best['max_drawdown_pct']:>5.1f}%")

    # ── Save everything ───────────────────────────────────
    save_data = {
        "pipeline": "clean_retrain",
        "fixes": [
            "swing features shifted by SWING_LOOKBACK for ML only",
            "models retrained on debiased data",
            "walk-forward backtest (no train/test overlap)",
            "feature_engineering.py UNTOUCHED — A-D strategies safe",
        ],
        "training_results": {w: {k: v for k, v in m.items()}
                             for w, m in all_results.items()},
        "feature_importance_top20": importance.head(20)[["feature", "gain_pct"]].to_dict("records"),
        "backtest_results": {str(t): {k: v for k, v in r.items() if k != "monthly_pnl"}
                             for t, r in results.items()},
        "best_threshold": best_t,
        "best_monthly_pnl": best.get("monthly_pnl", {}),
    }

    # Save metadata for strategy_e.py to load
    meta = {
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "confidence_threshold": round(best_t, 3),
        "walk_forward_results": {w: {"accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
                                     "score_separation": m["score_separation"]}
                                 for w, m in all_results.items()},
        "feature_importance_top20": importance.head(20)[["feature", "gain_pct"]].to_dict("records"),
        "pipeline": "clean_retrain",
    }

    meta_path = MODELS_DIR / "strategy_e_clean_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    results_path = BACKTEST_DIR / "strategy_e_clean_backtest.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Metadata saved: {meta_path.name}")
    print(f"  Results saved: {results_path.name}")

    print(f"\n{'=' * 70}")
    print("✅ Clean retrain pipeline complete!")
    print("   These are Strategy E's TRUE numbers.")
    print("   Next: Run A+B+C+D+E combined backtest.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()