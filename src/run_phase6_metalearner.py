"""
Project MIDAS v2 — Phase 6: Meta-Learner Training
Walk-forward co-training pipeline for meta-learner candidates.

Collects training data from strategy signals, trains 4 meta-learner
candidates (LightGBM, LogisticRegression, Ridge, WeightedEnsemble),
evaluates on walk-forward windows, picks the winner.

Usage:
    python run_phase6_metalearner.py

Requirements:
    pip install lightgbm scikit-learn --break-system-packages
"""

import sys
import json
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

import config
from config import (
    DATA_RAW, DATA_PROCESSED, BACKTEST_DIR, MODELS_DIR,
    ATR_PERIOD, CONFIDENCE_THRESHOLD
)
from feature_engineering import FeatureEngine
from backtester import Backtester, StrategyBase, TradeSignal

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    print("WARNING: lightgbm not installed. Run: pip install lightgbm")
    print("         Will skip LightGBM candidate.\n")
    HAS_LGBM = False


# ─── Walk-Forward Windows ────────────────────────────────

WALK_FORWARD_WINDOWS = [
    {"name": "W1", "train_start": "2021-01-01", "train_end": "2023-12-31",
     "test_start": "2024-01-01", "test_end": "2024-06-30"},
    {"name": "W2", "train_start": "2021-07-01", "train_end": "2024-06-30",
     "test_start": "2024-07-01", "test_end": "2024-12-31"},
    {"name": "W3", "train_start": "2022-01-01", "train_end": "2024-12-31",
     "test_start": "2025-01-01", "test_end": "2025-06-30"},
]

# Final holdout: 2025-07-01 to 2026-04-03 (never trained on)
HOLDOUT = {"start": "2025-07-01", "end": "2026-12-31"}


# ─── Data Collection Strategy ────────────────────────────

class DataCollectorStrategy(StrategyBase):
    """
    Wraps all four strategies and collects feature vectors for
    every signal generated. Outcomes are matched AFTER backtest completes.
    """

    name = "DataCollector"

    def __init__(self):
        from strategies.strategy_a import StrategyA
        from strategies.strategy_b import StrategyB
        from strategies.strategy_c import StrategyC
        from strategies.strategy_d import StrategyD

        self.strategy_a = StrategyA()
        self.strategy_b = StrategyB()
        self.strategy_c = StrategyC()
        self.strategy_d = StrategyD()

        # Store ALL signals keyed by entry datetime string
        self.signal_features: Dict[str, Dict] = {}

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[TradeSignal]:
        """Run all strategies, collect features, return winning signal."""
        row = df.iloc[idx]

        signals = []
        sig_a = self.strategy_a.generate_signal(df, idx)
        sig_b = self.strategy_b.generate_signal(df, idx)
        sig_c = self.strategy_c.generate_signal(df, idx)
        sig_d = self.strategy_d.generate_signal(df, idx)

        if sig_a: signals.append(("A", sig_a))
        if sig_b: signals.append(("B", sig_b))
        if sig_c: signals.append(("C", sig_c))
        if sig_d: signals.append(("D", sig_d))

        if not signals:
            return None

        # Check conflicts
        directions = set(s.direction for _, s in signals)
        if len(directions) > 1:
            return None

        # Pick best signal (same voting logic as combined runner)
        if len(signals) == 1:
            best_name, best_signal = signals[0]
        else:
            best_name, best_signal = max(signals, key=lambda x: x[1].confidence)
            boost = (len(signals) - 1) * 10
            best_signal.confidence = min(100, best_signal.confidence + boost)

        # Store feature vector keyed by datetime
        feature_vector = self._extract_features(row, signals, best_name, best_signal)
        dt_key = str(best_signal.datetime)
        self.signal_features[dt_key] = feature_vector

        return best_signal

    def match_outcomes(self, trades) -> pd.DataFrame:
        """Match collected signal features with trade outcomes by entry time."""
        matched = []
        for trade in trades:
            dt_key = str(trade.entry_time)
            if dt_key in self.signal_features:
                features = self.signal_features[dt_key].copy()
                features["outcome"] = 1 if trade.pnl_dollars > 0 else 0
                features["pnl"] = trade.pnl_dollars
                features["exit_reason"] = trade.exit_reason
                matched.append(features)

        if not matched:
            return pd.DataFrame()
        return pd.DataFrame(matched)

    def _extract_features(self, row, signals, best_name, best_signal) -> Dict:
        """Extract all features for meta-learner input at signal time."""
        signal_names = [n for n, _ in signals]

        features = {
            # Which strategy fired
            "source_strategy": best_name,
            "num_strategies_agreed": len(signals),
            "strategy_a_fired": int("A" in signal_names),
            "strategy_b_fired": int("B" in signal_names),
            "strategy_c_fired": int("C" in signal_names),
            "strategy_d_fired": int("D" in signal_names),

            # Signal properties
            "confidence": best_signal.confidence,
            "direction": 1 if best_signal.direction == "buy" else -1,

            # Risk metrics
            "sl_distance": abs(best_signal.entry_price - best_signal.sl_price),
            "tp_distance": abs(best_signal.tp_price - best_signal.entry_price),
            "rr_ratio": (abs(best_signal.tp_price - best_signal.entry_price) /
                         max(abs(best_signal.entry_price - best_signal.sl_price), 0.001)),

            # Market context (M5)
            "atr": row.get(f"atr_{ATR_PERIOD}", np.nan),
            "rsi": row.get(f"rsi_{ATR_PERIOD}", np.nan),
            "adx": row.get(f"adx_{ATR_PERIOD}", np.nan),
            "ema_stack": row.get("ema_stack", 0),
            "above_ema200": row.get("above_ema200", 0),
            "body_ratio": row.get("body_ratio", 0),
            "is_bullish": row.get("is_bullish", 0),
            "candle_range": row.get("candle_range", 0),
            "dist_ema_21_atr": row.get("dist_ema_21_atr", 0),
            "dist_ema_50_atr": row.get("dist_ema_50_atr", 0),
            "atr_percentile": row.get("atr_percentile", 0.5),

            # M15 context
            "m15_ema_stack": row.get("m15_ema_stack", 0),
            "m15_above_ema200": row.get("m15_above_ema200", 0),
            "m15_trend_strong": row.get("m15_trend_strong", 0),
            "m15_ema_50_slope": row.get("m15_ema_50_slope", 0),

            # Volatility regime
            "vol_low": row.get("vol_low_volatility", 0),
            "vol_normal": row.get("vol_normal", 0),
            "vol_high": row.get("vol_high_volatility", 0),

            # Session
            "session_asian": row.get("session_asian", 0),
            "session_london": row.get("session_london", 0),
            "session_new_york": row.get("session_new_york", 0),

            # Time
            "hour": row.get("hour", 0),
            "day_of_week": row.get("day_of_week", 0),
            "hour_sin": row.get("hour_sin", 0),
            "hour_cos": row.get("hour_cos", 0),

            # Price context
            "close": row.get("close", 0),
            "dist_to_swing_high": row.get("dist_to_swing_high", 0),
            "dist_to_swing_low": row.get("dist_to_swing_low", 0),
        }

        return features


# CollectingBacktester no longer needed — outcomes matched via strategy.match_outcomes()


# ─── Meta-Learner Candidates ─────────────────────────────

FEATURE_COLS = [
    "num_strategies_agreed", "strategy_a_fired", "strategy_b_fired",
    "strategy_c_fired", "strategy_d_fired", "confidence", "direction",
    "sl_distance", "tp_distance", "rr_ratio",
    "atr", "rsi", "adx", "ema_stack", "above_ema200",
    "body_ratio", "is_bullish", "candle_range",
    "dist_ema_21_atr", "dist_ema_50_atr", "atr_percentile",
    "m15_ema_stack", "m15_above_ema200", "m15_trend_strong", "m15_ema_50_slope",
    "vol_low", "vol_normal", "vol_high",
    "session_asian", "session_london", "session_new_york",
    "hour", "day_of_week", "hour_sin", "hour_cos",
    "dist_to_swing_high", "dist_to_swing_low",
]


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    train_probs = model.predict_proba(X_train_s)[:, 1]
    test_probs = model.predict_proba(X_test_s)[:, 1]

    return model, scaler, train_probs, test_probs


def train_ridge(X_train, y_train, X_test, y_test):
    """Train and evaluate Ridge Classifier."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RidgeClassifier(alpha=1.0, random_state=42)
    model.fit(X_train_s, y_train)

    # Ridge doesn't have predict_proba, use decision function
    train_scores = model.decision_function(X_train_s)
    test_scores = model.decision_function(X_test_s)

    # Convert to probabilities via sigmoid
    train_probs = 1 / (1 + np.exp(-train_scores))
    test_probs = 1 / (1 + np.exp(-test_scores))

    return model, scaler, train_probs, test_probs


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train and evaluate LightGBM with heavy regularization."""
    if not HAS_LGBM:
        return None, None, None, None

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 7,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    return model, None, train_probs, test_probs


def train_weighted_ensemble(X_train, y_train, X_test, y_test, train_df, test_df):
    """Weighted average of strategy confidences — no actual ML model."""
    # Simple: weighted sum of strategy fire indicators × confidence
    # Optimized on training set

    from scipy.optimize import minimize

    def neg_pf(weights):
        """Negative profit factor as loss function."""
        w_a, w_b, w_c, w_d = weights
        scores = (
            train_df["strategy_a_fired"] * train_df["confidence"] * w_a +
            train_df["strategy_b_fired"] * train_df["confidence"] * w_b +
            train_df["strategy_c_fired"] * train_df["confidence"] * w_c +
            train_df["strategy_d_fired"] * train_df["confidence"] * w_d
        )
        # Predict: take trade if score > median
        threshold = scores.median()
        take = scores > threshold
        wins = (y_train[take] == 1).sum()
        losses = (y_train[take] == 0).sum()
        pf = wins / max(losses, 1)
        return -pf  # Minimize negative PF

    result = minimize(neg_pf, x0=[1, 1, 1, 1], method="Nelder-Mead",
                      options={"maxiter": 1000})
    weights = result.x
    weights = weights / weights.sum()  # Normalize

    # Generate probabilities
    def compute_scores(df):
        return (
            df["strategy_a_fired"] * df["confidence"] * weights[0] +
            df["strategy_b_fired"] * df["confidence"] * weights[1] +
            df["strategy_c_fired"] * df["confidence"] * weights[2] +
            df["strategy_d_fired"] * df["confidence"] * weights[3]
        )

    train_scores = compute_scores(train_df)
    test_scores = compute_scores(test_df)

    # Normalize to 0-1
    min_s, max_s = train_scores.min(), train_scores.max()
    if max_s > min_s:
        train_probs = (train_scores - min_s) / (max_s - min_s)
        test_probs = (test_scores - min_s) / (max_s - min_s)
    else:
        train_probs = np.full(len(train_scores), 0.5)
        test_probs = np.full(len(test_scores), 0.5)

    return weights, None, train_probs.values, test_probs.values


# ─── Evaluation ──────────────────────────────────────────

def evaluate_metalearner(test_probs, test_outcomes, test_pnls, threshold=0.5):
    """Evaluate meta-learner predictions against trading outcomes."""
    if len(test_probs) == 0:
        return {"pf": 0, "wr": 0, "trades": 0, "pnl": 0, "filtered": 0}

    take_mask = test_probs >= threshold
    skip_mask = ~take_mask

    trades_taken = take_mask.sum()
    trades_skipped = skip_mask.sum()

    if trades_taken == 0:
        return {"pf": 0, "wr": 0, "trades": 0, "pnl": 0, "filtered": int(trades_skipped)}

    taken_outcomes = test_outcomes[take_mask]
    taken_pnls = test_pnls[take_mask]

    wins = (taken_outcomes == 1).sum()
    losses = (taken_outcomes == 0).sum()
    wr = wins / trades_taken * 100

    gross_win = taken_pnls[taken_pnls > 0].sum()
    gross_loss = abs(taken_pnls[taken_pnls <= 0].sum())
    pf = gross_win / max(gross_loss, 0.01)

    total_pnl = taken_pnls.sum()

    # Check quality of filtered trades (trades we skipped)
    if trades_skipped > 0:
        skipped_pnls = test_pnls[skip_mask]
        skipped_avg = skipped_pnls.mean()
    else:
        skipped_avg = 0

    return {
        "pf": round(pf, 3),
        "wr": round(wr, 1),
        "trades": int(trades_taken),
        "pnl": round(total_pnl, 2),
        "filtered": int(trades_skipped),
        "skipped_avg_pnl": round(skipped_avg, 3),
    }


# ─── Data Loading ────────────────────────────────────────

def load_full_data():
    """Load and prepare all data (same as combined runner)."""
    engine = FeatureEngine()

    m15_processed = DATA_PROCESSED / "XAUUSD_M15_features.csv"
    if m15_processed.exists():
        df_m15 = pd.read_csv(m15_processed, parse_dates=["datetime"])
    else:
        df_m15 = pd.read_csv(DATA_RAW / "XAUUSD_M15.csv", parse_dates=["datetime"])
        df_m15 = engine.compute_shared(df_m15, timeframe="M15")
        df_m15.to_csv(m15_processed, index=False)

    print("  Computing FVG + SMC on M15...")
    df_m15 = engine.compute_fvg(df_m15)
    df_m15 = engine.compute_smc(df_m15)

    m5_processed = DATA_PROCESSED / "XAUUSD_M5_features.csv"
    if m5_processed.exists():
        df_m5 = pd.read_csv(m5_processed, parse_dates=["datetime"])
    else:
        df_m5 = pd.read_csv(DATA_RAW / "XAUUSD_M5.csv", parse_dates=["datetime"])
        df_m5 = engine.compute_shared(df_m5, timeframe="M5")
        df_m5.to_csv(m5_processed, index=False)

    print("  Computing Trend + Session Range on M5...")
    df_m5 = engine.compute_trend(df_m5)
    df_m5 = engine.compute_session_range(df_m5)

    print("  Merging M15 into M5...")
    df_merged = merge_all(df_m5, df_m15)
    print(f"  Ready: {len(df_merged):,} candles, {len(df_merged.columns)} columns\n")
    return df_merged


def merge_all(df_m5, df_m15):
    """Full merge with all features."""
    df_m5 = df_m5.sort_values("datetime").reset_index(drop=True)
    df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

    m15_cols = ["datetime"]
    m15_cols.extend([c for c in df_m15.columns if c.startswith("fvg_")])
    m15_cols.extend([c for c in df_m15.columns if c.startswith(("bos_", "ob_", "equal_", "last_equal_"))])
    m15_cols.extend(["open", "high", "low", "close"])

    for col in ["ema_9", "ema_21", "ema_50", "ema_200", "ema_50_slope", "ema_stack",
                 "ema_partial_bull", "ema_partial_bear", "above_ema200",
                 f"atr_{ATR_PERIOD}", f"rsi_{ATR_PERIOD}", f"adx_{ATR_PERIOD}",
                 "trend_strong", "trend_very_strong", "trend_weak",
                 "candle_range", "body", "is_bullish",
                 "last_swing_high", "last_swing_low", "vol_regime"]:
        if col in df_m15.columns:
            m15_cols.append(col)

    m15_cols = list(dict.fromkeys(m15_cols))
    df_sub = df_m15[m15_cols].copy()
    idx_vals = df_sub.index.values
    rename = {c: f"m15_{c}" for c in df_sub.columns if c != "datetime"}
    df_sub = df_sub.rename(columns=rename).rename(columns={"datetime": "m15_datetime_key"})
    df_sub["m15_idx"] = idx_vals

    merged = pd.merge_asof(df_m5, df_sub, left_on="datetime",
                           right_on="m15_datetime_key", direction="backward")
    merged = merged.rename(columns={"m15_datetime_key": "m15_datetime"})
    return merged


# ─── Main Pipeline ───────────────────────────────────────

def run_phase6():
    """Run complete Phase 6 meta-learner training pipeline."""

    print("=" * 70)
    print("  PHASE 6: META-LEARNER TRAINING")
    print("=" * 70)

    # Lower threshold so we capture all possible signals
    original_thresh = config.CONFIDENCE_THRESHOLD
    config.CONFIDENCE_THRESHOLD = 55

    print("\n  Loading data...")
    df = load_full_data()

    # ── Collect training data per walk-forward window ──
    all_window_results = []

    for wi, window in enumerate(WALK_FORWARD_WINDOWS):
        print(f"\n{'='*70}")
        print(f"  WINDOW {window['name']}: Train {window['train_start']} to {window['train_end']}")
        print(f"             Test  {window['test_start']} to {window['test_end']}")
        print(f"{'='*70}")

        # Filter data for this window
        df_train = df[(df["datetime"] >= window["train_start"]) &
                      (df["datetime"] <= window["train_end"])].copy()
        df_test = df[(df["datetime"] >= window["test_start"]) &
                     (df["datetime"] <= window["test_end"])].copy()

        print(f"  Train candles: {len(df_train):,} | Test candles: {len(df_test):,}")

        # ── Collect training signals ──
        print("  Collecting training signals...")
        collector_train = DataCollectorStrategy()
        bt_train = Backtester(verbose=False)
        res_train = bt_train.run(df_train, collector_train, mode="full")
        train_data = collector_train.match_outcomes(res_train["trades"])

        print("  Collecting test signals...")
        collector_test = DataCollectorStrategy()
        bt_test = Backtester(verbose=False)
        res_test = bt_test.run(df_test, collector_test, mode="full")
        test_data = collector_test.match_outcomes(res_test["trades"])

        print(f"  Training samples: {len(train_data)} | Test samples: {len(test_data)}")

        if len(train_data) < 50 or len(test_data) < 20:
            print(f"  ⚠️  Insufficient data for window {window['name']}, skipping")
            continue

        # Prepare features
        available_features = [f for f in FEATURE_COLS if f in train_data.columns]
        X_train = train_data[available_features].fillna(0).values
        y_train = train_data["outcome"].values
        pnl_train = train_data["pnl"].values

        X_test = test_data[available_features].fillna(0).values
        y_test = test_data["outcome"].values
        pnl_test = test_data["pnl"].values

        print(f"  Features used: {len(available_features)}")
        print(f"  Train WR: {y_train.mean()*100:.1f}% | Test WR: {y_test.mean()*100:.1f}%")

        # ── Baseline: take ALL trades (no filtering) ──
        baseline = evaluate_metalearner(
            np.ones(len(y_test)), y_test, pnl_test, threshold=0.5
        )
        print(f"\n  Baseline (no filter): PnL=${baseline['pnl']:+.2f} | "
              f"WR={baseline['wr']:.1f}% | PF={baseline['pf']:.2f} | "
              f"Trades={baseline['trades']}")

        # ── Train candidates ──
        window_results = {"window": window["name"], "baseline": baseline, "candidates": {}}

        # 1. Logistic Regression
        print("\n  Training LogisticRegression...")
        try:
            model_lr, scaler_lr, _, test_probs_lr = train_logistic_regression(
                X_train, y_train, X_test, y_test)
            for thresh in [0.45, 0.50, 0.55]:
                res = evaluate_metalearner(test_probs_lr, y_test, pnl_test, threshold=thresh)
                key = f"LogReg_t{int(thresh*100)}"
                window_results["candidates"][key] = res
                print(f"    {key}: PnL=${res['pnl']:+.2f} | WR={res['wr']:.1f}% | "
                      f"PF={res['pf']:.2f} | Trades={res['trades']} | "
                      f"Filtered={res['filtered']} | SkippedAvg=${res['skipped_avg_pnl']:+.3f}")
        except Exception as e:
            print(f"    LogReg failed: {e}")

        # 2. Ridge
        print("\n  Training Ridge...")
        try:
            model_ridge, scaler_ridge, _, test_probs_ridge = train_ridge(
                X_train, y_train, X_test, y_test)
            for thresh in [0.45, 0.50, 0.55]:
                res = evaluate_metalearner(test_probs_ridge, y_test, pnl_test, threshold=thresh)
                key = f"Ridge_t{int(thresh*100)}"
                window_results["candidates"][key] = res
                print(f"    {key}: PnL=${res['pnl']:+.2f} | WR={res['wr']:.1f}% | "
                      f"PF={res['pf']:.2f} | Trades={res['trades']} | "
                      f"Filtered={res['filtered']} | SkippedAvg=${res['skipped_avg_pnl']:+.3f}")
        except Exception as e:
            print(f"    Ridge failed: {e}")

        # 3. LightGBM
        if HAS_LGBM:
            print("\n  Training LightGBM...")
            try:
                model_lgb, _, _, test_probs_lgb = train_lightgbm(
                    X_train, y_train, X_test, y_test)
                for thresh in [0.45, 0.50, 0.55]:
                    res = evaluate_metalearner(test_probs_lgb, y_test, pnl_test, threshold=thresh)
                    key = f"LightGBM_t{int(thresh*100)}"
                    window_results["candidates"][key] = res
                    print(f"    {key}: PnL=${res['pnl']:+.2f} | WR={res['wr']:.1f}% | "
                          f"PF={res['pf']:.2f} | Trades={res['trades']} | "
                          f"Filtered={res['filtered']} | SkippedAvg=${res['skipped_avg_pnl']:+.3f}")

                # Feature importance
                if model_lgb is not None:
                    importances = model_lgb.feature_importances_
                    top_features = sorted(zip(available_features, importances),
                                          key=lambda x: x[1], reverse=True)[:10]
                    print(f"\n    Top 10 features:")
                    for feat, imp in top_features:
                        print(f"      {feat:<30} {imp:>6}")
            except Exception as e:
                print(f"    LightGBM failed: {e}")

        # 4. Weighted Ensemble
        print("\n  Training WeightedEnsemble...")
        try:
            weights, _, _, test_probs_we = train_weighted_ensemble(
                X_train, y_train, X_test, y_test,
                train_data[available_features + ["confidence",
                    "strategy_a_fired", "strategy_b_fired",
                    "strategy_c_fired", "strategy_d_fired"]].fillna(0),
                test_data[available_features + ["confidence",
                    "strategy_a_fired", "strategy_b_fired",
                    "strategy_c_fired", "strategy_d_fired"]].fillna(0),
            )
            for thresh in [0.45, 0.50, 0.55]:
                res = evaluate_metalearner(test_probs_we, y_test, pnl_test, threshold=thresh)
                key = f"Weighted_t{int(thresh*100)}"
                window_results["candidates"][key] = res
                print(f"    {key}: PnL=${res['pnl']:+.2f} | WR={res['wr']:.1f}% | "
                      f"PF={res['pf']:.2f} | Trades={res['trades']} | "
                      f"Filtered={res['filtered']} | SkippedAvg=${res['skipped_avg_pnl']:+.3f}")
            if isinstance(weights, np.ndarray):
                print(f"    Weights: A={weights[0]:.3f} B={weights[1]:.3f} "
                      f"C={weights[2]:.3f} D={weights[3]:.3f}")
        except Exception as e:
            print(f"    WeightedEnsemble failed: {e}")

        all_window_results.append(window_results)

    # ── Cross-Window Comparison ──
    print(f"\n{'='*70}")
    print(f"  PHASE 6: CROSS-WINDOW COMPARISON")
    print(f"{'='*70}")

    # Aggregate results across windows
    candidate_names = set()
    for wr in all_window_results:
        candidate_names.update(wr["candidates"].keys())

    print(f"\n  {'Candidate':<25}", end="")
    for wr in all_window_results:
        print(f"  {wr['window']+' PnL':>10} {wr['window']+' PF':>8}", end="")
    print(f"  {'Avg PnL':>10} {'Avg PF':>8} {'Consistent':>10}")
    print("  " + "-" * (25 + len(all_window_results) * 20 + 30))

    # Baseline row
    print(f"  {'Baseline (no filter)':<25}", end="")
    bl_pnls, bl_pfs = [], []
    for wr in all_window_results:
        bl = wr["baseline"]
        print(f"  ${bl['pnl']:>+8.0f} {bl['pf']:>7.2f}", end="")
        bl_pnls.append(bl["pnl"])
        bl_pfs.append(bl["pf"])
    avg_bl_pnl = np.mean(bl_pnls)
    avg_bl_pf = np.mean(bl_pfs)
    all_positive = all(p > 0 for p in bl_pnls)
    print(f"  ${avg_bl_pnl:>+8.0f} {avg_bl_pf:>7.2f} {'✅ all' if all_positive else '❌ no':>10}")

    # Candidate rows
    candidate_summary = []
    for cname in sorted(candidate_names):
        pnls, pfs = [], []
        row_str = f"  {cname:<25}"
        for wr in all_window_results:
            if cname in wr["candidates"]:
                c = wr["candidates"][cname]
                row_str += f"  ${c['pnl']:>+8.0f} {c['pf']:>7.2f}"
                pnls.append(c["pnl"])
                pfs.append(c["pf"])
            else:
                row_str += f"  {'N/A':>10} {'N/A':>8}"

        if pnls:
            avg_pnl = np.mean(pnls)
            avg_pf = np.mean(pfs)
            all_pos = all(p > 0 for p in pnls)
            row_str += f"  ${avg_pnl:>+8.0f} {avg_pf:>7.2f} {'✅ all' if all_pos else '❌ no':>10}"
            candidate_summary.append({
                "name": cname, "avg_pnl": avg_pnl, "avg_pf": avg_pf,
                "consistent": all_pos, "pnls": pnls, "pfs": pfs
            })
        print(row_str)

    # Pick winner
    # Priority: consistent across windows > highest avg PF
    consistent = [c for c in candidate_summary if c["consistent"]]
    if consistent:
        winner = max(consistent, key=lambda c: c["avg_pf"])
    elif candidate_summary:
        winner = max(candidate_summary, key=lambda c: c["avg_pnl"])
    else:
        winner = None

    print(f"\n  {'='*70}")
    if winner:
        print(f"  🏆 META-LEARNER WINNER: {winner['name']}")
        print(f"     Avg PnL: ${winner['avg_pnl']:+,.2f}")
        print(f"     Avg PF: {winner['avg_pf']:.3f}")
        print(f"     Consistent: {'Yes ✅' if winner['consistent'] else 'No ❌'}")
        print(f"     Window PnLs: {['$'+f'{p:+.0f}' for p in winner['pnls']]}")

        # Compare to baseline
        improvement_pnl = winner["avg_pnl"] - avg_bl_pnl
        improvement_pf = winner["avg_pf"] - avg_bl_pf
        print(f"\n     vs Baseline:")
        print(f"       PnL: ${improvement_pnl:+,.0f} ({'better' if improvement_pnl > 0 else 'worse'})")
        print(f"       PF:  {improvement_pf:+.3f} ({'better' if improvement_pf > 0 else 'worse'})")

        if improvement_pf > 0:
            print(f"\n     ✅ META-LEARNER ADDS VALUE — proceed to Phase 7 with this model")
        else:
            print(f"\n     ⚠️  META-LEARNER DOESN'T IMPROVE PF — consider using baseline only")
    else:
        print(f"  ❌ NO WINNER — all candidates failed. Proceed with rule-based baseline.")

    # Restore config
    config.CONFIDENCE_THRESHOLD = original_thresh

    # Save results
    save_path = BACKTEST_DIR / "phase6_metalearner_results.json"
    save_data = {
        "windows": [
            {
                "name": wr["window"],
                "baseline": wr["baseline"],
                "candidates": wr["candidates"]
            }
            for wr in all_window_results
        ],
        "winner": winner["name"] if winner else "none",
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {save_path}")

    return all_window_results, winner


if __name__ == "__main__":
    results, winner = run_phase6()