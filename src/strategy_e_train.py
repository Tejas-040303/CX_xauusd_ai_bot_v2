"""
Project MIDAS v2 — Strategy E: LightGBM Training
==================================================
Walk-forward training of LightGBM on 167+ features.
Trains one model per walk-forward window, evaluates on held-out test period,
and saves the best model for live inference.

Usage:
    python strategy_e_train.py

Output:
    models/strategy_e_W1.txt, W2.txt, W3.txt  (LightGBM models)
    models/strategy_e_final.txt                 (retrained on all data up to holdout)
    models/strategy_e_metadata.json             (thresholds, feature importance, stats)
    backtest_results/strategy_e_evaluation.png  (performance charts)
"""

import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore", category=UserWarning)

from config import DATA_PROCESSED, MODELS_DIR, BACKTEST_DIR, ATR_PERIOD

# ─── Import from data prep ───────────────────────────────
from strategy_e_data_prep import (
    WALK_FORWARD_WINDOWS, HOLDOUT_START, HOLDOUT_END, LABEL_COL
)


# ─── LightGBM Hyperparameters ────────────────────────────
# Conservative starting point — designed for noisy financial data.
# Key choices:
#   - Low learning_rate + high n_estimators = stable convergence
#   - max_depth=6 = prevents overfitting to noise
#   - feature_fraction/bagging = regularization via subsampling
#   - min_child_samples=100 = needs strong patterns (not fitting to 5 samples)
#   - scale_pos_weight handled per-window for class imbalance
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "n_estimators": 2000,       # With early stopping, won't use all
    "max_depth": 6,
    "num_leaves": 31,           # 2^5 - 1 (conservative for depth 6)
    "min_child_samples": 100,   # Needs 100+ samples per leaf (anti-overfit)
    "feature_fraction": 0.7,    # Use 70% of features per tree
    "bagging_fraction": 0.8,    # Use 80% of samples per tree
    "bagging_freq": 5,          # Bagging every 5 iterations
    "lambda_l1": 0.1,           # L1 regularization
    "lambda_l2": 1.0,           # L2 regularization
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

EARLY_STOPPING_ROUNDS = 100     # Stop if no improvement for 100 rounds
CONFIDENCE_THRESHOLD_DEFAULT = 0.55  # Minimum predicted prob to emit signal


def load_data():
    """Load the train-ready dataset and feature list."""
    data_path = DATA_PROCESSED / "strategy_e_train_ready.parquet"
    feature_path = DATA_PROCESSED / "strategy_e_feature_list.json"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Train-ready data not found at {data_path}. "
            "Run strategy_e_data_prep.py first."
        )

    df = pd.read_parquet(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    with open(feature_path) as f:
        feature_cols = json.load(f)

    # Verify features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing from data: {missing[:5]}...")
        feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"  Loaded: {len(df):,} rows, {len(feature_cols)} features")
    return df, feature_cols


def split_window(df, feature_cols, window_dates):
    """Split data into train/test for a walk-forward window."""
    train_mask = (
        (df["datetime"] >= window_dates["train_start"]) &
        (df["datetime"] <= window_dates["train_end"]) &
        (df[LABEL_COL] != -1)  # Exclude WAIT bars from training
    )
    test_mask = (
        (df["datetime"] >= window_dates["test_start"]) &
        (df["datetime"] <= window_dates["test_end"]) &
        (df[LABEL_COL] != -1)  # Evaluate only on non-WAIT bars
    )

    train = df[train_mask]
    test = df[test_mask]

    X_train = train[feature_cols].values
    y_train = train[LABEL_COL].values.astype(int)
    X_test = test[feature_cols].values
    y_test = test[LABEL_COL].values.astype(int)

    # Also return full test set (including WAITs) for signal generation analysis
    test_all_mask = (
        (df["datetime"] >= window_dates["test_start"]) &
        (df["datetime"] <= window_dates["test_end"])
    )
    test_all = df[test_all_mask]

    return X_train, y_train, X_test, y_test, train, test, test_all


def train_single_window(X_train, y_train, X_test, y_test, feature_cols, window_name):
    """
    Train LightGBM on one walk-forward window.
    Returns the model and evaluation metrics.
    """
    # Handle class imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / max(n_pos, 1)

    params = LGBM_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos

    print(f"    Train: {len(y_train):,} samples (BUY={n_pos:,}, SELL={n_neg:,}, "
          f"ratio={scale_pos:.3f})")
    print(f"    Test:  {len(y_test):,} samples")

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)

    # Use 10% of training data as validation for early stopping
    val_size = int(len(X_train) * 0.1)
    # Take last 10% as validation (preserves temporal ordering)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_sub = X_train[:-val_size]
    y_train_sub = y_train[:-val_size]

    train_sub = lgb.Dataset(X_train_sub, label=y_train_sub, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_sub, feature_name=feature_cols)

    # Train with early stopping
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=200),
    ]

    model = lgb.train(
        params,
        train_sub,
        valid_sets=[train_sub, val_data],
        valid_names=["train", "valid"],
        num_boost_round=params.pop("n_estimators", 2000),
        callbacks=callbacks,
    )

    print(f"    Best iteration: {model.best_iteration}")

    # ── Predictions on test set ───────────────────────────
    y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # ── Metrics ───────────────────────────────────────────
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_buy": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "precision_sell": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        "recall_buy": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_sell": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        "f1_buy": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_sell": f1_score(y_test, y_pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "log_loss": log_loss(y_test, y_pred_prob),
        "best_iteration": model.best_iteration,
    }

    # ── Score separation (key metric from TFT analysis) ───
    buy_scores = y_pred_prob[y_test == 1]
    sell_scores = y_pred_prob[y_test == 0]
    metrics["mean_score_buy"] = float(np.mean(buy_scores))
    metrics["mean_score_sell"] = float(np.mean(sell_scores))
    metrics["score_separation"] = metrics["mean_score_buy"] - metrics["mean_score_sell"]

    # ── Confidence-binned accuracy ────────────────────────
    # How accurate is the model when it's MORE confident?
    # This is crucial — if high confidence = high accuracy, we have a usable signal.
    confidence_bins = {}
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        # Predictions above threshold → BUY, below (1-threshold) → SELL
        high_conf_buy = y_pred_prob >= threshold
        high_conf_sell = y_pred_prob <= (1 - threshold)
        high_conf_mask = high_conf_buy | high_conf_sell

        if high_conf_mask.sum() > 0:
            high_conf_preds = np.where(high_conf_buy, 1, 0)[high_conf_mask]
            high_conf_actual = y_test[high_conf_mask]
            conf_acc = accuracy_score(high_conf_actual, high_conf_preds)
            conf_count = high_conf_mask.sum()
        else:
            conf_acc = 0.0
            conf_count = 0

        confidence_bins[f"thresh_{threshold:.2f}"] = {
            "accuracy": round(conf_acc, 4),
            "sample_count": int(conf_count),
            "pct_of_test": round(100 * conf_count / len(y_test), 1),
        }

    metrics["confidence_bins"] = confidence_bins

    # ── Print results ─────────────────────────────────────
    print(f"\n    {window_name} Results:")
    print(f"      Accuracy:     {metrics['accuracy']:.4f}")
    print(f"      ROC AUC:      {metrics['roc_auc']:.4f}")
    print(f"      Score Sep:    {metrics['score_separation']:.4f} "
          f"(BUY avg={metrics['mean_score_buy']:.4f}, "
          f"SELL avg={metrics['mean_score_sell']:.4f})")
    print(f"      F1 (BUY):     {metrics['f1_buy']:.4f}")
    print(f"      F1 (SELL):    {metrics['f1_sell']:.4f}")
    print(f"\n      Confidence-binned accuracy:")
    for thresh_name, stats in confidence_bins.items():
        t = thresh_name.replace("thresh_", "")
        print(f"        ≥{t}: {stats['accuracy']:.4f} "
              f"({stats['sample_count']:,} samples, {stats['pct_of_test']}% of test)")

    return model, metrics, y_pred_prob


def get_feature_importance(model, feature_cols, top_n=30):
    """Extract and rank feature importance."""
    importance_gain = model.feature_importance(importance_type="gain")
    importance_split = model.feature_importance(importance_type="split")

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": importance_gain,
        "split": importance_split,
    }).sort_values("gain", ascending=False)

    # Normalize to percentages
    importance_df["gain_pct"] = 100 * importance_df["gain"] / importance_df["gain"].sum()
    importance_df["split_pct"] = 100 * importance_df["split"] / importance_df["split"].sum()

    return importance_df.head(top_n)


def find_optimal_threshold(y_test, y_pred_prob):
    """
    Find the confidence threshold that maximizes profit-adjusted accuracy.
    We want: high threshold → fewer but better trades.

    Returns the threshold and a trade-off table.
    """
    results = []

    for thresh in np.arange(0.50, 0.76, 0.01):
        buy_mask = y_pred_prob >= thresh
        sell_mask = y_pred_prob <= (1 - thresh)
        signal_mask = buy_mask | sell_mask

        if signal_mask.sum() < 10:
            continue

        preds = np.where(buy_mask, 1, 0)[signal_mask]
        actuals = y_test[signal_mask]

        acc = accuracy_score(actuals, preds)
        n_signals = signal_mask.sum()
        coverage = n_signals / len(y_test)

        # Edge = how much better than 50% (random)
        edge = acc - 0.5

        # Profit proxy = edge * coverage (more trades with edge = more profit)
        profit_proxy = edge * coverage

        results.append({
            "threshold": round(thresh, 2),
            "accuracy": round(acc, 4),
            "n_signals": int(n_signals),
            "coverage": round(coverage, 4),
            "edge": round(edge, 4),
            "profit_proxy": round(profit_proxy, 6),
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return 0.55, results_df

    # Best threshold = max profit proxy
    best_idx = results_df["profit_proxy"].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]

    return best_threshold, results_df


def evaluate_on_holdout(model, df, feature_cols, threshold):
    """Final evaluation on the holdout period (never trained on)."""
    holdout_mask = (
        (df["datetime"] >= HOLDOUT_START) &
        (df["datetime"] <= HOLDOUT_END) &
        (df[LABEL_COL] != -1)
    )
    holdout = df[holdout_mask]

    if len(holdout) == 0:
        print("  No holdout data available (may not have data past 2025-07-01)")
        return None

    X_holdout = holdout[feature_cols].values
    y_holdout = holdout[LABEL_COL].values.astype(int)

    y_pred_prob = model.predict(X_holdout)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # With threshold
    buy_mask = y_pred_prob >= threshold
    sell_mask = y_pred_prob <= (1 - threshold)
    signal_mask = buy_mask | sell_mask

    metrics = {
        "accuracy_all": accuracy_score(y_holdout, y_pred),
        "roc_auc": roc_auc_score(y_holdout, y_pred_prob),
    }

    if signal_mask.sum() > 0:
        preds_filtered = np.where(buy_mask, 1, 0)[signal_mask]
        actuals_filtered = y_holdout[signal_mask]
        metrics["accuracy_filtered"] = accuracy_score(actuals_filtered, preds_filtered)
        metrics["n_signals"] = int(signal_mask.sum())
        metrics["coverage"] = round(signal_mask.sum() / len(y_holdout), 4)

    print(f"\n  HOLDOUT Evaluation ({HOLDOUT_START} → {HOLDOUT_END}):")
    print(f"    Samples: {len(y_holdout):,}")
    print(f"    Accuracy (all):      {metrics['accuracy_all']:.4f}")
    print(f"    ROC AUC:             {metrics['roc_auc']:.4f}")
    if "accuracy_filtered" in metrics:
        print(f"    Accuracy (≥{threshold}): {metrics['accuracy_filtered']:.4f} "
              f"({metrics['n_signals']} signals, {100*metrics['coverage']:.1f}% coverage)")

    return metrics


def save_evaluation_plots(all_results, feature_importances):
    """Save evaluation charts to backtest_results/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Strategy E — LightGBM Evaluation", fontsize=16, fontweight="bold")

    # ── Plot 1: Accuracy across windows ───────────────────
    ax = axes[0, 0]
    windows = list(all_results.keys())
    accs = [all_results[w]["metrics"]["accuracy"] for w in windows]
    aucs = [all_results[w]["metrics"]["roc_auc"] for w in windows]
    seps = [all_results[w]["metrics"]["score_separation"] for w in windows]

    x = np.arange(len(windows))
    ax.bar(x - 0.2, accs, 0.3, label="Accuracy", color="#4CAF50", alpha=0.8)
    ax.bar(x + 0.1, aucs, 0.3, label="ROC AUC", color="#2196F3", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Score")
    ax.set_title("Performance Across Walk-Forward Windows")
    ax.legend()
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_ylim(0.45, 0.70)

    # ── Plot 2: Score separation ──────────────────────────
    ax = axes[0, 1]
    buy_avgs = [all_results[w]["metrics"]["mean_score_buy"] for w in windows]
    sell_avgs = [all_results[w]["metrics"]["mean_score_sell"] for w in windows]
    ax.bar(x - 0.15, buy_avgs, 0.3, label="BUY avg score", color="#4CAF50")
    ax.bar(x + 0.15, sell_avgs, 0.3, label="SELL avg score", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Predicted Probability")
    ax.set_title(f"Score Separation (BUY vs SELL)")
    ax.legend()
    for i, sep in enumerate(seps):
        ax.annotate(f"Δ={sep:.4f}", (i, max(buy_avgs[i], sell_avgs[i]) + 0.005),
                    ha="center", fontsize=9, fontweight="bold")

    # ── Plot 3: Confidence-binned accuracy (last window) ──
    ax = axes[1, 0]
    last_window = windows[-1]
    conf_bins = all_results[last_window]["metrics"]["confidence_bins"]
    thresholds = [float(k.replace("thresh_", "")) for k in conf_bins.keys()]
    conf_accs = [v["accuracy"] for v in conf_bins.values()]
    conf_counts = [v["sample_count"] for v in conf_bins.values()]

    ax2 = ax.twinx()
    bars = ax.bar(thresholds, conf_accs, width=0.04, color="#FF9800", alpha=0.8,
                  label="Accuracy")
    line = ax2.plot(thresholds, conf_counts, "b-o", label="Sample count")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Accuracy", color="#FF9800")
    ax2.set_ylabel("Sample Count", color="blue")
    ax.set_title(f"Confidence vs Accuracy ({last_window})")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

    # ── Plot 4: Top 15 feature importance ─────────────────
    ax = axes[1, 1]
    if feature_importances is not None and len(feature_importances) > 0:
        top15 = feature_importances.head(15).sort_values("gain_pct")
        ax.barh(top15["feature"], top15["gain_pct"], color="#9C27B0", alpha=0.8)
        ax.set_xlabel("Importance (% of total gain)")
        ax.set_title("Top 15 Features by Gain")
    else:
        ax.text(0.5, 0.5, "No feature importance data", ha="center", va="center")

    plt.tight_layout()
    output_path = BACKTEST_DIR / "strategy_e_evaluation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Evaluation plots saved to {output_path.name}")
    plt.close()


def main():
    print("=" * 70)
    print("MIDAS v2 — Strategy E: LightGBM Training")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────
    print("\n[1/5] Loading train-ready data...")
    df, feature_cols = load_data()

    # ── Walk-forward training ─────────────────────────────
    print("\n[2/5] Walk-forward training...")
    all_results = {}
    all_models = {}

    for window_name, window_dates in WALK_FORWARD_WINDOWS.items():
        print(f"\n{'─' * 60}")
        print(f"  Training {window_name}: "
              f"{window_dates['train_start']} → {window_dates['train_end']} | "
              f"Test: {window_dates['test_start']} → {window_dates['test_end']}")
        print(f"{'─' * 60}")

        X_train, y_train, X_test, y_test, train_df, test_df, test_all_df = \
            split_window(df, feature_cols, window_dates)

        model, metrics, y_pred_prob = train_single_window(
            X_train, y_train, X_test, y_test, feature_cols, window_name
        )

        # Find optimal threshold for this window
        opt_threshold, threshold_table = find_optimal_threshold(y_test, y_pred_prob)
        metrics["optimal_threshold"] = opt_threshold

        print(f"\n    Optimal threshold: {opt_threshold:.2f}")
        if len(threshold_table) > 0:
            print(f"    Threshold analysis (top 5 by profit proxy):")
            top5 = threshold_table.nlargest(5, "profit_proxy")
            for _, row in top5.iterrows():
                print(f"      t={row['threshold']:.2f}: acc={row['accuracy']:.4f}, "
                      f"signals={row['n_signals']}, "
                      f"edge={row['edge']:.4f}, proxy={row['profit_proxy']:.6f}")

        # Save model
        model_path = MODELS_DIR / f"strategy_e_{window_name}.txt"
        model.save_model(str(model_path))
        print(f"    Model saved to {model_path.name}")

        all_results[window_name] = {
            "metrics": metrics,
            "optimal_threshold": opt_threshold,
        }
        all_models[window_name] = model

    # ── Feature importance (from last window) ─────────────
    print("\n[3/5] Feature importance analysis...")
    last_model = all_models[list(WALK_FORWARD_WINDOWS.keys())[-1]]
    importance_df = get_feature_importance(last_model, feature_cols, top_n=30)
    print(f"\n  Top 30 Features (by gain):")
    for i, row in importance_df.iterrows():
        print(f"    {row['feature']:<40s} gain={row['gain_pct']:>5.2f}%  "
              f"splits={row['split_pct']:>5.2f}%")

    # ── Train final model (all data up to holdout) ────────
    print("\n[4/5] Training final model (all data up to holdout)...")
    final_train_mask = (
        (df["datetime"] < HOLDOUT_START) &
        (df[LABEL_COL] != -1)
    )
    final_train = df[final_train_mask]
    X_final = final_train[feature_cols].values
    y_final = final_train[LABEL_COL].values.astype(int)

    n_pos = (y_final == 1).sum()
    n_neg = (y_final == 0).sum()

    params = LGBM_PARAMS.copy()
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)
    n_estimators = params.pop("n_estimators", 2000)

    # Use median best_iteration from walk-forward as final n_estimators
    best_iters = [all_results[w]["metrics"]["best_iteration"]
                  for w in WALK_FORWARD_WINDOWS]
    final_n_rounds = int(np.median(best_iters))
    print(f"  Using {final_n_rounds} rounds (median of WF best iterations: {best_iters})")

    final_train_data = lgb.Dataset(X_final, label=y_final, feature_name=feature_cols)
    final_model = lgb.train(
        params,
        final_train_data,
        num_boost_round=final_n_rounds,
        callbacks=[lgb.log_evaluation(period=500)],
    )

    final_model_path = MODELS_DIR / "strategy_e_final.txt"
    final_model.save_model(str(final_model_path))
    print(f"  Final model saved to {final_model_path.name}")

    # Use average optimal threshold across windows
    avg_threshold = np.mean([all_results[w]["optimal_threshold"]
                             for w in WALK_FORWARD_WINDOWS])
    print(f"  Average optimal threshold: {avg_threshold:.3f}")

    # ── Holdout evaluation ────────────────────────────────
    print("\n[5/5] Holdout evaluation...")
    holdout_metrics = evaluate_on_holdout(final_model, df, feature_cols, avg_threshold)

    # ── Save metadata ─────────────────────────────────────
    metadata = {
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "confidence_threshold": round(avg_threshold, 3),
        "walk_forward_results": {},
        "feature_importance_top30": importance_df[["feature", "gain_pct", "split_pct"]].to_dict("records"),
        "final_model_rounds": final_n_rounds,
        "lgbm_params": {k: v for k, v in LGBM_PARAMS.items()
                        if k not in ["verbose", "n_jobs"]},
    }

    for w_name, w_result in all_results.items():
        m = w_result["metrics"]
        metadata["walk_forward_results"][w_name] = {
            "accuracy": m["accuracy"],
            "roc_auc": m["roc_auc"],
            "score_separation": m["score_separation"],
            "f1_buy": m["f1_buy"],
            "f1_sell": m["f1_sell"],
            "optimal_threshold": w_result["optimal_threshold"],
            "best_iteration": m["best_iteration"],
            "confidence_bins": m["confidence_bins"],
        }

    if holdout_metrics:
        metadata["holdout"] = holdout_metrics

    metadata_path = MODELS_DIR / "strategy_e_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\n  Metadata saved to {metadata_path.name}")

    # ── Plots ─────────────────────────────────────────────
    save_evaluation_plots(all_results, importance_df)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Strategy E — Training Summary")
    print("=" * 70)
    for w_name in WALK_FORWARD_WINDOWS:
        m = all_results[w_name]["metrics"]
        print(f"  {w_name}: Acc={m['accuracy']:.4f}  AUC={m['roc_auc']:.4f}  "
              f"Sep={m['score_separation']:.4f}  Thresh={all_results[w_name]['optimal_threshold']:.2f}")

    print(f"\n  Final threshold: {avg_threshold:.3f}")
    print(f"  Top 5 features: {', '.join(importance_df['feature'].head(5).tolist())}")

    # ── Key question: is this better than TFT? ────────────
    avg_sep = np.mean([all_results[w]["metrics"]["score_separation"]
                       for w in WALK_FORWARD_WINDOWS])
    avg_acc = np.mean([all_results[w]["metrics"]["accuracy"]
                       for w in WALK_FORWARD_WINDOWS])

    print(f"\n  Average score separation: {avg_sep:.4f}")
    print(f"  Average accuracy: {avg_acc:.4f}")

    if avg_sep > 0.05:
        print("  ✅ STRONG signal — Strategy E has meaningful predictive power!")
    elif avg_sep > 0.02:
        print("  🟡 MODERATE signal — usable as validator, test as independent carefully.")
    elif avg_sep > 0.005:
        print("  🟠 WEAK signal — similar to TFT v2. May only work as validator.")
    else:
        print("  ❌ NO signal — score separation too low. Feature engineering needed.")

    print("\n✅ Training complete! Next: integrate Strategy E into decision engine.")
    print("=" * 70)


if __name__ == "__main__":
    main()