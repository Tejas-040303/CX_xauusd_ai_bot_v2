"""
Project MIDAS v2 — Strategy E: Keep the Model Up to Date
=========================================================
Bridges the gap between the data a model was trained on and today.

  1. Status: how old is the model's training data, and was it built with
     the current feature pipeline (strategy_e_features.PIPELINE_VERSION)?
  2. Sync:   append the M5/M15 bars missing since the last download (MT5)
  3. Build:  features + labels for a rolling E_TRAIN_YEARS window that ends
             at the latest bar (not at a hardcoded date)
  4. Check:  two walk-forward blocks of E_TEST_MONTHS each, ending today.
             The newest block is data the candidate model has never seen;
             it must reach E_MIN_TEST_AUC or the old model stays in place.
  5. Train:  final model on the whole window, rounds = median early-stop
  6. Save:   previous model archived to models/archive/<time>/, new one
             swapped in atomically

main.py runs ensure_fresh_model() at startup. Standalone:

    python src/retrain_e.py --check      # show model status only
    python src/retrain_e.py              # retrain if stale (needs MT5 running)
    python src/retrain_e.py --force      # retrain now
    python src/retrain_e.py --no-sync    # use saved CSVs, don't touch MT5
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS_DIR, E_MODEL_MAX_AGE_DAYS, E_TRAIN_YEARS, E_TEST_MONTHS, E_MIN_TEST_AUC,
)
from strategy_e_features import (
    PIPELINE_VERSION, LABEL_COL, LABEL_HORIZON_BARS, M5_BAR,
    build_e_features, create_labels, get_feature_columns,
)

MODEL_FILE = MODELS_DIR / "strategy_e_clean_final.txt"
META_FILE = MODELS_DIR / "strategy_e_clean_metadata.json"
ARCHIVE_DIR = MODELS_DIR / "archive"

# Same hyperparameters as strategy_e_clean_retrain.py
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
MAX_ROUNDS = 2000

# Extra history before the first training bar so rolling indicators settle
WARMUP = timedelta(days=30)
# Gap between train and test: labels look LABEL_HORIZON_BARS ahead, so the
# last train rows would otherwise be labelled with test-period prices
EMBARGO = LABEL_HORIZON_BARS * M5_BAR


def _utc_now() -> pd.Timestamp:
    # Naive, to compare with bar timestamps (naive broker time). Broker time is
    # a few hours off UTC, which doesn't matter for an age measured in days.
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))


# ═════════════════════════════════════════════════════════
# Status
# ═════════════════════════════════════════════════════════

def read_metadata():
    try:
        with open(META_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def model_status(now: pd.Timestamp = None) -> dict:
    """state: fresh | stale | incompatible | missing"""
    now = now if now is not None else _utc_now()
    meta = read_metadata()
    if meta is None or not MODEL_FILE.exists():
        return {"state": "missing", "reason": "no trained model found"}

    version = meta.get("pipeline_version", 1)
    trained_through = meta.get("trained_through")
    age_days = (now - pd.Timestamp(trained_through)).days if trained_through else None
    status = {"pipeline_version": version, "trained_through": trained_through,
              "age_days": age_days}

    if version != PIPELINE_VERSION:
        status.update(state="incompatible",
                      reason=f"built with feature pipeline v{version}, current is v{PIPELINE_VERSION}")
    elif age_days is None:
        status.update(state="stale", reason="training data end date unknown")
    elif age_days > E_MODEL_MAX_AGE_DAYS:
        status.update(state="stale",
                      reason=f"training data ends {str(trained_through)[:10]} ({age_days} days ago)")
    else:
        status.update(state="fresh",
                      reason=f"training data ends {str(trained_through)[:10]} ({age_days} days ago)")
    return status


# ═════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════

def sync_data(log=print):
    """Append missing M5/M15 bars from MT5 (MT5 must be initialized)."""
    from data_downloader import update_timeframe
    for tf in ("M5", "M15"):
        update_timeframe(tf)


def build_dataset(log=print):
    """Features + labels for the E_TRAIN_YEARS ending at the latest saved bar."""
    from data_downloader import raw_file

    m5 = pd.read_csv(raw_file("M5"), parse_dates=["datetime"])
    m15 = pd.read_csv(raw_file("M15"), parse_dates=["datetime"])
    data_end = m5["datetime"].max()
    start = data_end - pd.DateOffset(years=E_TRAIN_YEARS)

    m5 = m5[m5["datetime"] >= start - WARMUP].reset_index(drop=True)
    m15 = m15[m15["datetime"] >= start - WARMUP - timedelta(days=15)].reset_index(drop=True)
    log(f"  Building features: {len(m5):,} M5 + {len(m15):,} M15 bars "
        f"({start:%Y-%m-%d} → {data_end:%Y-%m-%d})")

    df = build_e_features(m5, m15, quiet=True)
    df = create_labels(df)
    df = df[df["datetime"] >= start].reset_index(drop=True)
    return df, data_end


# ═════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════

def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), ties averaged."""
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit(X, y, feature_cols, rounds: int = None):
    """Train one model. Without `rounds`, early-stop on the last 10% (in time)."""
    import lightgbm as lgb

    params = dict(LGBM_PARAMS, scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    if rounds is not None:
        ds = lgb.Dataset(X, label=y, feature_name=feature_cols)
        return lgb.train(params, ds, num_boost_round=rounds), rounds

    cut = int(len(X) * 0.9)
    train_ds = lgb.Dataset(X[:cut], label=y[:cut], feature_name=feature_cols)
    val_ds = lgb.Dataset(X[cut:], label=y[cut:], reference=train_ds, feature_name=feature_cols)
    model = lgb.train(params, train_ds, num_boost_round=MAX_ROUNDS, valid_sets=[val_ds],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return model, max(int(model.best_iteration or 0), 1)


def _xy(df, feature_cols):
    return df[feature_cols].to_numpy(dtype=np.float32), df[LABEL_COL].to_numpy(dtype=int)


def walk_forward(labeled: pd.DataFrame, feature_cols, data_end, log=print) -> list:
    """Two out-of-sample blocks of E_TEST_MONTHS, the newest ending at data_end."""
    results = []
    for k in (1, 0):
        test_end = data_end - pd.DateOffset(months=k * E_TEST_MONTHS)
        test_start = test_end - pd.DateOffset(months=E_TEST_MONTHS)
        train = labeled[labeled["datetime"] < test_start - EMBARGO]
        test = labeled[(labeled["datetime"] >= test_start) & (labeled["datetime"] <= test_end)]
        if len(train) < 10_000 or len(test) < 1_000:
            raise RuntimeError(f"not enough data for walk-forward block ending {test_end:%Y-%m-%d} "
                               f"(train={len(train):,}, test={len(test):,})")

        X_tr, y_tr = _xy(train, feature_cols)
        X_te, y_te = _xy(test, feature_cols)
        model, best_iter = fit(X_tr, y_tr, feature_cols)
        p = model.predict(X_te, num_iteration=best_iter)
        r = {
            "test_start": f"{test_start:%Y-%m-%d}", "test_end": f"{test_end:%Y-%m-%d}",
            "n_train": len(train), "n_test": len(test),
            "auc": roc_auc(y_te, p),
            "accuracy": float(((p >= 0.5).astype(int) == y_te).mean()),
            "separation": float(p[y_te == 1].mean() - p[y_te == 0].mean()),
            "best_iteration": best_iter,
        }
        log(f"  Walk-forward {r['test_start']} → {r['test_end']}: AUC={r['auc']:.4f} "
            f"Acc={r['accuracy']:.4f} Sep={r['separation']:+.4f} ({best_iter} rounds)")
        results.append(r)
    return results


def save_model(model, meta: dict, log=print):
    """Archive the current model, then swap the new one in."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_FILE.exists():
        dest = ARCHIVE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        for f in (MODEL_FILE, META_FILE):
            if f.exists():
                shutil.copy2(f, dest / f.name)
        log(f"  Previous model archived to {dest}")

    tmp_model = MODEL_FILE.with_suffix(".txt.tmp")
    tmp_meta = META_FILE.with_suffix(".json.tmp")
    model.save_model(str(tmp_model))
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    tmp_model.replace(MODEL_FILE)
    tmp_meta.replace(META_FILE)


def retrain(sync: bool = True, log=print) -> dict:
    t0 = time.time()
    if sync:
        log("  Syncing M5/M15 data from MT5...")
        sync_data(log)

    df, data_end = build_dataset(log)
    feature_cols = get_feature_columns(df)
    labeled = df[df[LABEL_COL] != -1].reset_index(drop=True)
    log(f"  Dataset: {len(labeled):,} labelled bars, {len(feature_cols)} features")

    windows = walk_forward(labeled, feature_cols, data_end, log)
    latest = windows[-1]
    if not (latest["auc"] >= E_MIN_TEST_AUC and latest["separation"] > 0):
        return {"ok": False, "windows": windows, "data_end": str(data_end),
                "reason": f"latest out-of-sample AUC {latest['auc']:.4f} "
                          f"(needs ≥ {E_MIN_TEST_AUC}) / separation {latest['separation']:+.4f}"}

    rounds = int(np.median([w["best_iteration"] for w in windows]))
    log(f"  Training final model on {len(labeled):,} bars ({rounds} rounds)...")
    X, y = _xy(labeled, feature_cols)
    model, _ = fit(X, y, feature_cols, rounds=rounds)

    meta = {
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "pipeline": "auto_retrain",
        "pipeline_version": PIPELINE_VERSION,
        "trained_through": str(labeled["datetime"].max()),
        "train_start": str(labeled["datetime"].min()),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rounds": rounds,
        "n_train": len(labeled),
        "walk_forward": windows,
    }
    save_model(model, meta, log)
    minutes = (time.time() - t0) / 60
    log(f"  ✅ Strategy E retrained through {meta['trained_through'][:16]} in {minutes:.1f} min")
    return {"ok": True, "windows": windows, "meta": meta, "minutes": round(minutes, 1)}


def ensure_fresh_model(force: bool = False, sync: bool = True, log=print) -> dict:
    """Retrain if the model is missing, stale or from an older pipeline.

    Returns {"action": none|retrained|rejected|failed, "status": ..., ...}.
    Never raises: on any failure the existing model (if any) stays in place.
    """
    status = model_status()
    if status["state"] == "fresh" and not force:
        log(f"  Strategy E model is up to date ({status['reason']})")
        return {"action": "none", "status": status}

    log(f"  Strategy E model {status['state']}: {status['reason']} → retraining")
    try:
        result = retrain(sync=sync, log=log)
    except Exception as e:
        traceback.print_exc()
        return {"action": "failed", "status": status, "error": f"{type(e).__name__}: {e}"}
    if not result["ok"]:
        log(f"  ⚠️ New model rejected: {result['reason']}. Keeping the current model.")
        return {"action": "rejected", "status": status, **result}
    return {"action": "retrained", "status": status, **result}


def main():
    parser = argparse.ArgumentParser(description="Keep Strategy E's model up to date")
    parser.add_argument("--check", action="store_true", help="Only print model status")
    parser.add_argument("--force", action="store_true", help="Retrain even if the model is fresh")
    parser.add_argument("--no-sync", action="store_true", help="Don't download new bars from MT5")
    args = parser.parse_args()

    if args.check:
        print(json.dumps(model_status(), indent=2, default=str))
        return

    mt5_started = False
    if not args.no_sync:
        from data_downloader import initialize_mt5
        initialize_mt5()
        mt5_started = True
    try:
        result = ensure_fresh_model(force=args.force, sync=not args.no_sync)
    finally:
        if mt5_started:
            import MetaTrader5 as mt5
            mt5.shutdown()
    print(f"\nResult: {result['action']}")
    sys.exit(0 if result["action"] in ("none", "retrained") else 1)


if __name__ == "__main__":
    main()
