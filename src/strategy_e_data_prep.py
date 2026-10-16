"""
Project MIDAS v2 — Strategy E: Data Preparation
=================================================
Builds the full Strategy E training dataset from the raw M5/M15 CSVs with
strategy_e_features.build_e_features() — the SAME function the live bot
uses — then adds ATR-filtered directional labels.

The swing features in the output are already debiased (see
strategy_e_features); the .json sidecar next to the parquet records that.

For keeping the live model current, use retrain_e.py instead (rolling
windows ending today). This script + strategy_e_clean_retrain.py are the
research path with the fixed walk-forward windows below.

Usage:
    python strategy_e_data_prep.py

Output:
    data/processed/strategy_e_train_ready.parquet  (+ .json sidecar)
    data/processed/strategy_e_feature_list.json
"""

import json
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW
from strategy_e_features import (
    LABEL_COL, LABEL_HORIZON_BARS, ATR_FILTER_THRESHOLD, TRAIN_READY_PATH,
    build_e_features, create_labels, get_feature_columns, write_dataset_info,
)

# ─── Walk-Forward Windows ────────────────────────────────
WALK_FORWARD_WINDOWS = {
    "W1": {
        "train_start": "2021-01-01",
        "train_end":   "2023-12-31",
        "test_start":  "2024-01-01",
        "test_end":    "2024-06-30",
    },
    "W2": {
        "train_start": "2021-07-01",
        "train_end":   "2024-06-30",
        "test_start":  "2024-07-01",
        "test_end":    "2024-12-31",
    },
    "W3": {
        "train_start": "2022-01-01",
        "train_end":   "2024-06-30",
        "test_start":  "2025-01-01",
        "test_end":    "2025-06-30",
    },
}

# Holdout: Jul 2025 - Apr 2026 (never trained on, final validation)
HOLDOUT_START = "2025-07-01"
HOLDOUT_END = "2026-04-30"


def print_window_stats(df: pd.DataFrame, windows: dict):
    """Print label distribution for each walk-forward window."""
    print("\n" + "=" * 70)
    print("Walk-Forward Window Statistics")
    print("=" * 70)

    for name, dates in windows.items():
        mask = (
            (df["datetime"] >= dates["test_start"]) &
            (df["datetime"] <= dates["test_end"])
        )
        subset = df[mask]
        trainable = subset[subset[LABEL_COL] != -1]

        train_mask = (
            (df["datetime"] >= dates["train_start"]) &
            (df["datetime"] <= dates["train_end"])
        )
        train_subset = df[train_mask]
        train_trainable = train_subset[train_subset[LABEL_COL] != -1]

        print(f"\n  {name}:")
        print(f"    Train: {dates['train_start']} → {dates['train_end']}")
        print(f"      Total bars: {len(train_subset):,}")
        print(f"      Trainable:  {len(train_trainable):,} "
              f"(BUY={int((train_trainable[LABEL_COL]==1).sum()):,}, "
              f"SELL={int((train_trainable[LABEL_COL]==0).sum()):,})")
        print(f"    Test:  {dates['test_start']} → {dates['test_end']}")
        print(f"      Total bars: {len(subset):,}")
        print(f"      Trainable:  {len(trainable):,} "
              f"(BUY={int((trainable[LABEL_COL]==1).sum()):,}, "
              f"SELL={int((trainable[LABEL_COL]==0).sum()):,})")

    # Holdout
    holdout_mask = (
        (df["datetime"] >= HOLDOUT_START) &
        (df["datetime"] <= HOLDOUT_END)
    )
    holdout = df[holdout_mask]
    holdout_trainable = holdout[holdout[LABEL_COL] != -1]
    print(f"\n  HOLDOUT (never trained on):")
    print(f"    {HOLDOUT_START} → {HOLDOUT_END}")
    print(f"    Total bars: {len(holdout):,}")
    print(f"    Testable:   {len(holdout_trainable):,} "
          f"(BUY={int((holdout_trainable[LABEL_COL]==1).sum()):,}, "
          f"SELL={int((holdout_trainable[LABEL_COL]==0).sum()):,})")


def main():
    print("=" * 70)
    print("MIDAS v2 — Strategy E: Data Preparation")
    print("=" * 70)

    print("\n[1/3] Loading raw M5 + M15 data...")
    m5 = pd.read_csv(DATA_RAW / "XAUUSD_M5.csv", parse_dates=["datetime"])
    m15 = pd.read_csv(DATA_RAW / "XAUUSD_M15.csv", parse_dates=["datetime"])
    print(f"  M5: {len(m5):,} bars | M15: {len(m15):,} bars")

    print("\n[2/3] Building features (shared live/training pipeline)...")
    df = build_e_features(m5, m15)
    print(f"  {len(df.columns)} columns")

    print(f"\n[3/3] Creating labels (horizon={LABEL_HORIZON_BARS} bars, "
          f"ATR threshold={ATR_FILTER_THRESHOLD})...")
    df = create_labels(df)
    counts = df[LABEL_COL].value_counts()
    print(f"  BUY={counts.get(1, 0):,}  SELL={counts.get(0, 0):,}  WAIT={counts.get(-1, 0):,}")

    feature_cols = get_feature_columns(df)
    print(f"  Feature count: {len(feature_cols)}")
    with open(DATA_PROCESSED / "strategy_e_feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    df.to_parquet(TRAIN_READY_PATH, index=False)
    write_dataset_info(TRAIN_READY_PATH, data_end=df["datetime"].max())
    print(f"  Dataset saved to {TRAIN_READY_PATH.name} ({len(df):,} rows, {len(df.columns)} cols)")

    print_window_stats(df, WALK_FORWARD_WINDOWS)

    print("\n" + "=" * 70)
    print("✅ Data preparation complete! Next: python strategy_e_clean_retrain.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
