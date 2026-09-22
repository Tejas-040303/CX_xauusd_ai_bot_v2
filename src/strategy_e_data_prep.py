"""
Project MIDAS v2 — Strategy E: Data Preparation
=================================================
Loads processed M5 data (with all 167 features from feature_engineering.py),
creates ATR-filtered directional labels, handles feature types, and exports
train-ready datasets for walk-forward validation.

Usage:
    python strategy_e_data_prep.py

Output:
    data/processed/strategy_e_train_ready.parquet  (full dataset with labels)
    data/processed/strategy_e_feature_list.json     (feature column names)
    Prints label distribution stats per walk-forward window.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_PROCESSED, DATA_RAW, ATR_PERIOD, PIP_VALUE,
    BACKTEST_TRAIN_END, BACKTEST_FORWARD_START
)
from feature_engineering import FeatureEngine


# ─── Label Configuration ─────────────────────────────────
LABEL_HORIZON_BARS = 12          # 12 M5 bars = 1 hour lookahead
ATR_FILTER_THRESHOLD = 0.5       # Minimum move in ATR multiples to qualify
LABEL_COL = "label"              # 1 = BUY, 0 = SELL, -1 = WAIT (excluded)

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


# ─── Columns to EXCLUDE from features ────────────────────
# These are either raw prices (non-stationary), identifiers,
# or string columns that need special handling.
EXCLUDE_COLS = [
    # Identifiers / metadata
    "datetime", "timeframe",
    # Raw prices (non-stationary — their info is in derived features)
    "open", "high", "low", "close", "volume",
    # Raw price levels (non-stationary)
    "swing_high_price", "swing_low_price",
    "last_swing_high", "last_swing_low",
    "bos_bull_level", "bos_bear_level",
    "ob_bull_high", "ob_bull_low", "ob_bear_high", "ob_bear_low",
    "equal_highs_level", "equal_lows_level",
    "last_equal_highs_level", "last_equal_lows_level",
    "fvg_bull_top", "fvg_bull_bottom", "fvg_bull_ce",
    "fvg_bear_top", "fvg_bear_bottom", "fvg_bear_ce",
    "asian_range_high", "asian_range_low", "asian_range_mid",
    # String columns (handled separately via encoding)
    "session", "vol_regime", "fvg_direction",
    # Target-related (created during labeling)
    LABEL_COL, "future_return", "future_return_atr",
    # Temporary columns
    "_trade_date",
]

# These raw price columns CAN be made stationary by normalizing to ATR.
# We'll create relative versions instead of dropping entirely.
PRICE_COLS_TO_NORMALIZE = {
    "ob_bull_impulse_size": "ob_bull_impulse_size_atr",
    "ob_bear_impulse_size": "ob_bear_impulse_size_atr",
    "fvg_bull_size": "fvg_bull_size_atr",  # Already exists
    "fvg_bear_size": "fvg_bear_size_atr",  # Already exists
    "asian_range_width": "asian_range_width_atr",  # Already exists
}


def load_m5_with_all_features() -> pd.DataFrame:
    """
    Load M5 data and compute ALL feature stages.
    If a pre-computed file exists, load it; otherwise compute from raw.
    """
    processed_file = DATA_PROCESSED / "XAUUSD_M5_all_features.parquet"

    if processed_file.exists():
        print(f"  Loading pre-computed features from {processed_file.name}...")
        df = pd.read_parquet(processed_file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
        return df

    print("  Pre-computed file not found. Computing all features from raw data...")

    # Load raw M5
    raw_m5 = DATA_RAW / "XAUUSD_M5.csv"
    if not raw_m5.exists():
        raise FileNotFoundError(f"Raw M5 data not found at {raw_m5}")

    df = pd.read_csv(raw_m5, parse_dates=["datetime"])
    print(f"  Raw M5: {len(df):,} candles")

    engine = FeatureEngine()

    # Phase 0: Shared features
    print("  Computing shared features...")
    df = engine.compute_shared(df, timeframe="M5")

    # Phase 1: FVG features
    print("  Computing FVG features...")
    df = engine.compute_fvg(df)

    # Phase 2: Trend features
    print("  Computing trend features...")
    df = engine.compute_trend(df)

    # Phase 3: Session range features
    print("  Computing session range features...")
    df = engine.compute_session_range(df)

    # Phase 4: SMC features
    print("  Computing SMC features...")
    df = engine.compute_smc(df)

    print(f"  All features computed: {len(df.columns)} columns")

    # Save for future runs
    df.to_parquet(processed_file, index=False)
    print(f"  Saved to {processed_file.name}")

    return df


def merge_m15_context(df_m5: pd.DataFrame) -> pd.DataFrame:
    """
    Merge M15 features as higher-timeframe context into M5 data.
    Each M5 bar gets the most recent M15 bar's features (forward-filled).

    This gives Strategy E multi-timeframe awareness:
    M5 features = what's happening NOW (micro structure)
    M15 features = what's happening on the STRUCTURE level (BOS, OBs, FVGs)
    """
    raw_m15 = DATA_RAW / "XAUUSD_M15.csv"
    if not raw_m15.exists():
        print("  WARNING: M15 raw data not found. Skipping M15 context merge.")
        return df_m5

    print("  Loading M15 data for context merge...")
    df_m15 = pd.read_csv(raw_m15, parse_dates=["datetime"])

    engine = FeatureEngine()
    df_m15 = engine.compute_shared(df_m15, timeframe="M15")
    df_m15 = engine.compute_fvg(df_m15)
    df_m15 = engine.compute_smc(df_m15)

    # Select M15 features to merge (structural ones, not redundant with M5)
    m15_features = [
        "datetime",
        f"atr_{ATR_PERIOD}",
        "ema_stack",
        f"adx_{ATR_PERIOD}",
        f"rsi_{ATR_PERIOD}",
        "bos_bull", "bos_bear",
        "fvg_detected",
        "body_ratio",
        "ema_9_slope", "ema_21_slope", "ema_50_slope",
        "above_ema200",
        "trend_strong",
        "atr_percentile",
    ]

    # Keep only columns that actually exist
    m15_features = [c for c in m15_features if c in df_m15.columns]
    df_m15_subset = df_m15[m15_features].copy()

    # Rename with m15_ prefix (except datetime)
    rename_map = {c: f"m15_{c}" for c in m15_features if c != "datetime"}
    df_m15_subset = df_m15_subset.rename(columns=rename_map)

    # Merge using merge_asof: each M5 bar gets the most recent M15 bar
    df_m5 = df_m5.sort_values("datetime")
    df_m15_subset = df_m15_subset.sort_values("datetime")

    df_merged = pd.merge_asof(
        df_m5,
        df_m15_subset,
        on="datetime",
        direction="backward",  # Use the most recent M15 bar at or before this M5 bar
    )

    new_cols = len(df_merged.columns) - len(df_m5.columns)
    print(f"  M15 context merged: +{new_cols} columns → {len(df_merged.columns)} total")

    return df_merged


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ATR-filtered directional labels.

    For each M5 bar, look ahead LABEL_HORIZON_BARS (12 = 1 hour):
      - future_return = close[i + horizon] - close[i]
      - If |future_return| < ATR_FILTER_THRESHOLD * ATR → WAIT (-1)
      - If future_return > threshold → BUY (1)
      - If future_return < -threshold → SELL (0)

    WAIT bars are excluded from training but kept in the dataframe
    for completeness (Strategy E outputs WAIT when unsure).
    """
    atr_col = f"atr_{ATR_PERIOD}"
    horizon = LABEL_HORIZON_BARS

    print(f"  Creating labels (horizon={horizon} bars, ATR threshold={ATR_FILTER_THRESHOLD})...")

    # Future close price
    df["_future_close"] = df["close"].shift(-horizon)
    df["future_return"] = df["_future_close"] - df["close"]
    df["future_return_atr"] = df["future_return"] / df[atr_col].replace(0, np.nan)

    # ATR-filtered labels
    threshold = ATR_FILTER_THRESHOLD
    df[LABEL_COL] = -1  # Default: WAIT

    buy_mask = df["future_return_atr"] >= threshold
    sell_mask = df["future_return_atr"] <= -threshold

    df.loc[buy_mask, LABEL_COL] = 1   # BUY
    df.loc[sell_mask, LABEL_COL] = 0  # SELL

    # Drop rows where we can't compute labels (last N bars)
    valid_mask = df["_future_close"].notna()
    df = df[valid_mask].copy()

    # Drop temp column
    df = df.drop(columns=["_future_close"])

    # Stats
    total = len(df)
    buys = (df[LABEL_COL] == 1).sum()
    sells = (df[LABEL_COL] == 0).sum()
    waits = (df[LABEL_COL] == -1).sum()
    trainable = buys + sells

    print(f"  Labels created:")
    print(f"    BUY:  {buys:>7,} ({100*buys/total:.1f}%)")
    print(f"    SELL: {sells:>7,} ({100*sells/total:.1f}%)")
    print(f"    WAIT: {waits:>7,} ({100*waits/total:.1f}%) — excluded from training")
    print(f"    Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"    Buy/Sell ratio: {buys/max(sells,1):.3f}")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode string/categorical columns for LightGBM.
    LightGBM handles categoricals natively, but we need integer encoding.
    """
    cat_cols = {
        "session": ["off", "asian", "london", "london_ny_overlap", "new_york"],
        "vol_regime": ["low_volatility", "normal", "high_volatility"],
        "fvg_direction": ["none", "bull", "bear", "both"],
    }

    for col, categories in cat_cols.items():
        if col in df.columns:
            cat_map = {cat: i for i, cat in enumerate(categories)}
            df[f"{col}_encoded"] = df[col].map(cat_map).fillna(-1).astype(int)

    return df


def add_relative_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert absolute price-level features into ATR-relative features.
    This makes them stationary and comparable across different price regimes.
    """
    atr_col = f"atr_{ATR_PERIOD}"
    if atr_col not in df.columns:
        return df

    atr = df[atr_col].replace(0, np.nan)

    # Distance from close to OB zones (if OB exists on this bar)
    for side in ["bull", "bear"]:
        high_col = f"ob_{side}_high"
        low_col = f"ob_{side}_low"
        if high_col in df.columns:
            ob_mid = (df[high_col] + df[low_col]) / 2
            df[f"dist_to_ob_{side}_atr"] = (df["close"] - ob_mid) / atr

    # Distance to equal highs/lows levels
    if "last_equal_highs_level" in df.columns:
        df["dist_to_eq_highs_atr"] = (df["close"] - df["last_equal_highs_level"]) / atr
    if "last_equal_lows_level" in df.columns:
        df["dist_to_eq_lows_atr"] = (df["close"] - df["last_equal_lows_level"]) / atr

    # Impulse sizes normalized (some may already exist)
    for col, norm_col in PRICE_COLS_TO_NORMALIZE.items():
        if col in df.columns and norm_col not in df.columns:
            df[norm_col] = df[col] / atr

    # Swing distances normalized
    if "dist_to_swing_high" in df.columns:
        df["dist_to_swing_high_atr"] = df["dist_to_swing_high"] / atr
    if "dist_to_swing_low" in df.columns:
        df["dist_to_swing_low_atr"] = df["dist_to_swing_low"] / atr

    # Session range distances normalized
    if "dist_to_range_high" in df.columns:
        df["dist_to_range_high_atr"] = df["dist_to_range_high"] / atr
    if "dist_to_range_low" in df.columns:
        df["dist_to_range_low_atr"] = df["dist_to_range_low"] / atr

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Determine which columns are valid features for LightGBM.
    Excludes metadata, raw prices, string columns, and the target.
    """
    all_cols = set(df.columns)
    exclude = set(EXCLUDE_COLS)

    # Also exclude any column that's still a string/object type
    object_cols = set(df.select_dtypes(include=["object", "category"]).columns)
    exclude.update(object_cols)

    # Also exclude future_return variants (leakage!)
    leakage_cols = {c for c in all_cols if "future" in c.lower()}
    exclude.update(leakage_cols)

    feature_cols = sorted(all_cols - exclude)

    # Verify all are numeric
    numeric_features = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            print(f"  WARNING: Dropping non-numeric feature: {col} (dtype={df[col].dtype})")

    return numeric_features


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

    # ── Step 1: Load M5 data with all features ────────────
    print("\n[1/6] Loading M5 data with all features...")
    df = load_m5_with_all_features()

    # ── Step 2: Merge M15 context ─────────────────────────
    print("\n[2/6] Merging M15 higher-timeframe context...")
    df = merge_m15_context(df)

    # ── Step 3: Add relative price features ───────────────
    print("\n[3/6] Adding ATR-relative price features...")
    df = add_relative_price_features(df)

    # ── Step 4: Encode categoricals ───────────────────────
    print("\n[4/6] Encoding categorical columns...")
    df = encode_categoricals(df)

    # ── Step 5: Create labels ─────────────────────────────
    print("\n[5/6] Creating ATR-filtered directional labels...")
    df = create_labels(df)

    # ── Step 6: Identify and save feature list ────────────
    print("\n[6/6] Identifying feature columns...")
    feature_cols = get_feature_columns(df)
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  Sample features: {feature_cols[:10]}...")

    # Save feature list
    feature_list_path = DATA_PROCESSED / "strategy_e_feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Feature list saved to {feature_list_path.name}")

    # Save train-ready dataset
    output_path = DATA_PROCESSED / "strategy_e_train_ready.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  Dataset saved to {output_path.name} ({len(df):,} rows, {len(df.columns)} cols)")

    # Print walk-forward stats
    print_window_stats(df, WALK_FORWARD_WINDOWS)

    print("\n" + "=" * 70)
    print("✅ Data preparation complete! Next: python strategy_e_train.py")
    print("=" * 70)


if __name__ == "__main__":
    main()