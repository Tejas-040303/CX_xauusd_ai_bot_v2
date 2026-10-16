"""
Project MIDAS v2 — Strategy E: Shared Feature Pipeline
=======================================================
ONE place that turns raw M5 + M15 bars into Strategy E model inputs.

Used by:
  - strategy_e_data_prep.py / retrain_e.py   (training)
  - strategies/live_strategy_e.py            (live inference)

Training and live MUST build features the same way. Before this module they
didn't: live skipped compute_fvg/compute_smc on M5, the relative-price
features and the categorical encodings, so 28 of 133 model inputs were
always NaN in live trading.

Pipeline (order matters):
  1. M5:  shared → fvg → trend → session range → smc
  2. M15: shared → fvg → smc
  3. Attach M15 context to each M5 bar using CLOSE times: an M5 bar only
     sees M15 bars that had already closed when the M5 bar closed.
     (The old merge used open times, which leaked up to 10 minutes of the
     future in training — live trading can never see that.)
  4. Debias swing-derived features (shift by SWING_LOOKBACK — a swing is
     only confirmed SWING_LOOKBACK bars after it happens)
  5. ATR-relative price features — AFTER debiasing, so they don't
     reintroduce the look-ahead step 4 removed
  6. Integer-encode categoricals

Bump PIPELINE_VERSION whenever the output of build_e_features() changes:
models trained with another version are treated as stale and retrained.
"""

import contextlib
import io
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from config import ATR_PERIOD, SWING_LOOKBACK, DATA_PROCESSED
from feature_engineering import FeatureEngine

PIPELINE_VERSION = 3  # v3: session labels / Asian range in real UTC (DST-aware)

# ─── Labels ──────────────────────────────────────────────
LABEL_HORIZON_BARS = 12          # 12 M5 bars = 1 hour lookahead
ATR_FILTER_THRESHOLD = 0.5       # Minimum move in ATR multiples to qualify
LABEL_COL = "label"              # 1 = BUY, 0 = SELL, -1 = WAIT (excluded)

M5_BAR = timedelta(minutes=5)
M15_BAR = timedelta(minutes=15)

# M15 features attached to each M5 bar as higher-timeframe context
M15_CONTEXT_FEATURES = [
    f"atr_{ATR_PERIOD}", "ema_stack", f"adx_{ATR_PERIOD}", f"rsi_{ATR_PERIOD}",
    "bos_bull", "bos_bear", "fvg_detected", "body_ratio",
    "ema_9_slope", "ema_21_slope", "ema_50_slope",
    "above_ema200", "trend_strong", "atr_percentile",
]

# Columns that are never model inputs
EXCLUDE_COLS = [
    # Identifiers / metadata
    "datetime", "timeframe",
    # Raw prices (non-stationary — their info is in derived features)
    "open", "high", "low", "close", "volume", "spread",
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
    # String columns (encoded separately)
    "session", "vol_regime", "fvg_direction",
    # Target-related
    LABEL_COL, "future_return", "future_return_atr",
    # Temporary columns
    "_trade_date",
]

# Absolute sizes that become stationary once divided by ATR
PRICE_COLS_TO_NORMALIZE = {
    "ob_bull_impulse_size": "ob_bull_impulse_size_atr",
    "ob_bear_impulse_size": "ob_bear_impulse_size_atr",
    "fvg_bull_size": "fvg_bull_size_atr",
    "fvg_bear_size": "fvg_bear_size_atr",
    "asian_range_width": "asian_range_width_atr",
}

CATEGORIES = {
    "session": ["off", "asian", "london", "london_ny_overlap", "new_york"],
    "vol_regime": ["low_volatility", "normal", "high_volatility"],
    "fvg_direction": ["none", "bull", "bear", "both"],
}


# ═════════════════════════════════════════════════════════
# Pipeline steps
# ═════════════════════════════════════════════════════════

def compute_m5_features(df_m5: pd.DataFrame, engine: FeatureEngine) -> pd.DataFrame:
    df = engine.compute_shared(df_m5, timeframe="M5")
    df = engine.compute_fvg(df)
    df = engine.compute_trend(df)
    df = engine.compute_session_range(df)
    df = engine.compute_smc(df)
    return df


def compute_m15_features(df_m15: pd.DataFrame, engine: FeatureEngine) -> pd.DataFrame:
    df = engine.compute_shared(df_m15, timeframe="M15")
    df = engine.compute_fvg(df)
    df = engine.compute_smc(df)
    return df


def merge_m15_context(df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    """Give each M5 bar the features of the last M15 bar CLOSED by the time
    that M5 bar closed (bar datetimes are open times)."""
    cols = ["datetime"] + [c for c in M15_CONTEXT_FEATURES if c in df_m15.columns]
    m15 = df_m15[cols].rename(columns={c: f"m15_{c}" for c in cols if c != "datetime"})
    m15["_close_time"] = m15.pop("datetime") + M15_BAR
    m15 = m15.sort_values("_close_time")

    m5 = df_m5.sort_values("datetime").copy()
    m5["_close_time"] = m5["datetime"] + M5_BAR

    merged = pd.merge_asof(m5, m15, on="_close_time", direction="backward")
    return merged.drop(columns=["_close_time"]).reset_index(drop=True)


def debias_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shift swing-derived features forward by SWING_LOOKBACK bars.

    Swing points are detected with a centred window (bars i-n .. i+n), so in
    real time a swing at bar i is only known at bar i+n.
    """
    n = SWING_LOOKBACK
    df = df.copy()
    atr_col = f"atr_{ATR_PERIOD}"
    atr = df[atr_col].replace(0, np.nan) if atr_col in df.columns else None

    for col in ["swing_high", "swing_low", "bos_bull", "bos_bear",
                "m15_bos_bull", "m15_bos_bear"]:
        if col in df.columns:
            df[col] = df[col].shift(n).fillna(0).astype(int)

    for col in ["swing_high_price", "swing_low_price",
                "bos_bull_level", "bos_bear_level",
                "ob_bull_high", "ob_bull_low", "ob_bear_high", "ob_bear_low",
                "ob_bull_impulse_size", "ob_bear_impulse_size",
                "equal_highs", "equal_lows", "equal_highs_count", "equal_lows_count",
                "equal_highs_level", "equal_lows_level"]:
        if col in df.columns:
            df[col] = df[col].shift(n)

    # Re-derive what was built from the (now shifted) columns
    if "swing_high_price" in df.columns:
        df["last_swing_high"] = df["swing_high_price"].ffill()
        df["dist_to_swing_high"] = df["close"] - df["last_swing_high"]
    if "swing_low_price" in df.columns:
        df["last_swing_low"] = df["swing_low_price"].ffill()
        df["dist_to_swing_low"] = df["close"] - df["last_swing_low"]
    if "equal_highs_level" in df.columns:
        df["last_equal_highs_level"] = df["equal_highs_level"].ffill()
    if "equal_lows_level" in df.columns:
        df["last_equal_lows_level"] = df["equal_lows_level"].ffill()

    if atr is not None:
        for side in ["high", "low"]:
            if f"dist_to_swing_{side}" in df.columns:
                df[f"dist_to_swing_{side}_atr"] = df[f"dist_to_swing_{side}"] / atr
    return df


def add_relative_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn absolute price levels into ATR-relative (stationary) features."""
    atr_col = f"atr_{ATR_PERIOD}"
    if atr_col not in df.columns:
        return df
    atr = df[atr_col].replace(0, np.nan)

    for side in ["bull", "bear"]:
        if f"ob_{side}_high" in df.columns:
            ob_mid = (df[f"ob_{side}_high"] + df[f"ob_{side}_low"]) / 2
            df[f"dist_to_ob_{side}_atr"] = (df["close"] - ob_mid) / atr

    if "last_equal_highs_level" in df.columns:
        df["dist_to_eq_highs_atr"] = (df["close"] - df["last_equal_highs_level"]) / atr
    if "last_equal_lows_level" in df.columns:
        df["dist_to_eq_lows_atr"] = (df["close"] - df["last_equal_lows_level"]) / atr

    for col, norm_col in PRICE_COLS_TO_NORMALIZE.items():
        if col in df.columns and norm_col not in df.columns:  # engine may already provide it
            df[norm_col] = df[col] / atr

    for name in ["swing_high", "swing_low", "range_high", "range_low"]:
        if f"dist_to_{name}" in df.columns:
            df[f"dist_to_{name}_atr"] = df[f"dist_to_{name}"] / atr
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col, categories in CATEGORIES.items():
        if col in df.columns:
            cat_map = {cat: i for i, cat in enumerate(categories)}
            df[f"{col}_encoded"] = df[col].map(cat_map).fillna(-1).astype(int)
    return df


def build_e_features(df_m5: pd.DataFrame, df_m15: pd.DataFrame,
                     engine: FeatureEngine = None, quiet: bool = False) -> pd.DataFrame:
    """Raw M5 + M15 bars (closed bars only) → Strategy E feature frame."""
    engine = engine or FeatureEngine()
    out = io.StringIO() if quiet else None
    with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
        m5 = compute_m5_features(df_m5, engine)
        m15 = compute_m15_features(df_m15, engine)
    df = merge_m15_context(m5, m15)
    df = debias_swing_features(df)
    df = add_relative_price_features(df)
    df = encode_categoricals(df)
    return df


# ═════════════════════════════════════════════════════════
# Training helpers
# ═════════════════════════════════════════════════════════

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """ATR-filtered direction LABEL_HORIZON_BARS ahead: 1=BUY, 0=SELL, -1=WAIT.
    Rows without a full horizon ahead (the newest bars) are dropped."""
    atr_col = f"atr_{ATR_PERIOD}"
    future_close = df["close"].shift(-LABEL_HORIZON_BARS)
    df["future_return"] = future_close - df["close"]
    df["future_return_atr"] = df["future_return"] / df[atr_col].replace(0, np.nan)

    df[LABEL_COL] = -1
    df.loc[df["future_return_atr"] >= ATR_FILTER_THRESHOLD, LABEL_COL] = 1
    df.loc[df["future_return_atr"] <= -ATR_FILTER_THRESHOLD, LABEL_COL] = 0
    return df[future_close.notna()].copy()


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = set(EXCLUDE_COLS)
    exclude.update(df.select_dtypes(include=["object", "category", "string"]).columns)
    exclude.update(c for c in df.columns if "future" in c.lower())
    return [c for c in sorted(set(df.columns) - exclude)
            if pd.api.types.is_numeric_dtype(df[c])]


# Sidecar written next to datasets built by this module, so research scripts
# know the swing features are already debiased (and don't shift them twice).
def write_dataset_info(parquet_path, **extra):
    info = {"pipeline_version": PIPELINE_VERSION, "debiased": True, **extra}
    with open(parquet_path.with_suffix(".json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


def is_debiased_dataset(parquet_path) -> bool:
    try:
        with open(parquet_path.with_suffix(".json")) as f:
            return bool(json.load(f).get("debiased"))
    except (OSError, ValueError):
        return False


TRAIN_READY_PATH = DATA_PROCESSED / "strategy_e_train_ready.parquet"
