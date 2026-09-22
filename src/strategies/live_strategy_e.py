"""
Project MIDAS v2 — Live Strategy E Inference
Loads the clean LightGBM model and generates predictions on live data.
Handles feature debiasing (swing point shift) on-the-fly.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from config import MODELS_DIR, DATA_PROCESSED, ATR_PERIOD, SWING_LOOKBACK


class LiveStrategyE:
    """
    Live inference for Strategy E.
    Uses the clean (debiased) final model.
    Applies swing feature debiasing on-the-fly before prediction.
    """

    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.loaded = False
        self._load()

    def _load(self):
        """Load model and metadata."""
        # Try clean model first, fall back to original
        for meta_name, model_name in [
            ("strategy_e_clean_metadata.json", "strategy_e_clean_final.txt"),
            ("strategy_e_metadata.json", "strategy_e_final.txt"),
        ]:
            meta_path = MODELS_DIR / meta_name
            model_path = MODELS_DIR / model_name

            if meta_path.exists() and model_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.feature_cols = meta["feature_cols"]
                self.model = lgb.Booster(model_file=str(model_path))
                self.loaded = True
                print(f"  Strategy E loaded: {model_name} ({len(self.feature_cols)} features)")
                return

        print("  WARNING: Strategy E model not found. E will be disabled.")

    def predict(self, df: pd.DataFrame, idx: int) -> float:
        """
        Get E's raw probability for the bar at idx.
        Returns probability of BUY (>0.5 = buy, <0.5 = sell).
        Returns 0.5 (neutral) if model not loaded or features missing.
        """
        if not self.loaded:
            return 0.5

        row = df.iloc[idx]

        # Build feature vector
        features = []
        for col in self.feature_cols:
            val = row.get(col, np.nan)
            features.append(val if not isinstance(val, str) else np.nan)

        X = np.array(features).reshape(1, -1)
        prob = self.model.predict(X)[0]
        return float(prob)

    def debias_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply swing feature debiasing to a DataFrame for E's prediction.
        Shifts swing-derived features forward by SWING_LOOKBACK bars.

        IMPORTANT: This modifies a COPY. The original df (used by A-D) is untouched.
        """
        df = df.copy()
        n = SWING_LOOKBACK

        # Core swing features
        for col in ["swing_high", "swing_low"]:
            if col in df.columns:
                df[col] = df[col].shift(n).fillna(0).astype(int)

        for col in ["swing_high_price", "swing_low_price"]:
            if col in df.columns:
                df[col] = df[col].shift(n)

        # Re-derive downstream
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

        # BOS features
        for col in ["bos_bull", "bos_bear"]:
            if col in df.columns:
                df[col] = df[col].shift(n).fillna(0).astype(int)

        for col in ["bos_bull_level", "bos_bear_level",
                    "ob_bull_high", "ob_bull_low", "ob_bear_high", "ob_bear_low",
                    "ob_bull_impulse_size", "ob_bear_impulse_size"]:
            if col in df.columns:
                df[col] = df[col].shift(n)

        # Equal highs/lows
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

        # M15 BOS
        for col in ["m15_bos_bull", "m15_bos_bear"]:
            if col in df.columns:
                df[col] = df[col].shift(n).fillna(0).astype(int)

        return df