"""
Project MIDAS v2 — Live Strategy E Inference
Loads the LightGBM model and generates predictions on live data.

Features are built by strategy_e_features.build_e_features(), the same
function training uses, so live inputs match what the model learned from.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import MODELS_DIR
from strategy_e_features import PIPELINE_VERSION, build_e_features

MODEL_FILE = MODELS_DIR / "strategy_e_clean_final.txt"
META_FILE = MODELS_DIR / "strategy_e_clean_metadata.json"


class LiveStrategyE:
    """Live inference for Strategy E."""

    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.meta = {}
        self.loaded = False
        self.missing_features = None  # Filled on first predict()
        self._load()

    def _load(self):
        """Load model and metadata."""
        if not (META_FILE.exists() and MODEL_FILE.exists()):
            print("  WARNING: Strategy E model not found. E will be disabled.")
            return

        with open(META_FILE) as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["feature_cols"]
        self.model = lgb.Booster(model_file=str(MODEL_FILE))
        self.loaded = True

        version = self.meta.get("pipeline_version", 1)
        through = str(self.meta.get("trained_through", "unknown"))[:10]
        print(f"  Strategy E loaded: {MODEL_FILE.name} ({len(self.feature_cols)} features, "
              f"pipeline v{version}, data through {through})")
        if version != PIPELINE_VERSION:
            print(f"  WARNING: Strategy E model was built with feature pipeline v{version} "
                  f"(current v{PIPELINE_VERSION}); retrain it: python src/retrain_e.py --force")

    def reload(self):
        """Pick up a freshly retrained model."""
        self.model, self.feature_cols, self.meta = None, None, {}
        self.loaded, self.missing_features = False, None
        self._load()

    def build_features(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
        """Raw closed M5/M15 bars → model input frame (same pipeline as training)."""
        return build_e_features(df_m5, df_m15, quiet=True)

    def predict(self, df: pd.DataFrame, idx: int) -> float:
        """
        Get E's raw probability for the bar at idx (a frame from build_features()).
        Returns probability of BUY (>0.5 = buy, <0.5 = sell), 0.5 if not loaded.
        """
        if not self.loaded:
            return 0.5

        if self.missing_features is None:
            self.missing_features = [c for c in self.feature_cols if c not in df.columns]
            if self.missing_features:
                print(f"  WARNING: Strategy E: {len(self.missing_features)}/{len(self.feature_cols)} "
                      f"model features are not built live and will be NaN: "
                      f"{', '.join(self.missing_features[:8])}...")

        row = df.iloc[idx]
        X = np.array([[pd.to_numeric(row.get(col, np.nan), errors="coerce")
                       for col in self.feature_cols]], dtype=float)
        return float(self.model.predict(X)[0])
