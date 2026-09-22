"""
Project MIDAS v2 — Phase 9: Regime Detector
=============================================
Online regime classification using exponential moving statistics.
No batch training needed. Runs on laptop CPU.

Regimes:
  TRENDING_BULL  — ADX strong + bullish EMA stack + above EMA200
  TRENDING_BEAR  — ADX strong + bearish EMA stack + below EMA200
  VOLATILE       — High ATR percentile + no directional clarity
  QUIET          — Low ATR percentile + weak ADX
  RANGING        — Everything else (normal sideways)

Each regime has strategy adjustments:
  - E independent threshold (higher = fewer signals)
  - E confidence boost/penalty
  - SL multiplier (wider in volatile, tighter in quiet)
  - Whether E independent is allowed

Usage:
    from regime_detector import RegimeDetector, REGIME_ADJUSTMENTS

    detector = RegimeDetector()
    regime = detector.detect(row)  # row = dict or Series
    adjustments = REGIME_ADJUSTMENTS[regime]
"""

import numpy as np
import pandas as pd
from config import ATR_PERIOD


# ─── Regime Definitions ──────────────────────────────────

REGIMES = ["TRENDING_BULL", "TRENDING_BEAR", "VOLATILE", "QUIET", "RANGING"]

# Strategy adjustments per regime
# These control how Strategy E behaves in each market condition
REGIME_ADJUSTMENTS = {
    "TRENDING_BULL": {
        "e_indep_threshold": 0.65,    # Lower → more signals (trend is our friend)
        "e_conf_boost": 5,            # Slight confidence boost
        "sl_mult": 1.0,               # Normal SL
        "tp_mult": 1.2,               # Slightly extended TP (trends run)
        "allow_e_independent": True,
        "e_val_strong_agree": 0.57,   # Easier to agree in trend
        "e_val_agree": 0.52,
        "e_val_conflict_skip": 0.62,
    },
    "TRENDING_BEAR": {
        "e_indep_threshold": 0.65,
        "e_conf_boost": 5,
        "sl_mult": 1.0,
        "tp_mult": 1.2,
        "allow_e_independent": True,
        "e_val_strong_agree": 0.57,
        "e_val_agree": 0.52,
        "e_val_conflict_skip": 0.62,
    },
    "VOLATILE": {
        "e_indep_threshold": 0.72,    # Higher → fewer signals (choppy = dangerous)
        "e_conf_boost": -5,           # Reduce confidence
        "sl_mult": 1.3,               # Wider SL (bigger swings)
        "tp_mult": 1.0,               # Normal TP
        "allow_e_independent": True,   # Still allow but stricter
        "e_val_strong_agree": 0.63,   # Harder to agree in volatility
        "e_val_agree": 0.55,
        "e_val_conflict_skip": 0.58,
    },
    "QUIET": {
        "e_indep_threshold": 0.75,    # Very strict (not enough movement)
        "e_conf_boost": -10,          # Big penalty
        "sl_mult": 0.8,               # Tighter SL (small range)
        "tp_mult": 0.8,               # Tighter TP
        "allow_e_independent": False,  # Disable E independent in quiet markets
        "e_val_strong_agree": 0.60,
        "e_val_agree": 0.52,
        "e_val_conflict_skip": 0.60,
    },
    "RANGING": {
        "e_indep_threshold": 0.70,    # Slightly strict (no trend to ride)
        "e_conf_boost": 0,            # Neutral
        "sl_mult": 1.0,               # Normal SL
        "tp_mult": 0.9,               # Slightly tighter TP (mean-reverting)
        "allow_e_independent": True,
        "e_val_strong_agree": 0.60,
        "e_val_agree": 0.52,
        "e_val_conflict_skip": 0.60,
    },
}


class RegimeDetector:
    """
    Online regime detector.
    Classifies each bar into one of 5 market regimes using
    features already computed by the feature engine.

    No look-ahead: uses only current bar's indicator values.
    No training needed: thresholds derived from indicator definitions.
    """

    def __init__(self,
                 adx_trend_threshold: float = 25,
                 adx_weak_threshold: float = 20,
                 atr_high_pct: float = 0.75,
                 atr_low_pct: float = 0.25):
        """
        Args:
            adx_trend_threshold: ADX above this = trending
            adx_weak_threshold: ADX below this = weak/quiet
            atr_high_pct: ATR percentile above this = volatile
            atr_low_pct: ATR percentile below this = quiet
        """
        self.adx_trend = adx_trend_threshold
        self.adx_weak = adx_weak_threshold
        self.atr_high = atr_high_pct
        self.atr_low = atr_low_pct

        # Online tracking
        self._regime_history = []
        self._regime_counts = {r: 0 for r in REGIMES}

    def detect(self, row) -> str:
        """
        Classify the current bar's regime.

        Args:
            row: dict or Series with features:
                 adx_14, atr_percentile, ema_stack, above_ema200,
                 trend_strong, ema_partial_bull, ema_partial_bear

        Returns:
            One of: TRENDING_BULL, TRENDING_BEAR, VOLATILE, QUIET, RANGING
        """
        # Extract features (handle dict or Series)
        adx = self._get(row, f"adx_{ATR_PERIOD}", 20)
        atr_pct = self._get(row, "atr_percentile", 0.5)
        ema_stack = self._get(row, "ema_stack", 0)
        above_ema200 = self._get(row, "above_ema200", 0)
        partial_bull = self._get(row, "ema_partial_bull", 0)
        partial_bear = self._get(row, "ema_partial_bear", 0)

        # Handle NaN
        if pd.isna(adx):
            adx = 20
        if pd.isna(atr_pct):
            atr_pct = 0.5

        # ── Classification logic ──────────────────────────

        # VOLATILE: high ATR + no clear direction
        if atr_pct > self.atr_high and adx < self.adx_trend:
            regime = "VOLATILE"

        # QUIET: low ATR + weak ADX
        elif atr_pct < self.atr_low and adx < self.adx_weak:
            regime = "QUIET"

        # TRENDING_BULL: strong ADX + bullish alignment
        elif adx >= self.adx_trend and (ema_stack == 1 or (partial_bull == 1 and above_ema200 == 1)):
            regime = "TRENDING_BULL"

        # TRENDING_BEAR: strong ADX + bearish alignment
        elif adx >= self.adx_trend and (ema_stack == -1 or (partial_bear == 1 and above_ema200 == 0)):
            regime = "TRENDING_BEAR"

        # RANGING: everything else
        else:
            regime = "RANGING"

        # Track
        self._regime_history.append(regime)
        self._regime_counts[regime] += 1

        return regime

    def get_adjustments(self, regime: str) -> dict:
        """Get strategy adjustments for a given regime."""
        return REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS["RANGING"])

    def detect_and_adjust(self, row) -> tuple:
        """Convenience: detect regime and return (regime, adjustments)."""
        regime = self.detect(row)
        return regime, self.get_adjustments(regime)

    def get_stats(self) -> dict:
        """Get regime distribution statistics."""
        total = sum(self._regime_counts.values())
        if total == 0:
            return {}

        return {
            regime: {
                "count": count,
                "pct": round(count / total * 100, 1),
            }
            for regime, count in self._regime_counts.items()
            if count > 0
        }

    def _get(self, row, key, default=0):
        """Safe get from dict or Series."""
        if isinstance(row, dict):
            return row.get(key, default)
        else:
            val = row.get(key, default) if hasattr(row, 'get') else getattr(row, key, default)
            return val if not pd.isna(val) else default

    def print_stats(self):
        """Print regime distribution."""
        stats = self.get_stats()
        total = sum(self._regime_counts.values())
        print(f"\n  REGIME DISTRIBUTION ({total:,} bars):")
        for regime in REGIMES:
            if regime in stats:
                s = stats[regime]
                bar = "█" * int(s["pct"] / 2)
                print(f"    {regime:<16s}: {s['count']:>7,} ({s['pct']:>5.1f}%) {bar}")


def precompute_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Precompute regime labels for entire DataFrame.
    Returns a Series of regime strings aligned with df index.
    """
    detector = RegimeDetector()
    regimes = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        regime = detector.detect(row)
        regimes.append(regime)

    detector.print_stats()
    return pd.Series(regimes, index=df.index, name="regime")


# ─── Quick Test ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 9 — Regime Detector Test")
    print("=" * 60)

    # Test with dummy data
    detector = RegimeDetector()

    test_cases = [
        {"adx_14": 35, "atr_percentile": 0.6, "ema_stack": 1,
         "above_ema200": 1, "ema_partial_bull": 1, "ema_partial_bear": 0},
        {"adx_14": 35, "atr_percentile": 0.6, "ema_stack": -1,
         "above_ema200": 0, "ema_partial_bull": 0, "ema_partial_bear": 1},
        {"adx_14": 15, "atr_percentile": 0.85, "ema_stack": 0,
         "above_ema200": 1, "ema_partial_bull": 0, "ema_partial_bear": 0},
        {"adx_14": 15, "atr_percentile": 0.15, "ema_stack": 0,
         "above_ema200": 1, "ema_partial_bull": 0, "ema_partial_bear": 0},
        {"adx_14": 22, "atr_percentile": 0.5, "ema_stack": 0,
         "above_ema200": 1, "ema_partial_bull": 0, "ema_partial_bear": 0},
    ]

    for i, tc in enumerate(test_cases):
        regime = detector.detect(tc)
        adj = detector.get_adjustments(regime)
        print(f"\n  Test {i+1}: ADX={tc['adx_14']}, ATR%={tc['atr_percentile']}, "
              f"Stack={tc['ema_stack']}")
        print(f"    Regime: {regime}")
        print(f"    E threshold: {adj['e_indep_threshold']}, "
              f"E allowed: {adj['allow_e_independent']}, "
              f"SL mult: {adj['sl_mult']}")

    print("\n✅ Regime detector working!")