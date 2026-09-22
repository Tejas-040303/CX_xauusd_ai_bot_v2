"""
Project MIDAS v2 — Strategy E: AI Signal Generator
====================================================
LightGBM-based strategy that operates in two modes:

  1. INDEPENDENT MODE — Generates its own BUY/SELL signals on every M5 bar.
     Same interface as Strategies A-D: direction + confidence + SL/TP.

  2. VALIDATOR MODE — Evaluates a signal from another strategy and returns
     an agreement score + recommended lot adjustment.

Usage:
    from strategy_e import StrategyE

    # Initialize (loads trained model)
    strat_e = StrategyE()

    # Mode 1: Independent signal
    signal = strat_e.generate_signal(features_dict)
    # → {"direction": "BUY", "confidence": 72.3, "sl_pips": 20, "tp_pips": 30}

    # Mode 2: Validate another strategy's signal
    validation = strat_e.validate_signal(features_dict, other_direction="BUY", other_confidence=70)
    # → {"agree": True, "ai_confidence": 65.2, "lot_multiplier": 1.0, "action": "EXECUTE"}
"""

import json
import numpy as np
import lightgbm as lgb
from pathlib import Path

from config import (
    MODELS_DIR, ATR_PERIOD, PIP_VALUE,
    LOT_SIZE_MIN, LOT_SIZE_MAX, LOT_SIZE_SINGLE, LOT_SIZE_MULTI,
    CONFIDENCE_THRESHOLD, MAX_TRADE_RISK,
)


# ─── Strategy E Configuration ────────────────────────────

# Confidence thresholds for independent signal generation
INDEPENDENT_THRESHOLD = 0.65     # Min confidence to fire independently (stricter)
VALIDATOR_THRESHOLD = 0.52       # Min confidence for validator agreement (more lenient)

# Lot size multipliers based on AI-strategy agreement
LOT_MULTIPLIERS = {
    "strong_agree":   1.5,   # AI high conf + strategy signal → increase lot
    "agree":          1.0,   # AI agrees → normal lot
    "weak_agree":     0.7,   # AI weakly agrees → reduce lot
    "conflict":       0.0,   # AI disagrees → skip trade
}

# SL/TP configuration (rule-based, same as strategies A-D)
SL_ATR_MULTIPLIER = 1.5     # SL = 1.5x ATR
TP_RR_BASE = 1.5            # Base risk:reward ratio
TP_RR_HIGH_CONF = 2.5       # Extended TP when AI is very confident


class StrategyE:
    """
    AI-powered trading strategy using LightGBM.

    Dual role:
      - Independent signal generator (fires on its own)
      - Signal validator (confirms/adjusts other strategies' signals)
    """

    def __init__(self, model_path: str = None, metadata_path: str = None):
        """
        Load the trained LightGBM model and metadata.

        Args:
            model_path: Path to .txt model file. Default: models/strategy_e_final.txt
            metadata_path: Path to metadata JSON. Default: models/strategy_e_metadata.json
        """
        model_path = model_path or str(MODELS_DIR / "strategy_e_final.txt")
        metadata_path = metadata_path or str(MODELS_DIR / "strategy_e_metadata.json")

        # Load model
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Strategy E model not found at {model_path}. "
                "Run strategy_e_train.py first."
            )

        self.model = lgb.Booster(model_file=model_path)
        print(f"  Strategy E model loaded: {Path(model_path).name}")

        # Load metadata (feature list, thresholds, etc.)
        if Path(metadata_path).exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self.feature_cols = self.metadata["feature_cols"]
            self.optimal_threshold = self.metadata.get("confidence_threshold", 0.55)
            print(f"  Features: {len(self.feature_cols)}, "
                  f"Threshold: {self.optimal_threshold:.3f}")
        else:
            print(f"  WARNING: Metadata not found at {metadata_path}. "
                  "Using defaults.")
            self.metadata = {}
            self.feature_cols = None
            self.optimal_threshold = 0.55

        # Track recent predictions for monitoring
        self._recent_predictions = []

    def _prepare_features(self, features: dict) -> np.ndarray:
        """
        Convert a feature dictionary to the ordered array expected by LightGBM.

        Args:
            features: Dict mapping feature names to values.
                      Can be a single row or already-ordered array.

        Returns:
            2D numpy array (1 x n_features) ready for prediction.
        """
        if isinstance(features, np.ndarray):
            return features.reshape(1, -1) if features.ndim == 1 else features

        if self.feature_cols is None:
            raise ValueError("Feature column list not available. Load metadata.")

        # Build ordered feature vector
        feature_values = []
        for col in self.feature_cols:
            val = features.get(col, np.nan)  # NaN for missing (LightGBM handles this)
            feature_values.append(val)

        return np.array(feature_values).reshape(1, -1)

    def predict_probability(self, features: dict) -> float:
        """
        Get raw BUY probability from the model.

        Args:
            features: Feature dictionary for current M5 bar.

        Returns:
            Float in [0, 1]. Above 0.5 = leans BUY, below 0.5 = leans SELL.
        """
        X = self._prepare_features(features)
        prob = self.model.predict(X)[0]
        return float(prob)

    def generate_signal(self, features: dict, atr: float = None) -> dict:
        """
        INDEPENDENT MODE — Generate a trade signal.
        Called on every M5 bar to check if Strategy E sees an opportunity.

        Args:
            features: Feature dictionary for current M5 bar.
            atr: Current ATR value (for SL/TP calculation).
                 If None, attempts to read from features dict.

        Returns:
            dict with keys:
              - direction: "BUY", "SELL", or "WAIT"
              - confidence: 0-100 (mapped from model probability)
              - sl_pips: Stop loss in pips (0 if WAIT)
              - tp_pips: Take profit in pips (0 if WAIT)
              - raw_probability: Model's raw output [0, 1]
              - strategy: "E"
        """
        prob = self.predict_probability(features)

        # Map probability to direction + confidence
        if prob >= self.optimal_threshold:
            direction = "BUY"
            confidence = self._prob_to_confidence(prob, is_buy=True)
        elif prob <= (1 - self.optimal_threshold):
            direction = "SELL"
            confidence = self._prob_to_confidence(prob, is_buy=False)
        else:
            direction = "WAIT"
            confidence = 0.0

        # Only fire independently if confidence is high enough
        if direction != "WAIT" and confidence < (INDEPENDENT_THRESHOLD * 100):
            direction = "WAIT"
            confidence = 0.0

        # Calculate SL/TP
        if atr is None:
            atr_col = f"atr_{ATR_PERIOD}"
            atr = features.get(atr_col, 2.0)  # Fallback: $2 ATR

        sl_pips, tp_pips = self._calculate_sl_tp(atr, confidence)

        if direction == "WAIT":
            sl_pips = 0
            tp_pips = 0

        signal = {
            "direction": direction,
            "confidence": round(confidence, 1),
            "sl_pips": round(sl_pips, 1),
            "tp_pips": round(tp_pips, 1),
            "raw_probability": round(prob, 4),
            "strategy": "E",
        }

        # Track for monitoring
        self._recent_predictions.append(prob)
        if len(self._recent_predictions) > 1000:
            self._recent_predictions = self._recent_predictions[-500:]

        return signal

    def validate_signal(
        self,
        features: dict,
        other_direction: str,
        other_confidence: float,
        other_strategy: str = "unknown",
    ) -> dict:
        """
        VALIDATOR MODE — Evaluate another strategy's signal.
        Called when Strategy A/B/C/D fires, to confirm or adjust the trade.

        Args:
            features: Feature dictionary for the M5 bar where the signal fired.
            other_direction: "BUY" or "SELL" from the other strategy.
            other_confidence: 0-100 confidence from the other strategy.
            other_strategy: Name of the firing strategy (for logging).

        Returns:
            dict with keys:
              - agree: bool — does AI agree with the direction?
              - ai_direction: "BUY" or "SELL" — what AI thinks
              - ai_confidence: 0-100 — AI's own confidence
              - lot_multiplier: float — multiply the base lot by this
              - action: "EXECUTE", "REDUCE", "SKIP"
              - reason: human-readable explanation
        """
        prob = self.predict_probability(features)

        # Determine AI's own opinion
        if prob >= 0.5:
            ai_direction = "BUY"
            ai_confidence = self._prob_to_confidence(prob, is_buy=True)
        else:
            ai_direction = "SELL"
            ai_confidence = self._prob_to_confidence(prob, is_buy=False)

        # Check agreement
        agree = (ai_direction == other_direction)

        # Determine action + lot multiplier
        if agree:
            if ai_confidence >= 70:
                action = "EXECUTE"
                lot_multiplier = LOT_MULTIPLIERS["strong_agree"]
                reason = (f"AI strongly confirms {other_strategy}'s {other_direction} "
                          f"(AI conf: {ai_confidence:.0f}%)")
            elif ai_confidence >= 55:
                action = "EXECUTE"
                lot_multiplier = LOT_MULTIPLIERS["agree"]
                reason = (f"AI confirms {other_strategy}'s {other_direction} "
                          f"(AI conf: {ai_confidence:.0f}%)")
            else:
                action = "REDUCE"
                lot_multiplier = LOT_MULTIPLIERS["weak_agree"]
                reason = (f"AI weakly agrees with {other_strategy}'s {other_direction} "
                          f"(AI conf: {ai_confidence:.0f}%) — reducing lot")
        else:
            # Conflict — AI sees opposite direction
            if ai_confidence >= 65:
                action = "SKIP"
                lot_multiplier = LOT_MULTIPLIERS["conflict"]
                reason = (f"AI CONFLICTS with {other_strategy}: "
                          f"strategy says {other_direction}, AI says {ai_direction} "
                          f"(AI conf: {ai_confidence:.0f}%) — SKIP trade")
            else:
                # AI disagrees but not confident — let strategy through with caution
                action = "REDUCE"
                lot_multiplier = LOT_MULTIPLIERS["weak_agree"]
                reason = (f"AI slightly disagrees with {other_strategy}'s {other_direction} "
                          f"(AI sees {ai_direction} at {ai_confidence:.0f}%) — reducing lot")

        return {
            "agree": agree,
            "ai_direction": ai_direction,
            "ai_confidence": round(ai_confidence, 1),
            "lot_multiplier": round(lot_multiplier, 2),
            "action": action,
            "reason": reason,
            "raw_probability": round(prob, 4),
        }

    def _prob_to_confidence(self, prob: float, is_buy: bool) -> float:
        """
        Map model probability [0, 1] to confidence [0, 100].

        The mapping is non-linear to reward high-conviction predictions:
          prob=0.50 → confidence=0
          prob=0.55 → confidence=~40
          prob=0.60 → confidence=~60
          prob=0.65 → confidence=~75
          prob=0.70 → confidence=~85
          prob=0.80 → confidence=~95

        For SELL, we use (1 - prob) as the effective probability.
        """
        if is_buy:
            effective_prob = prob
        else:
            effective_prob = 1 - prob

        # Clip to [0.5, 1.0] range
        effective_prob = max(effective_prob, 0.5)

        # Non-linear mapping: confidence = 100 * (2 * (p - 0.5)) ^ 0.7
        # This gives a gentle curve that's easier to reach medium confidence
        # but harder to reach very high confidence
        raw_conf = (2 * (effective_prob - 0.5)) ** 0.7
        confidence = min(100.0, raw_conf * 100.0)

        return confidence

    def _calculate_sl_tp(self, atr: float, confidence: float) -> tuple:
        """
        Calculate SL and TP in pips based on ATR and confidence.

        SL is always rule-based: 1.5x ATR (same as all strategies).
        TP scales with confidence:
          - Base: 1.5x RR
          - High confidence (>75%): 2.5x RR
        """
        sl_price = SL_ATR_MULTIPLIER * atr
        sl_pips = sl_price / PIP_VALUE

        # Cap SL to keep within risk limits
        max_sl_pips = MAX_TRADE_RISK / (LOT_SIZE_SINGLE * 100 * PIP_VALUE)
        sl_pips = min(sl_pips, max_sl_pips)

        # TP based on RR ratio
        if confidence >= 75:
            rr = TP_RR_HIGH_CONF
        else:
            rr = TP_RR_BASE

        tp_pips = sl_pips * rr

        return sl_pips, tp_pips

    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about recent predictions.
        Useful for monitoring Strategy E's behavior in live trading.
        """
        if not self._recent_predictions:
            return {"status": "no predictions yet"}

        preds = np.array(self._recent_predictions)
        return {
            "n_predictions": len(preds),
            "mean_probability": round(float(np.mean(preds)), 4),
            "std_probability": round(float(np.std(preds)), 4),
            "pct_above_threshold": round(
                100 * (preds >= self.optimal_threshold).mean(), 1
            ),
            "pct_below_threshold": round(
                100 * (preds <= 1 - self.optimal_threshold).mean(), 1
            ),
            "pct_wait": round(
                100 * ((preds > 1 - self.optimal_threshold) &
                       (preds < self.optimal_threshold)).mean(), 1
            ),
            "last_5_probs": [round(p, 4) for p in preds[-5:].tolist()],
        }


# ─── Standalone Testing ──────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Strategy E — Quick Test")
    print("=" * 60)

    try:
        strat_e = StrategyE()
    except FileNotFoundError as e:
        print(f"\n  {e}")
        print("  Train the model first: python strategy_e_train.py")
        exit(1)

    # Test with dummy features (all NaN — LightGBM will handle)
    dummy_features = {col: np.nan for col in strat_e.feature_cols}

    print("\n  Testing independent signal generation...")
    signal = strat_e.generate_signal(dummy_features, atr=2.0)
    print(f"    Signal: {signal}")

    print("\n  Testing validator mode...")
    validation = strat_e.validate_signal(
        dummy_features,
        other_direction="BUY",
        other_confidence=70,
        other_strategy="A",
    )
    print(f"    Validation: {validation}")

    print("\n  Diagnostics:")
    diag = strat_e.get_diagnostics()
    for k, v in diag.items():
        print(f"    {k}: {v}")

    print("\n✅ Strategy E class working!")