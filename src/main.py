# # # """
# # # Project MIDAS v2 — Main Live Trading Loop
# # # ==========================================
# # # Connects all components and runs the A+B+C+D+E+Regime strategy live on MT5.

# # # Architecture per M5 bar:
# # #   1. Fetch latest candles (M5 + M15) from MT5
# # #   2. Compute 167 features (FeatureEngine)
# # #   3. Detect market regime (RegimeDetector)
# # #   4. Run strategies A, B, C, D → collect signals
# # #   5. Run DecisionEngine (A-D signals + Strategy E validation + E independent)
# # #   6. Apply regime adjustments to E thresholds
# # #   7. RiskEngine final check
# # #   8. Execute via MT5 (Executor)
# # #   9. Monitor for closed trades → update daily P&L
# # #   10. Log + Telegram notify

# # # Usage:
# # #     python main.py               # Live trading
# # #     python main.py --dry-run     # Dry run (signals only, no orders)
# # #     python main.py --check       # Check MT5 connection and exit
# # # """

# # # import argparse
# # # import sys
# # # import time
# # # import json
# # # import traceback
# # # from datetime import datetime, date
# # # from pathlib import Path

# # # import numpy as np
# # # import pandas as pd
# # # import lightgbm as lgb

# # # sys.path.insert(0, str(Path(__file__).parent))

# # # from config import (
# # #     MT5_SYMBOL, MODELS_DIR, DATA_PROCESSED,
# # #     ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE,
# # #     LOT_SIZE_MIN, LOT_SIZE_MAX, LOT_SIZE_SINGLE,
# # #     SESSIONS,
# # # )
# # # from feature_engineering import FeatureEngine
# # # from strategies.strategy_a import StrategyA
# # # from strategies.strategy_b import StrategyB
# # # from strategies.strategy_c import StrategyC
# # # from strategies.strategy_d import StrategyD
# # # from regime_detector import RegimeDetector, REGIME_ADJUSTMENTS
# # # from decision_engine import DecisionEngine

# # # import logger
# # # import notifier
# # # from daily_limits import DailyLimits
# # # from risk_engine import RiskEngine
# # # from executor import Executor


# # # # ─── Strategy E Config (debiased best threshold) ─────────
# # # E_PROB_THRESHOLD    = 0.68
# # # E_VAL_STRONG_AGREE  = 0.60
# # # E_VAL_AGREE         = 0.52
# # # E_VAL_CONFLICT_SKIP = 0.60
# # # E_STRONG_BOOST      = 10
# # # E_WEAK_PENALTY      = -10
# # # E_SL_ATR_MULT       = 1.5
# # # E_TP_RR_BASE        = 1.5
# # # E_TP_RR_HIGH        = 2.5
# # # E_INDEPENDENT_MIN_CONF = 70

# # # # Walk-forward model segments (use final model for live trading)
# # # E_MODEL_PATH    = MODELS_DIR / "strategy_e_clean_final.txt"
# # # E_METADATA_PATH = MODELS_DIR / "strategy_e_clean_metadata.json"

# # # # How many candles to fetch for feature engineering
# # # M5_LOOKBACK  = 500    # Enough for EMA200 + all indicators
# # # M15_LOOKBACK = 200


# # # # ─── Session Detection ───────────────────────────────────

# # # def get_current_session(dt: datetime) -> str:
# # #     """Determine current trading session from UTC hour."""
# # #     hour = dt.hour
# # #     # london_ny_overlap: 12:00 - 14:00 UTC
# # #     if 12 <= hour < 14:
# # #         return "london_ny_overlap"
# # #     # london: 06:00 - 14:00 UTC
# # #     if 6 <= hour < 14:
# # #         return "london"
# # #     # new_york: 14:00 - 20:00 UTC (adjusted from config for overlap)
# # #     if 14 <= hour < 20:
# # #         return "new_york"
# # #     # asian: 00:00 - 06:00 UTC
# # #     if 0 <= hour < 6:
# # #         return "asian"
# # #     return "off"


# # # # ─── Feature Preparation ─────────────────────────────────

# # # def prepare_features(executor: Executor, feature_engine: FeatureEngine) -> tuple:
# # #     """
# # #     Fetch latest candles from MT5 and compute features.
# # #     Returns (df_featured: pd.DataFrame, current_row: pd.Series, current_idx: int)
# # #     or (None, None, None) on error.
# # #     """
# # #     df_m5 = executor.get_candles("M5", M5_LOOKBACK)
# # #     df_m15 = executor.get_candles("M15", M15_LOOKBACK)

# # #     if df_m5 is None or df_m15 is None:
# # #         logger.error("Failed to fetch candles from MT5")
# # #         return None, None, None

# # #     if len(df_m5) < 250:
# # #         logger.warning(f"Not enough M5 candles: {len(df_m5)} (need 250+)")
# # #         return None, None, None

# # #     try:
# # #         df = feature_engine.compute(df_m5, df_m15)
# # #         if df is None or df.empty:
# # #             logger.error("FeatureEngine returned empty dataframe")
# # #             return None, None, None

# # #         idx = len(df) - 1
# # #         current_row = df.iloc[idx]
# # #         return df, current_row, idx

# # #     except Exception as e:
# # #         logger.log_error("prepare_features", e)
# # #         return None, None, None


# # # # ─── Strategy E Inference ────────────────────────────────

# # # class StrategyELive:
# # #     """Thin wrapper around the trained LightGBM model for live inference."""

# # #     def __init__(self):
# # #         if not E_MODEL_PATH.exists():
# # #             raise FileNotFoundError(
# # #                 f"Strategy E model not found: {E_MODEL_PATH}\n"
# # #                 f"Run strategy_e_clean_retrain.py first."
# # #             )
# # #         self.model = lgb.Booster(model_file=str(E_MODEL_PATH))

# # #         with open(E_METADATA_PATH) as f:
# # #             meta = json.load(f)
# # #         self.feature_cols = meta["feature_cols"]

# # #     def predict(self, row: pd.Series) -> float:
# # #         """Return raw probability (0-1) for BUY direction."""
# # #         try:
# # #             feat = row[self.feature_cols].values.reshape(1, -1)
# # #             prob = float(self.model.predict(feat)[0])
# # #             return prob
# # #         except Exception as e:
# # #             logger.log_error("StrategyELive.predict", e)
# # #             return 0.5  # Neutral on error


# # # # ─── Main Signal Processing ──────────────────────────────

# # # def process_bar(
# # #     df: pd.DataFrame,
# # #     idx: int,
# # #     current_row: pd.Series,
# # #     strategies: dict,
# # #     strategy_e: StrategyELive,
# # #     regime_detector: RegimeDetector,
# # #     daily_limits: DailyLimits,
# # #     risk_engine: RiskEngine,
# # #     executor: Executor,
# # #     dry_run: bool = False,
# # # ) -> dict:
# # #     """
# # #     Process one M5 bar. Returns a result dict describing what happened.
# # #     """
# # #     now = datetime.now(UTC)
# # #     session = get_current_session(now)
# # #     bar_time = current_row.get("datetime", now)

# # #     result = {
# # #         "bar_time": str(bar_time),
# # #         "session": session,
# # #         "action": "SKIP",
# # #         "reason": "",
# # #         "ticket": None,
# # #     }

# # #     # ── Daily rollover check (handled in main loop) ───────
# # #     daily_limits.check_day_rollover()

# # #     # ── Pre-flight: can we trade at all? ──────────────────
# # #     can_trade, block_reason = daily_limits.can_trade(session, now)
# # #     if not can_trade:
# # #         result["reason"] = block_reason
# # #         logger.debug(f"Skipping bar — {block_reason}")
# # #         return result

# # #     # ── Run strategies A-D ────────────────────────────────
# # #     ad_signals = []
# # #     for name, strat in strategies.items():
# # #         try:
# # #             signal = strat.generate_signal(df, idx)
# # #             if signal is not None:
# # #                 ad_signals.append({
# # #                     "strategy": name,
# # #                     "direction": signal.direction.upper(),
# # #                     "confidence": signal.confidence,
# # #                     "sl_price": signal.sl_price,
# # #                     "tp_price": signal.tp_price,
# # #                     "entry_price": signal.entry_price,
# # #                 })
# # #         except Exception as e:
# # #             logger.log_error(f"strategy_{name}", e)

# # #     # ── Regime detection ─────────────────────────────────
# # #     regime = "RANGING"
# # #     adj = REGIME_ADJUSTMENTS.get("RANGING", {})
# # #     try:
# # #         regime, adj = regime_detector.detect_and_adjust(current_row)
# # #     except Exception as e:
# # #         logger.log_error("regime_detector", e)

# # #     # ── Strategy E prediction ─────────────────────────────
# # #     e_prob = 0.5
# # #     try:
# # #         e_prob = strategy_e.predict(current_row)
# # #     except Exception as e:
# # #         logger.log_error("strategy_e.predict", e)

# # #     # ── Resolve direction and build trade decision ─────────
# # #     # Uses same logic as run_phase9_regime.py
# # #     decision = _resolve_decision(
# # #         ad_signals, e_prob, adj, regime, df, idx, current_row, session
# # #     )

# # #     if decision["action"] == "SKIP":
# # #         result["reason"] = decision["reason"]
# # #         log_str = f"[{bar_time}] {session} | regime={regime} | SKIP: {decision['reason']}"
# # #         logger.debug(log_str)
# # #         return result

# # #     # ── Log the signal ────────────────────────────────────
# # #     logger.log_signal(
# # #         bar_time=bar_time,
# # #         strategy=decision["source"],
# # #         direction=decision["action"],
# # #         confidence=decision.get("confidence", 0),
# # #         regime=regime,
# # #         action_taken="EXECUTE" if not dry_run else "DRY_RUN",
# # #         reason=decision["reason"],
# # #     )

# # #     if dry_run:
# # #         result.update({"action": decision["action"], "reason": "DRY_RUN — no order sent"})
# # #         logger.info(
# # #             f"[DRY RUN] {decision['action']} | lot={decision['lot']} | "
# # #             f"source={decision['source']} | {decision['reason']}"
# # #         )
# # #         return result

# # #     # ── Risk engine final check ────────────────────────────
# # #     account = executor.get_account_info()
# # #     approved, risk_msg, final_lot = risk_engine.check_order(
# # #         direction=decision["action"],
# # #         lot=decision["lot"],
# # #         sl_pips=decision["sl_pips"],
# # #         daily_pnl=daily_limits.daily_pnl,
# # #         account_balance=account.get("balance", 5000),
# # #         account_equity=account.get("equity", 5000),
# # #     )

# # #     if not approved:
# # #         logger.log_risk_block(risk_msg, decision["source"])
# # #         notifier.notify_risk_block(risk_msg)
# # #         result["reason"] = f"RISK BLOCK: {risk_msg}"
# # #         return result

# # #     # Validate SL/TP sides
# # #     sl_valid, sl_msg = risk_engine.validate_sl_tp(
# # #         decision["action"],
# # #         decision["entry_price"],
# # #         decision["sl_price"],
# # #         decision["tp_price"],
# # #     )
# # #     if not sl_valid:
# # #         logger.log_risk_block(sl_msg, decision["source"])
# # #         result["reason"] = f"SL/TP INVALID: {sl_msg}"
# # #         return result

# # #     # ── Place the order ────────────────────────────────────
# # #     success, ticket, msg = executor.place_order(
# # #         direction=decision["action"],
# # #         lot=final_lot,
# # #         sl_price=decision["sl_price"],
# # #         tp_price=decision["tp_price"],
# # #         comment=f"MIDAS_{decision['source'][:10]}",
# # #     )

# # #     if not success:
# # #         logger.error(f"Order failed: {msg}")
# # #         notifier.notify_error("place_order", msg)
# # #         result["reason"] = f"ORDER FAILED: {msg}"
# # #         return result

# # #     # ── Record the trade ──────────────────────────────────
# # #     daily_limits.record_trade_opened(ticket, session, now)

# # #     logger.log_trade_opened(
# # #         ticket=ticket,
# # #         direction=decision["action"],
# # #         lot=final_lot,
# # #         entry_price=decision["entry_price"],
# # #         sl_price=decision["sl_price"],
# # #         tp_price=decision["tp_price"],
# # #         source=decision["source"],
# # #         regime=regime,
# # #         reason=decision["reason"],
# # #     )

# # #     notifier.notify_trade_opened(
# # #         ticket=ticket,
# # #         direction=decision["action"],
# # #         lot=final_lot,
# # #         entry_price=decision["entry_price"],
# # #         sl_price=decision["sl_price"],
# # #         tp_price=decision["tp_price"],
# # #         source=decision["source"],
# # #         regime=regime,
# # #     )

# # #     result.update({
# # #         "action": decision["action"],
# # #         "ticket": ticket,
# # #         "lot": final_lot,
# # #         "reason": decision["reason"],
# # #     })
# # #     return result


# # # def _resolve_decision(
# # #     ad_signals: list, e_prob: float, adj: dict, regime: str,
# # #     df: pd.DataFrame, idx: int, row: pd.Series, session: str,
# # # ) -> dict:
# # #     """
# # #     Combine A-D signals + regime + E probability into a final trade decision.
# # #     Mirrors the logic from run_phase9_regime.py's RegimeAwareCombinedABCDE.
# # #     """
# # #     skip = lambda reason: {
# # #         "action": "SKIP", "reason": reason, "lot": 0,
# # #         "entry_price": 0, "sl_price": 0, "tp_price": 0,
# # #         "sl_pips": 0, "source": "none", "confidence": 0,
# # #     }

# # #     atr = float(row.get(f"atr_{ATR_PERIOD}", row.get("atr", 2.0)))

# # #     # ── Get regime-adjusted E thresholds ─────────────────
# # #     e_indep_thresh  = adj.get("e_indep_threshold", E_PROB_THRESHOLD)
# # #     e_conf_boost    = adj.get("e_conf_boost", 0)
# # #     sl_mult         = adj.get("sl_mult", 1.0)
# # #     tp_mult         = adj.get("tp_mult", 1.0)
# # #     allow_e_indep   = adj.get("allow_e_independent", True)
# # #     e_strong_thresh = adj.get("e_val_strong_agree", E_VAL_STRONG_AGREE)
# # #     e_agree_thresh  = adj.get("e_val_agree", E_VAL_AGREE)
# # #     e_conflict_skip = adj.get("e_val_conflict_skip", E_VAL_CONFLICT_SKIP)

# # #     current_price = float(row.get("close", 0))

# # #     # ── Case 1: A-D signals exist ────────────────────────
# # #     if ad_signals:
# # #         buy_sigs  = [s for s in ad_signals if s["direction"] == "BUY"]
# # #         sell_sigs = [s for s in ad_signals if s["direction"] == "SELL"]

# # #         # Resolve direction conflict
# # #         if buy_sigs and sell_sigs:
# # #             buy_conf  = sum(s["confidence"] for s in buy_sigs)
# # #             sell_conf = sum(s["confidence"] for s in sell_sigs)
# # #             if buy_conf > sell_conf:
# # #                 direction, primary = "BUY", buy_sigs
# # #             elif sell_conf > buy_conf:
# # #                 direction, primary = "SELL", sell_sigs
# # #             else:
# # #                 return skip("A-D conflict — equal confidence")
# # #         elif buy_sigs:
# # #             direction, primary = "BUY", buy_sigs
# # #         else:
# # #             direction, primary = "SELL", sell_sigs

# # #         best = max(primary, key=lambda s: s["confidence"])

# # #         # E validation
# # #         if direction == "BUY":
# # #             e_agree = e_prob >= e_agree_thresh
# # #             e_strong = e_prob >= e_strong_thresh
# # #             e_conflict = (1 - e_prob) >= e_conflict_skip
# # #         else:
# # #             e_agree = (1 - e_prob) >= e_agree_thresh
# # #             e_strong = (1 - e_prob) >= e_strong_thresh
# # #             e_conflict = e_prob >= e_conflict_skip

# # #         if e_conflict:
# # #             return skip(f"E conflicts with {direction} (prob={e_prob:.3f})")

# # #         # Lot sizing based on agreement count + E boost
# # #         n_agree = len(primary)
# # #         if n_agree >= 2:
# # #             lot = 0.02
# # #         else:
# # #             lot = LOT_SIZE_SINGLE

# # #         # Use SL/TP from best signal
# # #         sl_price = best["sl_price"]
# # #         tp_price = best["tp_price"]
# # #         entry    = best["entry_price"] if best["entry_price"] > 0 else current_price

# # #         # Apply regime SL/TP multipliers
# # #         if sl_mult != 1.0:
# # #             sl_dist = abs(entry - sl_price) * sl_mult
# # #             sl_price = entry - sl_dist if direction == "BUY" else entry + sl_dist

# # #         if tp_mult != 1.0:
# # #             tp_dist = abs(tp_price - entry) * tp_mult
# # #             tp_price = entry + tp_dist if direction == "BUY" else entry - tp_dist

# # #         sl_pips = abs(entry - sl_price) / PIP_VALUE
# # #         source  = "+".join(s["strategy"] for s in primary) + "+E_val"
# # #         conf    = best["confidence"] + (e_conf_boost if e_agree else 0)

# # #         return {
# # #             "action": direction,
# # #             "lot": lot,
# # #             "entry_price": round(entry, 2),
# # #             "sl_price": round(sl_price, 2),
# # #             "tp_price": round(tp_price, 2),
# # #             "sl_pips": round(sl_pips, 1),
# # #             "source": source,
# # #             "confidence": conf,
# # #             "reason": (
# # #                 f"{source} conf={conf:.0f}% e_prob={e_prob:.3f} "
# # #                 f"regime={regime} n={n_agree}"
# # #             ),
# # #         }

# # #     # ── Case 2: No A-D signal — E independent ────────────
# # #     if not allow_e_indep:
# # #         return skip(f"E independent blocked by regime={regime}")

# # #     # BUY if prob >= threshold, SELL if (1-prob) >= threshold
# # #     if e_prob >= e_indep_thresh:
# # #         direction = "BUY"
# # #         e_conf = e_prob
# # #     elif (1 - e_prob) >= e_indep_thresh:
# # #         direction = "SELL"
# # #         e_conf = 1 - e_prob
# # #     else:
# # #         return skip(f"E no signal (prob={e_prob:.3f} thresh={e_indep_thresh:.2f})")

# # #     # E independent: fixed SL/TP from ATR
# # #     sl_atr = atr * E_SL_ATR_MULT * sl_mult
# # #     tp_atr = sl_atr * E_TP_RR_HIGH if e_conf >= 0.75 else sl_atr * E_TP_RR_BASE
# # #     tp_atr *= tp_mult

# # #     sl_pips = sl_atr / PIP_VALUE
# # #     entry   = current_price

# # #     if direction == "BUY":
# # #         sl_price = entry - sl_atr
# # #         tp_price = entry + tp_atr
# # #     else:
# # #         sl_price = entry + sl_atr
# # #         tp_price = entry - tp_atr

# # #     return {
# # #         "action": direction,
# # #         "lot": LOT_SIZE_SINGLE,
# # #         "entry_price": round(entry, 2),
# # #         "sl_price": round(sl_price, 2),
# # #         "tp_price": round(tp_price, 2),
# # #         "sl_pips": round(sl_pips, 1),
# # #         "source": "E_independent",
# # #         "confidence": round(e_conf * 100, 1),
# # #         "reason": (
# # #             f"E_indep {direction} prob={e_prob:.3f} "
# # #             f"regime={regime} sl={sl_pips:.0f}pips"
# # #         ),
# # #     }


# # # # ─── Closed Trade Monitor ────────────────────────────────

# # # def check_closed_trades(
# # #     executor: Executor,
# # #     daily_limits: DailyLimits,
# # #     session_start_ts: int,
# # #     already_processed: set,
# # # ) -> set:
# # #     """
# # #     Check for any trades that closed since session start.
# # #     Updates daily limits and sends notifications.
# # #     Returns updated set of processed deal tickets.
# # #     """
# # #     deals = executor.get_deals_since(session_start_ts)
# # #     for deal in deals:
# # #         deal_ticket = deal["deal_ticket"]
# # #         if deal_ticket in already_processed:
# # #             continue

# # #         daily_limits.record_trade_closed(deal["ticket"], deal["profit"])

# # #         logger.log_trade_closed(
# # #             ticket=deal["ticket"],
# # #             direction=deal["direction"],
# # #             lot=deal["lot"],
# # #             entry_price=0,  # Not available from deals, logged at open
# # #             close_price=deal["price"],
# # #             pnl=deal["profit"],
# # #             close_reason=deal.get("comment", "TP/SL/Manual"),
# # #         )

# # #         notifier.notify_trade_closed(
# # #             ticket=deal["ticket"],
# # #             direction=deal["direction"],
# # #             lot=deal["lot"],
# # #             entry_price=0,
# # #             close_price=deal["price"],
# # #             pnl=deal["profit"],
# # #             close_reason=deal.get("comment", "TP/SL/Manual"),
# # #             daily_pnl=daily_limits.daily_pnl,
# # #         )

# # #         already_processed.add(deal_ticket)

# # #     return already_processed


# # # # ─── Main Loop ───────────────────────────────────────────

# # # def run(dry_run: bool = False):
# # #     logger.log_startup({
# # #         "mode": "DRY RUN" if dry_run else "LIVE",
# # #         "symbol": MT5_SYMBOL,
# # #         "strategies": "A+B+C+D+E+Regime",
# # #         "e_threshold": E_PROB_THRESHOLD,
# # #     })

# # #     # ── Initialize components ─────────────────────────────
# # #     executor = Executor()
# # #     if not executor.connect():
# # #         logger.error("Cannot connect to MT5 — aborting")
# # #         return

# # #     logger.info("Loading feature engine and strategies...")
# # #     feature_engine = FeatureEngine()
# # #     strategies = {
# # #         "A": StrategyA(),
# # #         "B": StrategyB(),
# # #         "C": StrategyC(),
# # #         "D": StrategyD(),
# # #     }

# # #     logger.info("Loading Strategy E model...")
# # #     try:
# # #         strategy_e = StrategyELive()
# # #         logger.info(f"Strategy E loaded: {E_MODEL_PATH.name}")
# # #     except FileNotFoundError as e:
# # #         logger.error(str(e))
# # #         logger.info("Running in A+B+C+D only mode (no Strategy E)")
# # #         strategy_e = None

# # #     regime_detector = RegimeDetector()
# # #     daily_limits = DailyLimits()
# # #     risk_engine = RiskEngine()

# # #     notifier.notify_startup("DRY RUN" if dry_run else "LIVE")
# # #     logger.info("All components ready — waiting for first M5 bar...")

# # #     session_start_ts = int(datetime.now(UTC).timestamp())
# # #     processed_deals: set = set()
# # #     consecutive_errors = 0

# # #     # ── Timer state ───────────────────────────────────────
# # #     bot_start_time          = datetime.now(UTC)
# # #     last_heartbeat          = datetime.now(UTC)
# # #     last_position_update    = datetime.now(UTC)
# # #     HEARTBEAT_INTERVAL_SECS = 30 * 60        # 30 minutes
# # #     POSITION_UPDATE_SECS    =  5 * 60        # 5 minutes (aligns with M5 bar)

# # #     # Daily loss warning levels as % of MAX_DAILY_LOSS ($100)
# # #     # Resets each day alongside daily_limits
# # #     _warned_pct: set = set()    # tracks which % levels already sent today

# # #     # ── Main loop ─────────────────────────────────────────
# # #     while True:
# # #         try:
# # #             # Wait for next closed M5 bar
# # #             new_bar_time = executor.wait_for_new_bar("M5", poll_seconds=5.0)
# # #             logger.debug(f"New M5 bar: {new_bar_time}")

# # #             # Check for closed trades first
# # #             processed_deals = check_closed_trades(
# # #                 executor, daily_limits, session_start_ts, processed_deals
# # #             )

# # #             # Compute features
# # #             df, current_row, idx = prepare_features(executor, feature_engine)
# # #             if df is None:
# # #                 logger.warning("Feature computation failed — skipping bar")
# # #                 continue

# # #             # Skip if Strategy E not loaded (no independent signals)
# # #             effective_e = strategy_e  # None is handled gracefully in _resolve_decision

# # #             # Process bar
# # #             result = process_bar(
# # #                 df=df,
# # #                 idx=idx,
# # #                 current_row=current_row,
# # #                 strategies=strategies,
# # #                 strategy_e=effective_e,
# # #                 regime_detector=regime_detector,
# # #                 daily_limits=daily_limits,
# # #                 risk_engine=risk_engine,
# # #                 executor=executor,
# # #                 dry_run=dry_run,
# # #             )

# # #             # ── Daily rollover ────────────────────────────
# # #             if daily_limits.check_day_rollover():
# # #                 logger.info("New trading day — daily limits reset")
# # #                 _warned_pct.clear()
# # #                 notifier.notify_daily_summary(
# # #                     date.today().isoformat(),
# # #                     daily_limits.daily_pnl,
# # #                     daily_limits.trade_count,
# # #                     daily_limits.win_count,
# # #                     executor.get_account_info().get("balance", 0),
# # #                 )

# # #             # ── Daily rollover also resets warning set ────
# # #             if daily_limits.check_day_rollover():
# # #                 _warned_pct.clear()

# # #             consecutive_errors = 0

# # #             if result["action"] != "SKIP":
# # #                 logger.info(
# # #                     f"BAR PROCESSED | action={result['action']} "
# # #                     f"ticket={result.get('ticket')} "
# # #                     f"session={result['session']}"
# # #                 )

# # #             now_ts = datetime.now(UTC)

# # #             # ── Heartbeat (every 30 min) ──────────────────
# # #             if (now_ts - last_heartbeat).total_seconds() >= HEARTBEAT_INTERVAL_SECS:
# # #                 open_pos  = executor.get_open_positions()
# # #                 acct      = executor.get_account_info()
# # #                 uptime    = int((now_ts - bot_start_time).total_seconds() / 60)
# # #                 session   = get_current_session(now_ts)
# # #                 regime    = "RANGING"
# # #                 try:
# # #                     df_hb, row_hb, _ = prepare_features(executor, feature_engine)
# # #                     if row_hb is not None:
# # #                         regime, _ = regime_detector.detect_and_adjust(row_hb)
# # #                 except Exception:
# # #                     pass
# # #                 notifier.notify_heartbeat(
# # #                     uptime_mins=uptime,
# # #                     daily_pnl=daily_limits.daily_pnl,
# # #                     open_trades=len(open_pos),
# # #                     trade_count=daily_limits.trade_count,
# # #                     balance=acct.get("balance", 0),
# # #                     session=session,
# # #                     regime=regime,
# # #                 )
# # #                 last_heartbeat = now_ts

# # #             # ── Open position update (every 5 min) ────────
# # #             if (now_ts - last_position_update).total_seconds() >= POSITION_UPDATE_SECS:
# # #                 open_pos = executor.get_open_positions()
# # #                 if open_pos:
# # #                     notifier.notify_position_update(open_pos, daily_limits.daily_pnl)
# # #                 last_position_update = now_ts

# # #             # ── Daily loss warnings (50% / 75% / 100%) ───
# # #             from config import MAX_DAILY_LOSS as _MAX_DAILY_LOSS
# # #             dpnl = daily_limits.daily_pnl
# # #             for pct_level in [50, 75, 100]:
# # #                 if pct_level in _warned_pct:
# # #                     continue
# # #                 threshold = -(_MAX_DAILY_LOSS * pct_level / 100)
# # #                 if dpnl <= threshold:
# # #                     notifier.notify_daily_limit_warning(
# # #                         daily_pnl=dpnl,
# # #                         limit=_MAX_DAILY_LOSS,
# # #                         pct_used=pct_level,
# # #                         trades_today=daily_limits.trade_count,
# # #                     )
# # #                     _warned_pct.add(pct_level)
# # #                     logger.warning(
# # #                         f"Daily loss {pct_level}% warning — "
# # #                         f"${dpnl:.2f} of -${_MAX_DAILY_LOSS} limit"
# # #                     )

# # #         except KeyboardInterrupt:
# # #             logger.info("Keyboard interrupt — shutting down")
# # #             notifier.notify_shutdown("Keyboard interrupt")
# # #             break

# # #         except Exception as e:
# # #             consecutive_errors += 1
# # #             logger.log_error("main_loop", e)
# # #             logger.error(traceback.format_exc())

# # #             if consecutive_errors >= 5:
# # #                 notifier.notify_error("main_loop", f"5 consecutive errors — restarting MT5")
# # #                 executor.reconnect()
# # #                 consecutive_errors = 0
# # #             else:
# # #                 time.sleep(10)

# # #     executor.disconnect()
# # #     logger.info("MIDAS v2 stopped.")


# # # # ─── Entry Point ─────────────────────────────────────────

# # # if __name__ == "__main__":
# # #     parser = argparse.ArgumentParser(description="MIDAS v2 Live Trader")
# # #     parser.add_argument(
# # #         "--dry-run", action="store_true",
# # #         help="Run signals only — no orders sent to MT5"
# # #     )
# # #     parser.add_argument(
# # #         "--check", action="store_true",
# # #         help="Check MT5 connection and exit"
# # #     )
# # #     args = parser.parse_args()

# # #     if args.check:
# # #         ex = Executor()
# # #         if ex.connect():
# # #             info = ex.get_account_info()
# # #             print(f"MT5 OK | Balance: ${info['balance']:.2f} | Equity: ${info['equity']:.2f}")
# # #             ex.disconnect()
# # #         else:
# # #             print("MT5 connection FAILED")
# # #         sys.exit(0)

# # #     run(dry_run=args.dry_run)

# # """
# # Project MIDAS v2 — Main Live Trading Loop
# # ==========================================
# # Connects all components and runs the A+B+C+D+E+Regime strategy live on MT5.

# # Architecture per M5 bar:
# #   1. Fetch latest candles (M5 + M15) from MT5
# #   2. Compute 167 features (FeatureEngine)
# #   3. Detect market regime (RegimeDetector)
# #   4. Run strategies A, B, C, D → collect signals
# #   5. Run DecisionEngine (A-D signals + Strategy E validation + E independent)
# #   6. Apply regime adjustments to E thresholds
# #   7. RiskEngine final check
# #   8. Execute via MT5 (Executor)
# #   9. Monitor for closed trades → update daily P&L
# #   10. Log + Telegram notify

# # Usage:
# #     python main.py               # Live trading
# #     python main.py --dry-run     # Dry run (signals only, no orders)
# #     python main.py --check       # Check MT5 connection and exit
# # """

# # import argparse
# # import sys
# # import time
# # import json
# # import traceback
# # from datetime import datetime, date, UTC
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # import lightgbm as lgb

# # sys.path.insert(0, str(Path(__file__).parent))

# # from config import (
# #     MT5_SYMBOL, MODELS_DIR, DATA_PROCESSED,
# #     ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE,
# #     LOT_SIZE_MIN, LOT_SIZE_MAX, LOT_SIZE_SINGLE,
# #     SESSIONS,
# # )
# # from feature_engineering import FeatureEngine
# # from strategies.strategy_a import StrategyA
# # from strategies.strategy_b import StrategyB
# # from strategies.strategy_c import StrategyC
# # from strategies.strategy_d import StrategyD
# # from regime_detector import RegimeDetector, REGIME_ADJUSTMENTS
# # from decision_engine import DecisionEngine

# # import logger
# # import notifier
# # from daily_limits import DailyLimits
# # from risk_engine import RiskEngine
# # from executor import Executor


# # # ─── Strategy E Config (debiased best threshold) ─────────
# # E_PROB_THRESHOLD    = 0.68
# # E_VAL_STRONG_AGREE  = 0.60
# # E_VAL_AGREE         = 0.52
# # E_VAL_CONFLICT_SKIP = 0.60
# # E_STRONG_BOOST      = 10
# # E_WEAK_PENALTY      = -10
# # E_SL_ATR_MULT       = 1.5
# # E_TP_RR_BASE        = 1.5
# # E_TP_RR_HIGH        = 2.5
# # E_INDEPENDENT_MIN_CONF = 70

# # # Walk-forward model segments (use final model for live trading)
# # E_MODEL_PATH    = MODELS_DIR / "strategy_e_clean_final.txt"
# # E_METADATA_PATH = MODELS_DIR / "strategy_e_clean_metadata.json"

# # # How many candles to fetch for feature engineering
# # M5_LOOKBACK  = 500    # Enough for EMA200 + all indicators
# # M15_LOOKBACK = 200


# # # ─── Session Detection ───────────────────────────────────

# # def get_current_session(dt: datetime) -> str:
# #     """Determine current trading session from UTC hour."""
# #     hour = dt.hour
# #     # london_ny_overlap: 12:00 - 14:00 UTC
# #     if 12 <= hour < 14:
# #         return "london_ny_overlap"
# #     # london: 06:00 - 14:00 UTC
# #     if 6 <= hour < 14:
# #         return "london"
# #     # new_york: 14:00 - 20:00 UTC (adjusted from config for overlap)
# #     if 14 <= hour < 20:
# #         return "new_york"
# #     # asian: 00:00 - 06:00 UTC
# #     if 0 <= hour < 6:
# #         return "asian"
# #     return "off"


# # # ─── Feature Preparation ─────────────────────────────────

# # def prepare_features(executor: Executor, feature_engine: FeatureEngine) -> tuple:
# #     """
# #     Fetch latest candles from MT5 and compute features.
# #     Returns (df_featured: pd.DataFrame, current_row: pd.Series, current_idx: int)
# #     or (None, None, None) on error.
# #     """
# #     df_m5 = executor.get_candles("M5", M5_LOOKBACK)
# #     df_m15 = executor.get_candles("M15", M15_LOOKBACK)

# #     if df_m5 is None or df_m15 is None:
# #         logger.error("Failed to fetch candles from MT5")
# #         return None, None, None

# #     if len(df_m5) < 250:
# #         logger.warning(f"Not enough M5 candles: {len(df_m5)} (need 250+)")
# #         return None, None, None

# #     try:
# #         df = feature_engine.compute(df_m5, df_m15)
# #         if df is None or df.empty:
# #             logger.error("FeatureEngine returned empty dataframe")
# #             return None, None, None

# #         idx = len(df) - 1
# #         current_row = df.iloc[idx]
# #         return df, current_row, idx

# #     except Exception as e:
# #         logger.log_error("prepare_features", e)
# #         return None, None, None


# # # ─── Strategy E Inference ────────────────────────────────

# # class StrategyELive:
# #     """Thin wrapper around the trained LightGBM model for live inference."""

# #     def __init__(self):
# #         if not E_MODEL_PATH.exists():
# #             raise FileNotFoundError(
# #                 f"Strategy E model not found: {E_MODEL_PATH}\n"
# #                 f"Run strategy_e_clean_retrain.py first."
# #             )
# #         self.model = lgb.Booster(model_file=str(E_MODEL_PATH))

# #         with open(E_METADATA_PATH) as f:
# #             meta = json.load(f)
# #         self.feature_cols = meta["feature_cols"]

# #     def predict(self, row: pd.Series) -> float:
# #         """Return raw probability (0-1) for BUY direction."""
# #         try:
# #             feat = row[self.feature_cols].values.reshape(1, -1)
# #             prob = float(self.model.predict(feat)[0])
# #             return prob
# #         except Exception as e:
# #             logger.log_error("StrategyELive.predict", e)
# #             return 0.5  # Neutral on error


# # # ─── Main Signal Processing ──────────────────────────────

# # def process_bar(
# #     df: pd.DataFrame,
# #     idx: int,
# #     current_row: pd.Series,
# #     strategies: dict,
# #     strategy_e: StrategyELive,
# #     regime_detector: RegimeDetector,
# #     daily_limits: DailyLimits,
# #     risk_engine: RiskEngine,
# #     executor: Executor,
# #     dry_run: bool = False,
# # ) -> dict:
# #     """
# #     Process one M5 bar. Returns a result dict describing what happened.
# #     """
# #     now = datetime.now(UTC)
# #     session = get_current_session(now)
# #     bar_time = current_row.get("datetime", now)

# #     result = {
# #         "bar_time": str(bar_time),
# #         "session": session,
# #         "action": "SKIP",
# #         "reason": "",
# #         "ticket": None,
# #     }

# #     # ── Daily rollover check (handled in main loop) ───────
# #     daily_limits.check_day_rollover()

# #     # ── Pre-flight: can we trade at all? ──────────────────
# #     can_trade, block_reason = daily_limits.can_trade(session, now)
# #     if not can_trade:
# #         result["reason"] = block_reason
# #         logger.debug(f"Skipping bar — {block_reason}")
# #         return result

# #     # ── Run strategies A-D ────────────────────────────────
# #     ad_signals = []
# #     for name, strat in strategies.items():
# #         try:
# #             signal = strat.generate_signal(df, idx)
# #             if signal is not None:
# #                 ad_signals.append({
# #                     "strategy": name,
# #                     "direction": signal.direction.upper(),
# #                     "confidence": signal.confidence,
# #                     "sl_price": signal.sl_price,
# #                     "tp_price": signal.tp_price,
# #                     "entry_price": signal.entry_price,
# #                 })
# #         except Exception as e:
# #             logger.log_error(f"strategy_{name}", e)

# #     # ── Regime detection ─────────────────────────────────
# #     regime = "RANGING"
# #     adj = REGIME_ADJUSTMENTS.get("RANGING", {})
# #     try:
# #         regime, adj = regime_detector.detect_and_adjust(current_row)
# #     except Exception as e:
# #         logger.log_error("regime_detector", e)

# #     # ── Strategy E prediction ─────────────────────────────
# #     e_prob = 0.5
# #     try:
# #         e_prob = strategy_e.predict(current_row)
# #     except Exception as e:
# #         logger.log_error("strategy_e.predict", e)

# #     # ── Resolve direction and build trade decision ─────────
# #     # Uses same logic as run_phase9_regime.py
# #     decision = _resolve_decision(
# #         ad_signals, e_prob, adj, regime, df, idx, current_row, session
# #     )

# #     if decision["action"] == "SKIP":
# #         result["reason"] = decision["reason"]
# #         log_str = f"[{bar_time}] {session} | regime={regime} | SKIP: {decision['reason']}"
# #         logger.debug(log_str)
# #         return result

# #     # ── Log the signal ────────────────────────────────────
# #     logger.log_signal(
# #         bar_time=bar_time,
# #         strategy=decision["source"],
# #         direction=decision["action"],
# #         confidence=decision.get("confidence", 0),
# #         regime=regime,
# #         action_taken="EXECUTE" if not dry_run else "DRY_RUN",
# #         reason=decision["reason"],
# #     )

# #     if dry_run:
# #         result.update({"action": decision["action"], "reason": "DRY_RUN — no order sent"})
# #         logger.info(
# #             f"[DRY RUN] {decision['action']} | lot={decision['lot']} | "
# #             f"source={decision['source']} | {decision['reason']}"
# #         )
# #         return result

# #     # ── Risk engine final check ────────────────────────────
# #     account = executor.get_account_info()
# #     approved, risk_msg, final_lot = risk_engine.check_order(
# #         direction=decision["action"],
# #         lot=decision["lot"],
# #         sl_pips=decision["sl_pips"],
# #         daily_pnl=daily_limits.daily_pnl,
# #         account_balance=account.get("balance", 5000),
# #         account_equity=account.get("equity", 5000),
# #     )

# #     if not approved:
# #         logger.log_risk_block(risk_msg, decision["source"])
# #         notifier.notify_risk_block(risk_msg)
# #         result["reason"] = f"RISK BLOCK: {risk_msg}"
# #         return result

# #     # Validate SL/TP sides
# #     sl_valid, sl_msg = risk_engine.validate_sl_tp(
# #         decision["action"],
# #         decision["entry_price"],
# #         decision["sl_price"],
# #         decision["tp_price"],
# #     )
# #     if not sl_valid:
# #         logger.log_risk_block(sl_msg, decision["source"])
# #         result["reason"] = f"SL/TP INVALID: {sl_msg}"
# #         return result

# #     # ── Place the order ────────────────────────────────────
# #     success, ticket, msg = executor.place_order(
# #         direction=decision["action"],
# #         lot=final_lot,
# #         sl_price=decision["sl_price"],
# #         tp_price=decision["tp_price"],
# #         comment=f"MIDAS_{decision['source'][:10]}",
# #     )

# #     if not success:
# #         logger.error(f"Order failed: {msg}")
# #         notifier.notify_error("place_order", msg)
# #         result["reason"] = f"ORDER FAILED: {msg}"
# #         return result

# #     # ── Record the trade ──────────────────────────────────
# #     daily_limits.record_trade_opened(ticket, session, now)

# #     logger.log_trade_opened(
# #         ticket=ticket,
# #         direction=decision["action"],
# #         lot=final_lot,
# #         entry_price=decision["entry_price"],
# #         sl_price=decision["sl_price"],
# #         tp_price=decision["tp_price"],
# #         source=decision["source"],
# #         regime=regime,
# #         reason=decision["reason"],
# #     )

# #     notifier.notify_trade_opened(
# #         ticket=ticket,
# #         direction=decision["action"],
# #         lot=final_lot,
# #         entry_price=decision["entry_price"],
# #         sl_price=decision["sl_price"],
# #         tp_price=decision["tp_price"],
# #         source=decision["source"],
# #         regime=regime,
# #     )

# #     result.update({
# #         "action": decision["action"],
# #         "ticket": ticket,
# #         "lot": final_lot,
# #         "reason": decision["reason"],
# #     })
# #     return result


# # def _resolve_decision(
# #     ad_signals: list, e_prob: float, adj: dict, regime: str,
# #     df: pd.DataFrame, idx: int, row: pd.Series, session: str,
# # ) -> dict:
# #     """
# #     Combine A-D signals + regime + E probability into a final trade decision.
# #     Mirrors the logic from run_phase9_regime.py's RegimeAwareCombinedABCDE.
# #     """
# #     skip = lambda reason: {
# #         "action": "SKIP", "reason": reason, "lot": 0,
# #         "entry_price": 0, "sl_price": 0, "tp_price": 0,
# #         "sl_pips": 0, "source": "none", "confidence": 0,
# #     }

# #     atr = float(row.get(f"atr_{ATR_PERIOD}", row.get("atr", 2.0)))

# #     # ── Get regime-adjusted E thresholds ─────────────────
# #     e_indep_thresh  = adj.get("e_indep_threshold", E_PROB_THRESHOLD)
# #     e_conf_boost    = adj.get("e_conf_boost", 0)
# #     sl_mult         = adj.get("sl_mult", 1.0)
# #     tp_mult         = adj.get("tp_mult", 1.0)
# #     allow_e_indep   = adj.get("allow_e_independent", True)
# #     e_strong_thresh = adj.get("e_val_strong_agree", E_VAL_STRONG_AGREE)
# #     e_agree_thresh  = adj.get("e_val_agree", E_VAL_AGREE)
# #     e_conflict_skip = adj.get("e_val_conflict_skip", E_VAL_CONFLICT_SKIP)

# #     current_price = float(row.get("close", 0))

# #     # ── Case 1: A-D signals exist ────────────────────────
# #     if ad_signals:
# #         buy_sigs  = [s for s in ad_signals if s["direction"] == "BUY"]
# #         sell_sigs = [s for s in ad_signals if s["direction"] == "SELL"]

# #         # Resolve direction conflict
# #         if buy_sigs and sell_sigs:
# #             buy_conf  = sum(s["confidence"] for s in buy_sigs)
# #             sell_conf = sum(s["confidence"] for s in sell_sigs)
# #             if buy_conf > sell_conf:
# #                 direction, primary = "BUY", buy_sigs
# #             elif sell_conf > buy_conf:
# #                 direction, primary = "SELL", sell_sigs
# #             else:
# #                 return skip("A-D conflict — equal confidence")
# #         elif buy_sigs:
# #             direction, primary = "BUY", buy_sigs
# #         else:
# #             direction, primary = "SELL", sell_sigs

# #         best = max(primary, key=lambda s: s["confidence"])

# #         # E validation
# #         if direction == "BUY":
# #             e_agree = e_prob >= e_agree_thresh
# #             e_strong = e_prob >= e_strong_thresh
# #             e_conflict = (1 - e_prob) >= e_conflict_skip
# #         else:
# #             e_agree = (1 - e_prob) >= e_agree_thresh
# #             e_strong = (1 - e_prob) >= e_strong_thresh
# #             e_conflict = e_prob >= e_conflict_skip

# #         if e_conflict:
# #             return skip(f"E conflicts with {direction} (prob={e_prob:.3f})")

# #         # Lot sizing based on agreement count + E boost
# #         n_agree = len(primary)
# #         if n_agree >= 2:
# #             lot = 0.02
# #         else:
# #             lot = LOT_SIZE_SINGLE

# #         # Use SL/TP from best signal
# #         sl_price = best["sl_price"]
# #         tp_price = best["tp_price"]
# #         entry    = best["entry_price"] if best["entry_price"] > 0 else current_price

# #         # Apply regime SL/TP multipliers
# #         if sl_mult != 1.0:
# #             sl_dist = abs(entry - sl_price) * sl_mult
# #             sl_price = entry - sl_dist if direction == "BUY" else entry + sl_dist

# #         if tp_mult != 1.0:
# #             tp_dist = abs(tp_price - entry) * tp_mult
# #             tp_price = entry + tp_dist if direction == "BUY" else entry - tp_dist

# #         sl_pips = abs(entry - sl_price) / PIP_VALUE
# #         source  = "+".join(s["strategy"] for s in primary) + "+E_val"
# #         conf    = best["confidence"] + (e_conf_boost if e_agree else 0)

# #         return {
# #             "action": direction,
# #             "lot": lot,
# #             "entry_price": round(entry, 2),
# #             "sl_price": round(sl_price, 2),
# #             "tp_price": round(tp_price, 2),
# #             "sl_pips": round(sl_pips, 1),
# #             "source": source,
# #             "confidence": conf,
# #             "reason": (
# #                 f"{source} conf={conf:.0f}% e_prob={e_prob:.3f} "
# #                 f"regime={regime} n={n_agree}"
# #             ),
# #         }

# #     # ── Case 2: No A-D signal — E independent ────────────
# #     if not allow_e_indep:
# #         return skip(f"E independent blocked by regime={regime}")

# #     # BUY if prob >= threshold, SELL if (1-prob) >= threshold
# #     if e_prob >= e_indep_thresh:
# #         direction = "BUY"
# #         e_conf = e_prob
# #     elif (1 - e_prob) >= e_indep_thresh:
# #         direction = "SELL"
# #         e_conf = 1 - e_prob
# #     else:
# #         return skip(f"E no signal (prob={e_prob:.3f} thresh={e_indep_thresh:.2f})")

# #     # E independent: fixed SL/TP from ATR
# #     sl_atr = atr * E_SL_ATR_MULT * sl_mult
# #     tp_atr = sl_atr * E_TP_RR_HIGH if e_conf >= 0.75 else sl_atr * E_TP_RR_BASE
# #     tp_atr *= tp_mult

# #     sl_pips = sl_atr / PIP_VALUE
# #     entry   = current_price

# #     if direction == "BUY":
# #         sl_price = entry - sl_atr
# #         tp_price = entry + tp_atr
# #     else:
# #         sl_price = entry + sl_atr
# #         tp_price = entry - tp_atr

# #     return {
# #         "action": direction,
# #         "lot": LOT_SIZE_SINGLE,
# #         "entry_price": round(entry, 2),
# #         "sl_price": round(sl_price, 2),
# #         "tp_price": round(tp_price, 2),
# #         "sl_pips": round(sl_pips, 1),
# #         "source": "E_independent",
# #         "confidence": round(e_conf * 100, 1),
# #         "reason": (
# #             f"E_indep {direction} prob={e_prob:.3f} "
# #             f"regime={regime} sl={sl_pips:.0f}pips"
# #         ),
# #     }


# # # ─── Closed Trade Monitor ────────────────────────────────

# # def check_closed_trades(
# #     executor: Executor,
# #     daily_limits: DailyLimits,
# #     session_start_ts: int,
# #     already_processed: set,
# # ) -> set:
# #     """
# #     Check for any trades that closed since session start.
# #     Updates daily limits and sends notifications.
# #     Returns updated set of processed deal tickets.
# #     """
# #     deals = executor.get_deals_since(session_start_ts)
# #     for deal in deals:
# #         deal_ticket = deal["deal_ticket"]
# #         if deal_ticket in already_processed:
# #             continue

# #         daily_limits.record_trade_closed(deal["ticket"], deal["profit"])

# #         logger.log_trade_closed(
# #             ticket=deal["ticket"],
# #             direction=deal["direction"],
# #             lot=deal["lot"],
# #             entry_price=0,  # Not available from deals, logged at open
# #             close_price=deal["price"],
# #             pnl=deal["profit"],
# #             close_reason=deal.get("comment", "TP/SL/Manual"),
# #         )

# #         notifier.notify_trade_closed(
# #             ticket=deal["ticket"],
# #             direction=deal["direction"],
# #             lot=deal["lot"],
# #             entry_price=0,
# #             close_price=deal["price"],
# #             pnl=deal["profit"],
# #             close_reason=deal.get("comment", "TP/SL/Manual"),
# #             daily_pnl=daily_limits.daily_pnl,
# #         )

# #         already_processed.add(deal_ticket)

# #     return already_processed


# # # ─── Main Loop ───────────────────────────────────────────

# # def run(dry_run: bool = False):
# #     logger.log_startup({
# #         "mode": "DRY RUN" if dry_run else "LIVE",
# #         "symbol": MT5_SYMBOL,
# #         "strategies": "A+B+C+D+E+Regime",
# #         "e_threshold": E_PROB_THRESHOLD,
# #     })

# #     # ── Initialize components ─────────────────────────────
# #     executor = Executor()
# #     if not executor.connect():
# #         logger.error("Cannot connect to MT5 — aborting")
# #         return

# #     logger.info("Loading feature engine and strategies...")
# #     feature_engine = FeatureEngine()
# #     strategies = {
# #         "A": StrategyA(),
# #         "B": StrategyB(),
# #         "C": StrategyC(),
# #         "D": StrategyD(),
# #     }

# #     logger.info("Loading Strategy E model...")
# #     try:
# #         strategy_e = StrategyELive()
# #         logger.info(f"Strategy E loaded: {E_MODEL_PATH.name}")
# #     except FileNotFoundError as e:
# #         logger.error(str(e))
# #         logger.info("Running in A+B+C+D only mode (no Strategy E)")
# #         strategy_e = None

# #     regime_detector = RegimeDetector()
# #     daily_limits = DailyLimits()
# #     risk_engine = RiskEngine()

# #     notifier.notify_startup("DRY RUN" if dry_run else "LIVE")
# #     logger.info("All components ready — waiting for first M5 bar...")

# #     session_start_ts = int(datetime.now(UTC).timestamp())
# #     processed_deals: set = set()
# #     consecutive_errors = 0

# #     # ── Timer state ───────────────────────────────────────
# #     bot_start_time          = datetime.now(UTC)
# #     last_heartbeat          = datetime.now(UTC)
# #     last_position_update    = datetime.now(UTC)
# #     HEARTBEAT_INTERVAL_SECS = 30 * 60        # 30 minutes
# #     POSITION_UPDATE_SECS    =  5 * 60        # 5 minutes (aligns with M5 bar)

# #     # Daily loss warning levels as % of MAX_DAILY_LOSS ($100)
# #     # Resets each day alongside daily_limits
# #     _warned_pct: set = set()    # tracks which % levels already sent today

# #     # ── Main loop ─────────────────────────────────────────
# #     while True:
# #         try:
# #             # Wait for next closed M5 bar
# #             new_bar_time = executor.wait_for_new_bar("M5", poll_seconds=5.0)
# #             logger.debug(f"New M5 bar: {new_bar_time}")

# #             # Check for closed trades first
# #             processed_deals = check_closed_trades(
# #                 executor, daily_limits, session_start_ts, processed_deals
# #             )

# #             # Compute features
# #             df, current_row, idx = prepare_features(executor, feature_engine)
# #             if df is None:
# #                 logger.warning("Feature computation failed — skipping bar")
# #                 continue

# #             # Skip if Strategy E not loaded (no independent signals)
# #             effective_e = strategy_e  # None is handled gracefully in _resolve_decision

# #             # Process bar
# #             result = process_bar(
# #                 df=df,
# #                 idx=idx,
# #                 current_row=current_row,
# #                 strategies=strategies,
# #                 strategy_e=effective_e,
# #                 regime_detector=regime_detector,
# #                 daily_limits=daily_limits,
# #                 risk_engine=risk_engine,
# #                 executor=executor,
# #                 dry_run=dry_run,
# #             )

# #             # ── Daily rollover ────────────────────────────
# #             if daily_limits.check_day_rollover():
# #                 logger.info("New trading day — daily limits reset")
# #                 _warned_pct.clear()
# #                 notifier.notify_daily_summary(
# #                     date.today().isoformat(),
# #                     daily_limits.daily_pnl,
# #                     daily_limits.trade_count,
# #                     daily_limits.win_count,
# #                     executor.get_account_info().get("balance", 0),
# #                 )

# #             # ── Daily rollover also resets warning set ────
# #             if daily_limits.check_day_rollover():
# #                 _warned_pct.clear()

# #             consecutive_errors = 0

# #             if result["action"] != "SKIP":
# #                 logger.info(
# #                     f"BAR PROCESSED | action={result['action']} "
# #                     f"ticket={result.get('ticket')} "
# #                     f"session={result['session']}"
# #                 )

# #             now_ts = datetime.now(UTC)

# #             # ── Heartbeat (every 30 min) ──────────────────
# #             if (now_ts - last_heartbeat).total_seconds() >= HEARTBEAT_INTERVAL_SECS:
# #                 open_pos  = executor.get_open_positions()
# #                 acct      = executor.get_account_info()
# #                 uptime    = int((now_ts - bot_start_time).total_seconds() / 60)
# #                 session   = get_current_session(now_ts)
# #                 regime    = "RANGING"
# #                 try:
# #                     df_hb, row_hb, _ = prepare_features(executor, feature_engine)
# #                     if row_hb is not None:
# #                         regime, _ = regime_detector.detect_and_adjust(row_hb)
# #                 except Exception:
# #                     pass
# #                 notifier.notify_heartbeat(
# #                     uptime_mins=uptime,
# #                     daily_pnl=daily_limits.daily_pnl,
# #                     open_trades=len(open_pos),
# #                     trade_count=daily_limits.trade_count,
# #                     balance=acct.get("balance", 0),
# #                     session=session,
# #                     regime=regime,
# #                 )
# #                 last_heartbeat = now_ts

# #             # ── Open position update (every 5 min) ────────
# #             if (now_ts - last_position_update).total_seconds() >= POSITION_UPDATE_SECS:
# #                 open_pos = executor.get_open_positions()
# #                 if open_pos:
# #                     notifier.notify_position_update(open_pos, daily_limits.daily_pnl)
# #                 last_position_update = now_ts

# #             # ── Daily loss warnings (50% / 75% / 100%) ───
# #             from config import MAX_DAILY_LOSS as _MAX_DAILY_LOSS
# #             dpnl = daily_limits.daily_pnl
# #             for pct_level in [50, 75, 100]:
# #                 if pct_level in _warned_pct:
# #                     continue
# #                 threshold = -(_MAX_DAILY_LOSS * pct_level / 100)
# #                 if dpnl <= threshold:
# #                     notifier.notify_daily_limit_warning(
# #                         daily_pnl=dpnl,
# #                         limit=_MAX_DAILY_LOSS,
# #                         pct_used=pct_level,
# #                         trades_today=daily_limits.trade_count,
# #                     )
# #                     _warned_pct.add(pct_level)
# #                     logger.warning(
# #                         f"Daily loss {pct_level}% warning — "
# #                         f"${dpnl:.2f} of -${_MAX_DAILY_LOSS} limit"
# #                     )

# #         except KeyboardInterrupt:
# #             logger.info("Keyboard interrupt — shutting down")
# #             notifier.notify_shutdown("Keyboard interrupt")
# #             break

# #         except Exception as e:
# #             consecutive_errors += 1
# #             logger.log_error("main_loop", e)
# #             logger.error(traceback.format_exc())

# #             if consecutive_errors >= 5:
# #                 notifier.notify_error("main_loop", f"5 consecutive errors — restarting MT5")
# #                 executor.reconnect()
# #                 consecutive_errors = 0
# #             else:
# #                 time.sleep(10)

# #     executor.disconnect()
# #     logger.info("MIDAS v2 stopped.")


# # # ─── Entry Point ─────────────────────────────────────────

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="MIDAS v2 Live Trader")
# #     parser.add_argument(
# #         "--dry-run", action="store_true",
# #         help="Run signals only — no orders sent to MT5"
# #     )
# #     parser.add_argument(
# #         "--check", action="store_true",
# #         help="Check MT5 connection and exit"
# #     )
# #     args = parser.parse_args()

# #     if args.check:
# #         ex = Executor()
# #         if ex.connect():
# #             info = ex.get_account_info()
# #             print(f"MT5 OK | Balance: ${info['balance']:.2f} | Equity: ${info['equity']:.2f}")
# #             ex.disconnect()
# #         else:
# #             print("MT5 connection FAILED")
# #         sys.exit(0)
# # """
# # Project MIDAS v2 — Main Live Trading Loop
# # ==========================================
# # Connects all components and runs the A+B+C+D+E+Regime strategy live on MT5.

# # Architecture per M5 bar:
# #   1. Fetch latest candles (M5 + M15) from MT5
# #   2. Compute 167 features (FeatureEngine)
# #   3. Detect market regime (RegimeDetector)
# #   4. Run strategies A, B, C, D → collect signals
# #   5. Run DecisionEngine (A-D signals + Strategy E validation + E independent)
# #   6. Apply regime adjustments to E thresholds
# #   7. RiskEngine final check
# #   8. Execute via MT5 (Executor)
# #   9. Monitor for closed trades → update daily P&L
# #   10. Log + Telegram notify

# # Usage:
# #     python main.py               # Live trading
# #     python main.py --dry-run     # Dry run (signals only, no orders)
# #     python main.py --check       # Check MT5 connection and exit
# # """

# # import argparse
# # import sys
# # import time
# # import json
# # import traceback
# # from datetime import datetime, date
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # import lightgbm as lgb

# # sys.path.insert(0, str(Path(__file__).parent))

# # from config import (
# #     MT5_SYMBOL, MODELS_DIR, DATA_PROCESSED,
# #     ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE,
# #     LOT_SIZE_MIN, LOT_SIZE_MAX, LOT_SIZE_SINGLE,
# #     SESSIONS,
# # )
# # from feature_engineering import FeatureEngine
# # from strategies.strategy_a import StrategyA
# # from strategies.strategy_b import StrategyB
# # from strategies.strategy_c import StrategyC
# # from strategies.strategy_d import StrategyD
# # from regime_detector import RegimeDetector, REGIME_ADJUSTMENTS
# # from decision_engine import DecisionEngine

# # import logger
# # import notifier
# # from daily_limits import DailyLimits
# # from risk_engine import RiskEngine
# # from executor import Executor


# # # ─── Strategy E Config (debiased best threshold) ─────────
# # E_PROB_THRESHOLD    = 0.68
# # E_VAL_STRONG_AGREE  = 0.60
# # E_VAL_AGREE         = 0.52
# # E_VAL_CONFLICT_SKIP = 0.60
# # E_STRONG_BOOST      = 10
# # E_WEAK_PENALTY      = -10
# # E_SL_ATR_MULT       = 1.5
# # E_TP_RR_BASE        = 1.5
# # E_TP_RR_HIGH        = 2.5
# # E_INDEPENDENT_MIN_CONF = 70

# # # Walk-forward model segments (use final model for live trading)
# # E_MODEL_PATH    = MODELS_DIR / "strategy_e_clean_final.txt"
# # E_METADATA_PATH = MODELS_DIR / "strategy_e_clean_metadata.json"

# # # How many candles to fetch for feature engineering
# # M5_LOOKBACK  = 500    # Enough for EMA200 + all indicators
# # M15_LOOKBACK = 200


# # # ─── Session Detection ───────────────────────────────────

# # def get_current_session(dt: datetime) -> str:
# #     """Determine current trading session from UTC hour."""
# #     hour = dt.hour
# #     # london_ny_overlap: 12:00 - 14:00 UTC
# #     if 12 <= hour < 14:
# #         return "london_ny_overlap"
# #     # london: 06:00 - 14:00 UTC
# #     if 6 <= hour < 14:
# #         return "london"
# #     # new_york: 14:00 - 20:00 UTC (adjusted from config for overlap)
# #     if 14 <= hour < 20:
# #         return "new_york"
# #     # asian: 00:00 - 06:00 UTC
# #     if 0 <= hour < 6:
# #         return "asian"
# #     return "off"


# # # ─── Feature Preparation ─────────────────────────────────

# # def prepare_features(executor: Executor, feature_engine: FeatureEngine) -> tuple:
# #     """
# #     Fetch latest candles from MT5 and compute features.
# #     Returns (df_featured: pd.DataFrame, current_row: pd.Series, current_idx: int)
# #     or (None, None, None) on error.
# #     """
# #     df_m5 = executor.get_candles("M5", M5_LOOKBACK)
# #     df_m15 = executor.get_candles("M15", M15_LOOKBACK)

# #     if df_m5 is None or df_m15 is None:
# #         logger.error("Failed to fetch candles from MT5")
# #         return None, None, None

# #     if len(df_m5) < 250:
# #         logger.warning(f"Not enough M5 candles: {len(df_m5)} (need 250+)")
# #         return None, None, None

# #     try:
# #         df = feature_engine.compute(df_m5, df_m15)
# #         if df is None or df.empty:
# #             logger.error("FeatureEngine returned empty dataframe")
# #             return None, None, None

# #         idx = len(df) - 1
# #         current_row = df.iloc[idx]
# #         return df, current_row, idx

# #     except Exception as e:
# #         logger.log_error("prepare_features", e)
# #         return None, None, None


# # # ─── Strategy E Inference ────────────────────────────────

# # class StrategyELive:
# #     """Thin wrapper around the trained LightGBM model for live inference."""

# #     def __init__(self):
# #         if not E_MODEL_PATH.exists():
# #             raise FileNotFoundError(
# #                 f"Strategy E model not found: {E_MODEL_PATH}\n"
# #                 f"Run strategy_e_clean_retrain.py first."
# #             )
# #         self.model = lgb.Booster(model_file=str(E_MODEL_PATH))

# #         with open(E_METADATA_PATH) as f:
# #             meta = json.load(f)
# #         self.feature_cols = meta["feature_cols"]

# #     def predict(self, row: pd.Series) -> float:
# #         """Return raw probability (0-1) for BUY direction."""
# #         try:
# #             feat = row[self.feature_cols].values.reshape(1, -1)
# #             prob = float(self.model.predict(feat)[0])
# #             return prob
# #         except Exception as e:
# #             logger.log_error("StrategyELive.predict", e)
# #             return 0.5  # Neutral on error


# # # ─── Main Signal Processing ──────────────────────────────

# # def process_bar(
# #     df: pd.DataFrame,
# #     idx: int,
# #     current_row: pd.Series,
# #     strategies: dict,
# #     strategy_e: StrategyELive,
# #     regime_detector: RegimeDetector,
# #     daily_limits: DailyLimits,
# #     risk_engine: RiskEngine,
# #     executor: Executor,
# #     dry_run: bool = False,
# # ) -> dict:
# #     """
# #     Process one M5 bar. Returns a result dict describing what happened.
# #     """
# #     now = datetime.now(UTC)
# #     session = get_current_session(now)
# #     bar_time = current_row.get("datetime", now)

# #     result = {
# #         "bar_time": str(bar_time),
# #         "session": session,
# #         "action": "SKIP",
# #         "reason": "",
# #         "ticket": None,
# #     }

# #     # ── Daily rollover check (handled in main loop) ───────
# #     daily_limits.check_day_rollover()

# #     # ── Pre-flight: can we trade at all? ──────────────────
# #     can_trade, block_reason = daily_limits.can_trade(session, now)
# #     if not can_trade:
# #         result["reason"] = block_reason
# #         logger.debug(f"Skipping bar — {block_reason}")
# #         return result

# #     # ── Run strategies A-D ────────────────────────────────
# #     ad_signals = []
# #     for name, strat in strategies.items():
# #         try:
# #             signal = strat.generate_signal(df, idx)
# #             if signal is not None:
# #                 ad_signals.append({
# #                     "strategy": name,
# #                     "direction": signal.direction.upper(),
# #                     "confidence": signal.confidence,
# #                     "sl_price": signal.sl_price,
# #                     "tp_price": signal.tp_price,
# #                     "entry_price": signal.entry_price,
# #                 })
# #         except Exception as e:
# #             logger.log_error(f"strategy_{name}", e)

# #     # ── Regime detection ─────────────────────────────────
# #     regime = "RANGING"
# #     adj = REGIME_ADJUSTMENTS.get("RANGING", {})
# #     try:
# #         regime, adj = regime_detector.detect_and_adjust(current_row)
# #     except Exception as e:
# #         logger.log_error("regime_detector", e)

# #     # ── Strategy E prediction ─────────────────────────────
# #     e_prob = 0.5
# #     try:
# #         e_prob = strategy_e.predict(current_row)
# #     except Exception as e:
# #         logger.log_error("strategy_e.predict", e)

# #     # ── Resolve direction and build trade decision ─────────
# #     # Uses same logic as run_phase9_regime.py
# #     decision = _resolve_decision(
# #         ad_signals, e_prob, adj, regime, df, idx, current_row, session
# #     )

# #     if decision["action"] == "SKIP":
# #         result["reason"] = decision["reason"]
# #         log_str = f"[{bar_time}] {session} | regime={regime} | SKIP: {decision['reason']}"
# #         logger.debug(log_str)
# #         return result

# #     # ── Log the signal ────────────────────────────────────
# #     logger.log_signal(
# #         bar_time=bar_time,
# #         strategy=decision["source"],
# #         direction=decision["action"],
# #         confidence=decision.get("confidence", 0),
# #         regime=regime,
# #         action_taken="EXECUTE" if not dry_run else "DRY_RUN",
# #         reason=decision["reason"],
# #     )

# #     if dry_run:
# #         result.update({"action": decision["action"], "reason": "DRY_RUN — no order sent"})
# #         logger.info(
# #             f"[DRY RUN] {decision['action']} | lot={decision['lot']} | "
# #             f"source={decision['source']} | {decision['reason']}"
# #         )
# #         return result

# #     # ── Risk engine final check ────────────────────────────
# #     account = executor.get_account_info()
# #     approved, risk_msg, final_lot = risk_engine.check_order(
# #         direction=decision["action"],
# #         lot=decision["lot"],
# #         sl_pips=decision["sl_pips"],
# #         daily_pnl=daily_limits.daily_pnl,
# #         account_balance=account.get("balance", 5000),
# #         account_equity=account.get("equity", 5000),
# #     )

# #     if not approved:
# #         logger.log_risk_block(risk_msg, decision["source"])
# #         notifier.notify_risk_block(risk_msg)
# #         result["reason"] = f"RISK BLOCK: {risk_msg}"
# #         return result

# #     # Validate SL/TP sides
# #     sl_valid, sl_msg = risk_engine.validate_sl_tp(
# #         decision["action"],
# #         decision["entry_price"],
# #         decision["sl_price"],
# #         decision["tp_price"],
# #     )
# #     if not sl_valid:
# #         logger.log_risk_block(sl_msg, decision["source"])
# #         result["reason"] = f"SL/TP INVALID: {sl_msg}"
# #         return result

# #     # ── Place the order ────────────────────────────────────
# #     success, ticket, msg = executor.place_order(
# #         direction=decision["action"],
# #         lot=final_lot,
# #         sl_price=decision["sl_price"],
# #         tp_price=decision["tp_price"],
# #         comment=f"MIDAS_{decision['source'][:10]}",
# #     )

# #     if not success:
# #         logger.error(f"Order failed: {msg}")
# #         notifier.notify_error("place_order", msg)
# #         result["reason"] = f"ORDER FAILED: {msg}"
# #         return result

# #     # ── Record the trade ──────────────────────────────────
# #     daily_limits.record_trade_opened(ticket, session, now)

# #     logger.log_trade_opened(
# #         ticket=ticket,
# #         direction=decision["action"],
# #         lot=final_lot,
# #         entry_price=decision["entry_price"],
# #         sl_price=decision["sl_price"],
# #         tp_price=decision["tp_price"],
# #         source=decision["source"],
# #         regime=regime,
# #         reason=decision["reason"],
# #     )

# #     notifier.notify_trade_opened(
# #         ticket=ticket,
# #         direction=decision["action"],
# #         lot=final_lot,
# #         entry_price=decision["entry_price"],
# #         sl_price=decision["sl_price"],
# #         tp_price=decision["tp_price"],
# #         source=decision["source"],
# #         regime=regime,
# #     )

# #     result.update({
# #         "action": decision["action"],
# #         "ticket": ticket,
# #         "lot": final_lot,
# #         "reason": decision["reason"],
# #     })
# #     return result


# # def _resolve_decision(
# #     ad_signals: list, e_prob: float, adj: dict, regime: str,
# #     df: pd.DataFrame, idx: int, row: pd.Series, session: str,
# # ) -> dict:
# #     """
# #     Combine A-D signals + regime + E probability into a final trade decision.
# #     Mirrors the logic from run_phase9_regime.py's RegimeAwareCombinedABCDE.
# #     """
# #     skip = lambda reason: {
# #         "action": "SKIP", "reason": reason, "lot": 0,
# #         "entry_price": 0, "sl_price": 0, "tp_price": 0,
# #         "sl_pips": 0, "source": "none", "confidence": 0,
# #     }

# #     atr = float(row.get(f"atr_{ATR_PERIOD}", row.get("atr", 2.0)))

# #     # ── Get regime-adjusted E thresholds ─────────────────
# #     e_indep_thresh  = adj.get("e_indep_threshold", E_PROB_THRESHOLD)
# #     e_conf_boost    = adj.get("e_conf_boost", 0)
# #     sl_mult         = adj.get("sl_mult", 1.0)
# #     tp_mult         = adj.get("tp_mult", 1.0)
# #     allow_e_indep   = adj.get("allow_e_independent", True)
# #     e_strong_thresh = adj.get("e_val_strong_agree", E_VAL_STRONG_AGREE)
# #     e_agree_thresh  = adj.get("e_val_agree", E_VAL_AGREE)
# #     e_conflict_skip = adj.get("e_val_conflict_skip", E_VAL_CONFLICT_SKIP)

# #     current_price = float(row.get("close", 0))

# #     # ── Case 1: A-D signals exist ────────────────────────
# #     if ad_signals:
# #         buy_sigs  = [s for s in ad_signals if s["direction"] == "BUY"]
# #         sell_sigs = [s for s in ad_signals if s["direction"] == "SELL"]

# #         # Resolve direction conflict
# #         if buy_sigs and sell_sigs:
# #             buy_conf  = sum(s["confidence"] for s in buy_sigs)
# #             sell_conf = sum(s["confidence"] for s in sell_sigs)
# #             if buy_conf > sell_conf:
# #                 direction, primary = "BUY", buy_sigs
# #             elif sell_conf > buy_conf:
# #                 direction, primary = "SELL", sell_sigs
# #             else:
# #                 return skip("A-D conflict — equal confidence")
# #         elif buy_sigs:
# #             direction, primary = "BUY", buy_sigs
# #         else:
# #             direction, primary = "SELL", sell_sigs

# #         best = max(primary, key=lambda s: s["confidence"])

# #         # E validation
# #         if direction == "BUY":
# #             e_agree = e_prob >= e_agree_thresh
# #             e_strong = e_prob >= e_strong_thresh
# #             e_conflict = (1 - e_prob) >= e_conflict_skip
# #         else:
# #             e_agree = (1 - e_prob) >= e_agree_thresh
# #             e_strong = (1 - e_prob) >= e_strong_thresh
# #             e_conflict = e_prob >= e_conflict_skip

# #         if e_conflict:
# #             return skip(f"E conflicts with {direction} (prob={e_prob:.3f})")

# #         # Lot sizing based on agreement count + E boost
# #         n_agree = len(primary)
# #         if n_agree >= 2:
# #             lot = 0.02
# #         else:
# #             lot = LOT_SIZE_SINGLE

# #         # Use SL/TP from best signal
# #         sl_price = best["sl_price"]
# #         tp_price = best["tp_price"]
# #         entry    = best["entry_price"] if best["entry_price"] > 0 else current_price

# #         # Apply regime SL/TP multipliers
# #         if sl_mult != 1.0:
# #             sl_dist = abs(entry - sl_price) * sl_mult
# #             sl_price = entry - sl_dist if direction == "BUY" else entry + sl_dist

# #         if tp_mult != 1.0:
# #             tp_dist = abs(tp_price - entry) * tp_mult
# #             tp_price = entry + tp_dist if direction == "BUY" else entry - tp_dist

# #         sl_pips = abs(entry - sl_price) / PIP_VALUE
# #         source  = "+".join(s["strategy"] for s in primary) + "+E_val"
# #         conf    = best["confidence"] + (e_conf_boost if e_agree else 0)

# #         return {
# #             "action": direction,
# #             "lot": lot,
# #             "entry_price": round(entry, 2),
# #             "sl_price": round(sl_price, 2),
# #             "tp_price": round(tp_price, 2),
# #             "sl_pips": round(sl_pips, 1),
# #             "source": source,
# #             "confidence": conf,
# #             "reason": (
# #                 f"{source} conf={conf:.0f}% e_prob={e_prob:.3f} "
# #                 f"regime={regime} n={n_agree}"
# #             ),
# #         }

# #     # ── Case 2: No A-D signal — E independent ────────────
# #     if not allow_e_indep:
# #         return skip(f"E independent blocked by regime={regime}")

# #     # BUY if prob >= threshold, SELL if (1-prob) >= threshold
# #     if e_prob >= e_indep_thresh:
# #         direction = "BUY"
# #         e_conf = e_prob
# #     elif (1 - e_prob) >= e_indep_thresh:
# #         direction = "SELL"
# #         e_conf = 1 - e_prob
# #     else:
# #         return skip(f"E no signal (prob={e_prob:.3f} thresh={e_indep_thresh:.2f})")

# #     # E independent: fixed SL/TP from ATR
# #     sl_atr = atr * E_SL_ATR_MULT * sl_mult
# #     tp_atr = sl_atr * E_TP_RR_HIGH if e_conf >= 0.75 else sl_atr * E_TP_RR_BASE
# #     tp_atr *= tp_mult

# #     sl_pips = sl_atr / PIP_VALUE
# #     entry   = current_price

# #     if direction == "BUY":
# #         sl_price = entry - sl_atr
# #         tp_price = entry + tp_atr
# #     else:
# #         sl_price = entry + sl_atr
# #         tp_price = entry - tp_atr

# #     return {
# #         "action": direction,
# #         "lot": LOT_SIZE_SINGLE,
# #         "entry_price": round(entry, 2),
# #         "sl_price": round(sl_price, 2),
# #         "tp_price": round(tp_price, 2),
# #         "sl_pips": round(sl_pips, 1),
# #         "source": "E_independent",
# #         "confidence": round(e_conf * 100, 1),
# #         "reason": (
# #             f"E_indep {direction} prob={e_prob:.3f} "
# #             f"regime={regime} sl={sl_pips:.0f}pips"
# #         ),
# #     }


# # # ─── Closed Trade Monitor ────────────────────────────────

# # def check_closed_trades(
# #     executor: Executor,
# #     daily_limits: DailyLimits,
# #     session_start_ts: int,
# #     already_processed: set,
# # ) -> set:
# #     """
# #     Check for any trades that closed since session start.
# #     Updates daily limits and sends notifications.
# #     Returns updated set of processed deal tickets.
# #     """
# #     deals = executor.get_deals_since(session_start_ts)
# #     for deal in deals:
# #         deal_ticket = deal["deal_ticket"]
# #         if deal_ticket in already_processed:
# #             continue

# #         daily_limits.record_trade_closed(deal["ticket"], deal["profit"])

# #         logger.log_trade_closed(
# #             ticket=deal["ticket"],
# #             direction=deal["direction"],
# #             lot=deal["lot"],
# #             entry_price=0,  # Not available from deals, logged at open
# #             close_price=deal["price"],
# #             pnl=deal["profit"],
# #             close_reason=deal.get("comment", "TP/SL/Manual"),
# #         )

# #         notifier.notify_trade_closed(
# #             ticket=deal["ticket"],
# #             direction=deal["direction"],
# #             lot=deal["lot"],
# #             entry_price=0,
# #             close_price=deal["price"],
# #             pnl=deal["profit"],
# #             close_reason=deal.get("comment", "TP/SL/Manual"),
# #             daily_pnl=daily_limits.daily_pnl,
# #         )

# #         already_processed.add(deal_ticket)

# #     return already_processed


# # # ─── Main Loop ───────────────────────────────────────────

# # def run(dry_run: bool = False):
# #     logger.log_startup({
# #         "mode": "DRY RUN" if dry_run else "LIVE",
# #         "symbol": MT5_SYMBOL,
# #         "strategies": "A+B+C+D+E+Regime",
# #         "e_threshold": E_PROB_THRESHOLD,
# #     })

# #     # ── Initialize components ─────────────────────────────
# #     executor = Executor()
# #     if not executor.connect():
# #         logger.error("Cannot connect to MT5 — aborting")
# #         return

# #     logger.info("Loading feature engine and strategies...")
# #     feature_engine = FeatureEngine()
# #     strategies = {
# #         "A": StrategyA(),
# #         "B": StrategyB(),
# #         "C": StrategyC(),
# #         "D": StrategyD(),
# #     }

# #     logger.info("Loading Strategy E model...")
# #     try:
# #         strategy_e = StrategyELive()
# #         logger.info(f"Strategy E loaded: {E_MODEL_PATH.name}")
# #     except FileNotFoundError as e:
# #         logger.error(str(e))
# #         logger.info("Running in A+B+C+D only mode (no Strategy E)")
# #         strategy_e = None

# #     regime_detector = RegimeDetector()
# #     daily_limits = DailyLimits()
# #     risk_engine = RiskEngine()

# #     notifier.notify_startup("DRY RUN" if dry_run else "LIVE")
# #     logger.info("All components ready — waiting for first M5 bar...")

# #     session_start_ts = int(datetime.now(UTC).timestamp())
# #     processed_deals: set = set()
# #     consecutive_errors = 0

# #     # ── Timer state ───────────────────────────────────────
# #     bot_start_time          = datetime.now(UTC)
# #     last_heartbeat          = datetime.now(UTC)
# #     last_position_update    = datetime.now(UTC)
# #     HEARTBEAT_INTERVAL_SECS = 30 * 60        # 30 minutes
# #     POSITION_UPDATE_SECS    =  5 * 60        # 5 minutes (aligns with M5 bar)

# #     # Daily loss warning levels as % of MAX_DAILY_LOSS ($100)
# #     # Resets each day alongside daily_limits
# #     _warned_pct: set = set()    # tracks which % levels already sent today

# #     # ── Main loop ─────────────────────────────────────────
# #     while True:
# #         try:
# #             # Wait for next closed M5 bar
# #             new_bar_time = executor.wait_for_new_bar("M5", poll_seconds=5.0)
# #             logger.debug(f"New M5 bar: {new_bar_time}")

# #             # Check for closed trades first
# #             processed_deals = check_closed_trades(
# #                 executor, daily_limits, session_start_ts, processed_deals
# #             )

# #             # Compute features
# #             df, current_row, idx = prepare_features(executor, feature_engine)
# #             if df is None:
# #                 logger.warning("Feature computation failed — skipping bar")
# #                 continue

# #             # Skip if Strategy E not loaded (no independent signals)
# #             effective_e = strategy_e  # None is handled gracefully in _resolve_decision

# #             # Process bar
# #             result = process_bar(
# #                 df=df,
# #                 idx=idx,
# #                 current_row=current_row,
# #                 strategies=strategies,
# #                 strategy_e=effective_e,
# #                 regime_detector=regime_detector,
# #                 daily_limits=daily_limits,
# #                 risk_engine=risk_engine,
# #                 executor=executor,
# #                 dry_run=dry_run,
# #             )

# #             # ── Daily rollover ────────────────────────────
# #             if daily_limits.check_day_rollover():
# #                 logger.info("New trading day — daily limits reset")
# #                 _warned_pct.clear()
# #                 notifier.notify_daily_summary(
# #                     date.today().isoformat(),
# #                     daily_limits.daily_pnl,
# #                     daily_limits.trade_count,
# #                     daily_limits.win_count,
# #                     executor.get_account_info().get("balance", 0),
# #                 )

# #             # ── Daily rollover also resets warning set ────
# #             if daily_limits.check_day_rollover():
# #                 _warned_pct.clear()

# #             consecutive_errors = 0

# #             if result["action"] != "SKIP":
# #                 logger.info(
# #                     f"BAR PROCESSED | action={result['action']} "
# #                     f"ticket={result.get('ticket')} "
# #                     f"session={result['session']}"
# #                 )

# #             now_ts = datetime.now(UTC)

# #             # ── Heartbeat (every 30 min) ──────────────────
# #             if (now_ts - last_heartbeat).total_seconds() >= HEARTBEAT_INTERVAL_SECS:
# #                 open_pos  = executor.get_open_positions()
# #                 acct      = executor.get_account_info()
# #                 uptime    = int((now_ts - bot_start_time).total_seconds() / 60)
# #                 session   = get_current_session(now_ts)
# #                 regime    = "RANGING"
# #                 try:
# #                     df_hb, row_hb, _ = prepare_features(executor, feature_engine)
# #                     if row_hb is not None:
# #                         regime, _ = regime_detector.detect_and_adjust(row_hb)
# #                 except Exception:
# #                     pass
# #                 notifier.notify_heartbeat(
# #                     uptime_mins=uptime,
# #                     daily_pnl=daily_limits.daily_pnl,
# #                     open_trades=len(open_pos),
# #                     trade_count=daily_limits.trade_count,
# #                     balance=acct.get("balance", 0),
# #                     session=session,
# #                     regime=regime,
# #                 )
# #                 last_heartbeat = now_ts

# #             # ── Open position update (every 5 min) ────────
# #             if (now_ts - last_position_update).total_seconds() >= POSITION_UPDATE_SECS:
# #                 open_pos = executor.get_open_positions()
# #                 if open_pos:
# #                     notifier.notify_position_update(open_pos, daily_limits.daily_pnl)
# #                 last_position_update = now_ts

# #             # ── Daily loss warnings (50% / 75% / 100%) ───
# #             from config import MAX_DAILY_LOSS as _MAX_DAILY_LOSS
# #             dpnl = daily_limits.daily_pnl
# #             for pct_level in [50, 75, 100]:
# #                 if pct_level in _warned_pct:
# #                     continue
# #                 threshold = -(_MAX_DAILY_LOSS * pct_level / 100)
# #                 if dpnl <= threshold:
# #                     notifier.notify_daily_limit_warning(
# #                         daily_pnl=dpnl,
# #                         limit=_MAX_DAILY_LOSS,
# #                         pct_used=pct_level,
# #                         trades_today=daily_limits.trade_count,
# #                     )
# #                     _warned_pct.add(pct_level)
# #                     logger.warning(
# #                         f"Daily loss {pct_level}% warning — "
# #                         f"${dpnl:.2f} of -${_MAX_DAILY_LOSS} limit"
# #                     )

# #         except KeyboardInterrupt:
# #             logger.info("Keyboard interrupt — shutting down")
# #             notifier.notify_shutdown("Keyboard interrupt")
# #             break

# #         except Exception as e:
# #             consecutive_errors += 1
# #             logger.log_error("main_loop", e)
# #             logger.error(traceback.format_exc())

# #             if consecutive_errors >= 5:
# #                 notifier.notify_error("main_loop", f"5 consecutive errors — restarting MT5")
# #                 executor.reconnect()
# #                 consecutive_errors = 0
# #             else:
# #                 time.sleep(10)

# #     executor.disconnect()
# #     logger.info("MIDAS v2 stopped.")


# # # ─── Entry Point ─────────────────────────────────────────

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="MIDAS v2 Live Trader")
# #     parser.add_argument(
# #         "--dry-run", action="store_true",
# #         help="Run signals only — no orders sent to MT5"
# #     )
# #     parser.add_argument(
# #         "--check", action="store_true",
# #         help="Check MT5 connection and exit"
# #     )
# #     args = parser.parse_args()

# #     if args.check:
# #         ex = Executor()
# #         if ex.connect():
# #             info = ex.get_account_info()
# #             print(f"MT5 OK | Balance: ${info['balance']:.2f} | Equity: ${info['equity']:.2f}")
# #             ex.disconnect()
# #         else:
# #             print("MT5 connection FAILED")
# #         sys.exit(0)

# #     run(dry_run=args.dry_run)
# #     run(dry_run=args.dry_run)

# """
# Project MIDAS v2 — Main Live Trading Loop
# ==========================================
# Connects all components and runs the A+B+C+D+E+Regime strategy live on MT5.

# Architecture per M5 bar:
#   1. Fetch latest candles (M5 + M15) from MT5
#   2. Compute 167 features (FeatureEngine)
#   3. Detect market regime (RegimeDetector)
#   4. Run strategies A, B, C, D → collect signals
#   5. Combine with Strategy E (validator + independent)
#   6. Apply regime adjustments to E thresholds
#   7. RiskEngine final check
#   8. Execute via MT5 (Executor)
#   9. Monitor for closed trades → update daily P&L
#   10. Log + Telegram notify

# Usage:
#     python main.py               # Live trading
#     python main.py --dry-run     # Dry run (signals only, no orders)
#     python main.py --check       # Check MT5 connection and exit
# """

# import argparse
# import sys
# import time
# import json
# import traceback
# from datetime import datetime, date, timezone
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import lightgbm as lgb

# sys.path.insert(0, str(Path(__file__).parent))

# from config import (
#     MT5_SYMBOL, MODELS_DIR, DATA_PROCESSED,
#     ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE,
#     LOT_SIZE_MIN, LOT_SIZE_MAX, LOT_SIZE_SINGLE,
#     SESSIONS,
# )
# from feature_engineering import FeatureEngine
# from strategies.strategy_a import StrategyA
# from strategies.strategy_b import StrategyB
# from strategies.strategy_c import StrategyC
# from strategies.strategy_d import StrategyD
# from regime_detector import RegimeDetector, REGIME_ADJUSTMENTS

# import logger
# import notifier
# from daily_limits import DailyLimits
# from risk_engine import RiskEngine
# from executor import Executor
# from tsl_manager import TSLManager

# UTC = timezone.utc


# # ─── Strategy E Config (debiased best threshold) ─────────
# E_PROB_THRESHOLD       = 0.68
# E_VAL_STRONG_AGREE     = 0.60
# E_VAL_AGREE            = 0.52
# E_VAL_CONFLICT_SKIP    = 0.60
# E_STRONG_BOOST         = 10
# E_WEAK_PENALTY         = -10
# E_SL_ATR_MULT          = 1.5
# E_TP_RR_BASE           = 1.5
# E_TP_RR_HIGH           = 2.5
# E_INDEPENDENT_MIN_CONF = 70

# E_MODEL_PATH    = MODELS_DIR / "strategy_e_clean_final.txt"
# E_METADATA_PATH = MODELS_DIR / "strategy_e_clean_metadata.json"

# M5_LOOKBACK  = 500
# M15_LOOKBACK = 200


# # ─── Session Detection ───────────────────────────────────

# def get_current_session(dt: datetime) -> str:
#     hour = dt.hour
#     if 12 <= hour < 14:
#         return "london_ny_overlap"
#     if 6 <= hour < 14:
#         return "london"
#     if 14 <= hour < 20:
#         return "new_york"
#     if 0 <= hour < 6:
#         return "asian"
#     return "off"


# # ─── Feature Preparation ─────────────────────────────────

# def prepare_features(executor: Executor, feature_engine: FeatureEngine) -> tuple:
#     df_m5  = executor.get_candles("M5",  M5_LOOKBACK)
#     df_m15 = executor.get_candles("M15", M15_LOOKBACK)

#     if df_m5 is None or df_m15 is None:
#         logger.error("Failed to fetch candles from MT5")
#         return None, None, None

#     if len(df_m5) < 250:
#         logger.warning(f"Not enough M5 candles: {len(df_m5)} (need 250+)")
#         return None, None, None

#     try:
#         df = feature_engine.compute(df_m5, df_m15)
#         if df is None or df.empty:
#             logger.error("FeatureEngine returned empty dataframe")
#             return None, None, None
#         idx = len(df) - 1
#         return df, df.iloc[idx], idx
#     except Exception as e:
#         logger.log_error("prepare_features", e)
#         return None, None, None


# # ─── Strategy E Inference ────────────────────────────────

# class StrategyELive:
#     def __init__(self):
#         if not E_MODEL_PATH.exists():
#             raise FileNotFoundError(
#                 f"Strategy E model not found: {E_MODEL_PATH}\n"
#                 "Run strategy_e_clean_retrain.py on Colab first."
#             )
#         self.model = lgb.Booster(model_file=str(E_MODEL_PATH))
#         with open(E_METADATA_PATH) as f:
#             self.feature_cols = json.load(f)["feature_cols"]

#     def predict(self, row: pd.Series) -> float:
#         try:
#             feat = row[self.feature_cols].values.reshape(1, -1)
#             return float(self.model.predict(feat)[0])
#         except Exception as e:
#             logger.log_error("StrategyELive.predict", e)
#             return 0.5


# # ─── Signal Resolution ───────────────────────────────────

# def _resolve_decision(
#     ad_signals: list,
#     e_prob: float,
#     adj: dict,
#     regime: str,
#     row: pd.Series,
# ) -> dict:
#     def skip(reason):
#         return {
#             "action": "SKIP", "reason": reason, "lot": 0,
#             "entry_price": 0, "sl_price": 0, "tp_price": 0,
#             "sl_pips": 0, "source": "none", "confidence": 0,
#         }

#     atr           = float(row.get(f"atr_{ATR_PERIOD}", row.get("atr", 2.0)))
#     current_price = float(row.get("close", 0))

#     e_indep_thresh  = adj.get("e_indep_threshold",   E_PROB_THRESHOLD)
#     e_conf_boost    = adj.get("e_conf_boost",         0)
#     sl_mult         = adj.get("sl_mult",              1.0)
#     tp_mult         = adj.get("tp_mult",              1.0)
#     allow_e_indep   = adj.get("allow_e_independent",  True)
#     e_agree_thresh  = adj.get("e_val_agree",          E_VAL_AGREE)
#     e_conflict_skip = adj.get("e_val_conflict_skip",  E_VAL_CONFLICT_SKIP)

#     # ── Case 1: A-D signals exist ────────────────────────
#     if ad_signals:
#         buy_sigs  = [s for s in ad_signals if s["direction"] == "BUY"]
#         sell_sigs = [s for s in ad_signals if s["direction"] == "SELL"]

#         if buy_sigs and sell_sigs:
#             buy_conf  = sum(s["confidence"] for s in buy_sigs)
#             sell_conf = sum(s["confidence"] for s in sell_sigs)
#             if buy_conf > sell_conf:
#                 direction, primary = "BUY",  buy_sigs
#             elif sell_conf > buy_conf:
#                 direction, primary = "SELL", sell_sigs
#             else:
#                 return skip("A-D conflict — equal confidence")
#         elif buy_sigs:
#             direction, primary = "BUY",  buy_sigs
#         else:
#             direction, primary = "SELL", sell_sigs

#         best = max(primary, key=lambda s: s["confidence"])

#         if direction == "BUY":
#             e_agree    = e_prob >= e_agree_thresh
#             e_conflict = (1 - e_prob) >= e_conflict_skip
#         else:
#             e_agree    = (1 - e_prob) >= e_agree_thresh
#             e_conflict = e_prob >= e_conflict_skip

#         if e_conflict:
#             return skip(f"E conflicts with {direction} (prob={e_prob:.3f})")

#         n_agree  = len(primary)
#         lot      = 0.02 if n_agree >= 2 else LOT_SIZE_SINGLE
#         sl_price = best["sl_price"]
#         tp_price = best["tp_price"]
#         entry    = best["entry_price"] if best["entry_price"] > 0 else current_price

#         if sl_mult != 1.0:
#             sl_dist  = abs(entry - sl_price) * sl_mult
#             sl_price = (entry - sl_dist) if direction == "BUY" else (entry + sl_dist)
#         if tp_mult != 1.0:
#             tp_dist  = abs(tp_price - entry) * tp_mult
#             tp_price = (entry + tp_dist) if direction == "BUY" else (entry - tp_dist)

#         sl_pips = abs(entry - sl_price) / PIP_VALUE
#         source  = "+".join(s["strategy"] for s in primary) + "+E_val"
#         conf    = best["confidence"] + (e_conf_boost if e_agree else 0)

#         return {
#             "action":      direction,
#             "lot":         lot,
#             "entry_price": round(entry,    2),
#             "sl_price":    round(sl_price, 2),
#             "tp_price":    round(tp_price, 2),
#             "sl_pips":     round(sl_pips,  1),
#             "source":      source,
#             "confidence":  conf,
#             "reason":      f"{source} conf={conf:.0f}% e_prob={e_prob:.3f} regime={regime} n={n_agree}",
#         }

#     # ── Case 2: E independent ─────────────────────────────
#     if not allow_e_indep:
#         return skip(f"E independent blocked by regime={regime}")

#     if e_prob >= e_indep_thresh:
#         direction, e_conf = "BUY",  e_prob
#     elif (1 - e_prob) >= e_indep_thresh:
#         direction, e_conf = "SELL", 1 - e_prob
#     else:
#         return skip(f"E no signal (prob={e_prob:.3f} thresh={e_indep_thresh:.2f})")

#     sl_atr = atr * E_SL_ATR_MULT * sl_mult
#     tp_atr = (sl_atr * E_TP_RR_HIGH if e_conf >= 0.75 else sl_atr * E_TP_RR_BASE) * tp_mult
#     entry  = current_price

#     if direction == "BUY":
#         sl_price, tp_price = entry - sl_atr, entry + tp_atr
#     else:
#         sl_price, tp_price = entry + sl_atr, entry - tp_atr

#     return {
#         "action":      direction,
#         "lot":         LOT_SIZE_SINGLE,
#         "entry_price": round(entry,    2),
#         "sl_price":    round(sl_price, 2),
#         "tp_price":    round(tp_price, 2),
#         "sl_pips":     round(sl_atr / PIP_VALUE, 1),
#         "source":      "E_independent",
#         "confidence":  round(e_conf * 100, 1),
#         "reason":      f"E_indep {direction} prob={e_prob:.3f} regime={regime} sl={sl_atr/PIP_VALUE:.0f}pips",
#     }


# # ─── Bar Processing ──────────────────────────────────────

# def process_bar(
#     df: pd.DataFrame,
#     idx: int,
#     current_row: pd.Series,
#     strategies: dict,
#     strategy_e,
#     regime_detector: RegimeDetector,
#     daily_limits: DailyLimits,
#     risk_engine: RiskEngine,
#     executor: Executor,
#     tsl_manager: TSLManager,
#     dry_run: bool = False,
# ) -> dict:
#     now     = datetime.now(UTC)
#     session = get_current_session(now)

#     result = {
#         "bar_time": str(current_row.get("datetime", now)),
#         "session":  session,
#         "action":   "SKIP",
#         "reason":   "",
#         "ticket":   None,
#     }

#     # Pre-flight risk check
#     can_trade, block_reason = daily_limits.can_trade(session, now)
#     if not can_trade:
#         result["reason"] = block_reason
#         logger.debug(f"Skipping — {block_reason}")
#         return result

#     # Strategies A-D
#     ad_signals = []
#     for name, strat in strategies.items():
#         try:
#             signal = strat.generate_signal(df, idx)
#             if signal is not None:
#                 ad_signals.append({
#                     "strategy":    name,
#                     "direction":   signal.direction.upper(),
#                     "confidence":  signal.confidence,
#                     "sl_price":    signal.sl_price,
#                     "tp_price":    signal.tp_price,
#                     "entry_price": signal.entry_price,
#                 })
#         except Exception as e:
#             logger.log_error(f"strategy_{name}", e)

#     # Regime
#     regime, adj = "RANGING", REGIME_ADJUSTMENTS.get("RANGING", {})
#     try:
#         regime, adj = regime_detector.detect_and_adjust(current_row)
#     except Exception as e:
#         logger.log_error("regime_detector", e)

#     # Strategy E
#     e_prob = 0.5
#     if strategy_e is not None:
#         try:
#             e_prob = strategy_e.predict(current_row)
#         except Exception as e:
#             logger.log_error("strategy_e.predict", e)

#     # Final decision
#     decision = _resolve_decision(ad_signals, e_prob, adj, regime, current_row)

#     if decision["action"] == "SKIP":
#         result["reason"] = decision["reason"]
#         logger.debug(f"{session} | regime={regime} | SKIP: {decision['reason']}")
#         return result

#     logger.log_signal(
#         bar_time    = current_row.get("datetime", now),
#         strategy    = decision["source"],
#         direction   = decision["action"],
#         confidence  = decision.get("confidence", 0),
#         regime      = regime,
#         action_taken= "DRY_RUN" if dry_run else "EXECUTE",
#         reason      = decision["reason"],
#     )

#     if dry_run:
#         result.update({"action": decision["action"], "reason": "DRY_RUN — no order sent"})
#         logger.info(f"[DRY RUN] {decision['action']} lot={decision['lot']} src={decision['source']} | {decision['reason']}")
#         return result

#     # Risk engine
#     account = executor.get_account_info()
#     approved, risk_msg, final_lot = risk_engine.check_order(
#         direction      = decision["action"],
#         lot            = decision["lot"],
#         sl_pips        = decision["sl_pips"],
#         daily_pnl      = daily_limits.daily_pnl,
#         account_balance= account.get("balance", 5000),
#         account_equity = account.get("equity",  5000),
#     )
#     if not approved:
#         logger.log_risk_block(risk_msg, decision["source"])
#         notifier.notify_risk_block(risk_msg)
#         result["reason"] = f"RISK BLOCK: {risk_msg}"
#         return result

#     sl_valid, sl_msg = risk_engine.validate_sl_tp(
#         decision["action"], decision["entry_price"],
#         decision["sl_price"], decision["tp_price"],
#     )
#     if not sl_valid:
#         logger.log_risk_block(sl_msg, decision["source"])
#         result["reason"] = f"SL/TP INVALID: {sl_msg}"
#         return result

#     # Place order
#     success, ticket, msg = executor.place_order(
#         direction = decision["action"],
#         lot       = final_lot,
#         sl_price  = decision["sl_price"],
#         tp_price  = decision["tp_price"],
#         comment   = f"MIDAS_{decision['source'][:10]}",
#     )
#     if not success:
#         logger.error(f"Order failed: {msg}")
#         notifier.notify_error("place_order", msg)
#         result["reason"] = f"ORDER FAILED: {msg}"
#         return result

#     # Record + register
#     daily_limits.record_trade_opened(ticket, session, now)
#     tsl_manager.register(
#         ticket      = ticket,
#         direction   = decision["action"],
#         entry_price = decision["entry_price"],
#         sl_price    = decision["sl_price"],
#         tp_price    = decision["tp_price"],
#     )

#     logger.log_trade_opened(
#         ticket      = ticket,
#         direction   = decision["action"],
#         lot         = final_lot,
#         entry_price = decision["entry_price"],
#         sl_price    = decision["sl_price"],
#         tp_price    = decision["tp_price"],
#         source      = decision["source"],
#         regime      = regime,
#         reason      = decision["reason"],
#     )
#     notifier.notify_trade_opened(
#         ticket      = ticket,
#         direction   = decision["action"],
#         lot         = final_lot,
#         entry_price = decision["entry_price"],
#         sl_price    = decision["sl_price"],
#         tp_price    = decision["tp_price"],
#         source      = decision["source"],
#         regime      = regime,
#     )

#     result.update({"action": decision["action"], "ticket": ticket,
#                    "lot": final_lot, "reason": decision["reason"]})
#     return result


# # ─── Closed Trade Monitor ────────────────────────────────

# def check_closed_trades(
#     executor: Executor,
#     daily_limits: DailyLimits,
#     tsl_manager: TSLManager,
#     session_start_ts: int,
#     already_processed: set,
# ) -> set:
#     deals = executor.get_deals_since(session_start_ts)
#     for deal in deals:
#         if deal["deal_ticket"] in already_processed:
#             continue
#         daily_limits.record_trade_closed(deal["ticket"], deal["profit"])
#         tsl_manager.unregister(deal["ticket"])
#         logger.log_trade_closed(
#             ticket       = deal["ticket"],
#             direction    = deal["direction"],
#             lot          = deal["lot"],
#             entry_price  = 0,
#             close_price  = deal["price"],
#             pnl          = deal["profit"],
#             close_reason = deal.get("comment", "TP/SL/Manual"),
#         )
#         notifier.notify_trade_closed(
#             ticket       = deal["ticket"],
#             direction    = deal["direction"],
#             lot          = deal["lot"],
#             entry_price  = 0,
#             close_price  = deal["price"],
#             pnl          = deal["profit"],
#             close_reason = deal.get("comment", "TP/SL/Manual"),
#             daily_pnl    = daily_limits.daily_pnl,
#         )
#         already_processed.add(deal["deal_ticket"])
#     return already_processed


# # ─── Main Loop ───────────────────────────────────────────

# def run(dry_run: bool = False):
#     from config import MAX_DAILY_LOSS

#     logger.log_startup({
#         "mode":        "DRY RUN" if dry_run else "LIVE",
#         "symbol":      MT5_SYMBOL,
#         "strategies":  "A+B+C+D+E+Regime",
#         "e_threshold": E_PROB_THRESHOLD,
#     })

#     executor = Executor()
#     if not executor.connect():
#         logger.error("Cannot connect to MT5 — aborting")
#         return

#     feature_engine = FeatureEngine()
#     strategies = {"A": StrategyA(), "B": StrategyB(), "C": StrategyC(), "D": StrategyD()}

#     try:
#         strategy_e = StrategyELive()
#         logger.info(f"Strategy E loaded: {E_MODEL_PATH.name}")
#     except FileNotFoundError as e:
#         logger.error(str(e))
#         logger.info("Running A+B+C+D only mode")
#         strategy_e = None

#     regime_detector = RegimeDetector()
#     daily_limits    = DailyLimits()
#     risk_engine     = RiskEngine()
#     tsl_manager     = TSLManager()

#     notifier.notify_startup("DRY RUN" if dry_run else "LIVE")
#     logger.info("All components ready — waiting for first M5 bar...")

#     session_start_ts     = int(datetime.now(UTC).timestamp())
#     processed_deals: set = set()
#     consecutive_errors   = 0

#     bot_start_time       = datetime.now(UTC)
#     last_heartbeat       = datetime.now(UTC)
#     last_position_update = datetime.now(UTC)
#     last_tsl_update      = datetime.now(UTC)

#     HEARTBEAT_SECS       = 30 * 60
#     POSITION_UPDATE_SECS =  5 * 60
#     TSL_UPDATE_SECS      =       30

#     warned_pct: set = set()

#     while True:
#         try:
#             executor.wait_for_new_bar("M5", poll_seconds=5.0)
#             now_ts = datetime.now(UTC)

#             # Daily rollover — single call
#             if daily_limits.check_day_rollover():
#                 logger.info("New trading day — limits reset")
#                 warned_pct.clear()
#                 notifier.notify_daily_summary(
#                     date_str  = date.today().isoformat(),
#                     total_pnl = daily_limits.daily_pnl,
#                     trades    = daily_limits.trade_count,
#                     wins      = daily_limits.win_count,
#                     balance   = executor.get_account_info().get("balance", 0),
#                 )

#             # Closed trades
#             processed_deals = check_closed_trades(
#                 executor, daily_limits, tsl_manager,
#                 session_start_ts, processed_deals,
#             )

#             # Features
#             df, current_row, idx = prepare_features(executor, feature_engine)
#             if df is None:
#                 logger.warning("Feature computation failed — skipping bar")
#                 continue

#             # Process bar
#             result = process_bar(
#                 df=df, idx=idx, current_row=current_row,
#                 strategies=strategies, strategy_e=strategy_e,
#                 regime_detector=regime_detector, daily_limits=daily_limits,
#                 risk_engine=risk_engine, executor=executor,
#                 tsl_manager=tsl_manager, dry_run=dry_run,
#             )

#             if result["action"] != "SKIP":
#                 logger.info(
#                     f"BAR PROCESSED | action={result['action']} "
#                     f"ticket={result.get('ticket')} session={result['session']}"
#                 )

#             consecutive_errors = 0

#             # TSL + Dynamic TP (every 30 sec)
#             if (now_ts - last_tsl_update).total_seconds() >= TSL_UPDATE_SECS:
#                 for m in tsl_manager.update(executor):
#                     if m["success"]:
#                         logger.info(
#                             f"TSL/TP | #{m['ticket']} {m['direction']} type={m['type']} "
#                             f"price={m['current_price']:.2f} | "
#                             f"SL {m['old_sl']:.2f}→{m['new_sl']:.2f} | "
#                             f"TP {m['old_tp']:.2f}→{m['new_tp']:.2f}"
#                         )
#                     else:
#                         logger.warning(f"TSL/TP FAILED | #{m['ticket']} | {m['message']}")
#                 last_tsl_update = now_ts

#             # Position update (every 5 min)
#             if (now_ts - last_position_update).total_seconds() >= POSITION_UPDATE_SECS:
#                 open_pos = executor.get_open_positions()
#                 if open_pos:
#                     notifier.notify_position_update(open_pos, daily_limits.daily_pnl)
#                 last_position_update = now_ts

#             # Heartbeat (every 30 min)
#             if (now_ts - last_heartbeat).total_seconds() >= HEARTBEAT_SECS:
#                 open_pos = executor.get_open_positions()
#                 acct     = executor.get_account_info()
#                 uptime   = int((now_ts - bot_start_time).total_seconds() / 60)
#                 session  = get_current_session(now_ts)
#                 regime   = "RANGING"
#                 try:
#                     _, row_hb, _ = prepare_features(executor, feature_engine)
#                     if row_hb is not None:
#                         regime, _ = regime_detector.detect_and_adjust(row_hb)
#                 except Exception:
#                     pass
#                 notifier.notify_heartbeat(
#                     uptime_mins=uptime, daily_pnl=daily_limits.daily_pnl,
#                     open_trades=len(open_pos), trade_count=daily_limits.trade_count,
#                     balance=acct.get("balance", 0), session=session, regime=regime,
#                 )
#                 last_heartbeat = now_ts

#             # Daily loss warnings
#             dpnl = daily_limits.daily_pnl
#             for pct in [50, 75, 100]:
#                 if pct not in warned_pct and dpnl <= -(MAX_DAILY_LOSS * pct / 100):
#                     notifier.notify_daily_limit_warning(
#                         daily_pnl=dpnl, limit=MAX_DAILY_LOSS,
#                         pct_used=pct, trades_today=daily_limits.trade_count,
#                     )
#                     warned_pct.add(pct)
#                     logger.warning(f"Daily loss {pct}% — ${dpnl:.2f} of -${MAX_DAILY_LOSS}")

#         except KeyboardInterrupt:
#             logger.info("Keyboard interrupt — shutting down")
#             notifier.notify_shutdown("Keyboard interrupt")
#             break

#         except Exception as e:
#             consecutive_errors += 1
#             logger.log_error("main_loop", e)
#             logger.error(traceback.format_exc())
#             if consecutive_errors >= 5:
#                 notifier.notify_error("main_loop", "5 consecutive errors — reconnecting MT5")
#                 executor.reconnect()
#                 consecutive_errors = 0
#             else:
#                 time.sleep(10)

#     executor.disconnect()
#     logger.info("MIDAS v2 stopped.")


# # ─── Entry Point ─────────────────────────────────────────

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="MIDAS v2 Live Trader")
#     parser.add_argument("--dry-run", action="store_true",
#                         help="Signals only — no orders sent to MT5")
#     parser.add_argument("--check", action="store_true",
#                         help="Check MT5 connection and exit")
#     args = parser.parse_args()

#     if args.check:
#         ex = Executor()
#         if ex.connect():
#             info = ex.get_account_info()
#             print(f"MT5 OK | Balance: ${info['balance']:.2f} | Equity: ${info['equity']:.2f}")
#             ex.disconnect()
#         else:
#             print("MT5 connection FAILED")
#         sys.exit(0)

#     run(dry_run=args.dry_run)




# """
# Project MIDAS v2 — Main Live Trading Loop
# ===========================================
# Full system: A+B+C+D+E + Regime + Dynamic Lots

# Loop:
#   1. Wait for M5 bar close
#   2. Fetch M5 + M15 data from MT5
#   3. Compute all features
#   4. Merge M15 into M5
#   5. Compute Strategy E prediction (debiased)
#   6. Run CombinedABCDE strategy (with regime + dynamic lots)
#   7. If signal passes risk checks → execute on MT5
#   8. Monitor open positions
#   9. Log + notify

# Usage:
#     python main.py              # Live trading
#     python main.py --dry-run    # No real trades, just signals
#     python main.py --mode demo  # Demo account mode (default)

# Press Ctrl+C to stop gracefully.
# """

# import argparse
# import sys
# import time
# import traceback
# from datetime import datetime, timedelta, UTC
# from pathlib import Path

# import numpy as np
# import pandas as pd

# sys.path.insert(0, str(Path(__file__).parent))

# from config import (
#     ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE, MT5_SYMBOL,
#     SPREAD_SIMULATION_PIPS, LOT_SIZE_SINGLE,
#     MAX_TRADE_RISK, CONFIDENCE_THRESHOLD, SESSIONS,
# )
# from feature_engineering import FeatureEngine
# from mt5_executor import MT5Executor
# from risk_manager import RiskManager
# from trade_logger import TradeLogger
# from telegram_notifier import TelegramNotifier
# from strategies.live_strategy_e import LiveStrategyE
# from regime_detector import RegimeDetector, REGIME_ADJUSTMENTS
# from dynamic_lot_sizer import DynamicLotSizer

# # Strategy imports
# from strategies.strategy_a import StrategyA
# from strategies.strategy_b import StrategyB
# from strategies.strategy_c import StrategyC
# from strategies.strategy_d import StrategyD
# from backtester import TradeSignal

# # ─── Configuration ────────────────────────────────────────
# M5_HISTORY_BARS = 500       # Bars to fetch for feature computation
# M15_HISTORY_BARS = 200      # M15 bars for structure detection
# LOOP_INTERVAL_SECONDS = 10  # Check for new bars every 10s
# MAX_ERRORS_BEFORE_RESTART = 5

# # Strategy E thresholds (same as run_phase8_lots.py)
# E_PROB_THRESHOLD = 0.68
# E_VAL_STRONG_AGREE = 0.60
# E_VAL_AGREE = 0.52
# E_VAL_CONFLICT_SKIP = 0.60
# E_STRONG_BOOST = 10
# E_WEAK_PENALTY = -10
# E_SL_ATR_MULT = 1.5
# E_TP_RR_BASE = 1.5
# E_TP_RR_HIGH = 2.5
# E_INDEPENDENT_MIN_CONF = 70


# class MidasBot:
#     """
#     Main trading bot. Orchestrates all components.
#     """

#     def __init__(self, dry_run: bool = False, mode: str = "demo"):
#         self.dry_run = dry_run
#         self.mode = mode

#         # Components
#         self.mt5 = MT5Executor()
#         self.risk = RiskManager()
#         self.logger = TradeLogger()
#         self.notifier = TelegramNotifier()
#         self.feature_engine = FeatureEngine()
#         self.regime_detector = RegimeDetector()
#         self.lot_sizer = DynamicLotSizer()
#         self.live_e = LiveStrategyE()

#         # Strategies A-D
#         self.strategy_a = StrategyA()
#         self.strategy_b = StrategyB()
#         self.strategy_c = StrategyC()
#         self.strategy_d = StrategyD()

#         # State
#         self.last_m5_time = None
#         self.error_count = 0
#         self.running = True

#     # ─── Initialization ───────────────────────────────────

#     def start(self):
#         """Initialize all components and start the main loop."""
#         print("=" * 60)
#         print("  MIDAS v2 — XAUUSD AI Trading Bot")
#         print("=" * 60)
#         print(f"  Mode:        {self.mode.upper()}")
#         print(f"  Dry Run:     {'YES — no real trades' if self.dry_run else 'NO — live execution'}")
#         print(f"  Symbol:      {MT5_SYMBOL}")
#         print(f"  Strategies:  A+B+C+D+E (regime-aware, dynamic lots)")

#         # Connect to MT5
#         print("\n  Connecting to MT5...")
#         if not self.mt5.connect():
#             print("  FATAL: Cannot connect to MT5")
#             return

#         # Get account info
#         acc = self.mt5.get_account_info()
#         self.risk.update_balance(acc.get("balance", 5000))

#         # Load Strategy E
#         print("\n  Loading Strategy E model...")
#         if not self.live_e.loaded:
#             print("  WARNING: Strategy E not available. Running A-D only.")

#         # Notify
#         self.notifier.bot_started(acc.get("balance", 0), self.mode.upper())

#         print(f"\n  ✅ Bot started | Balance: ${acc.get('balance', 0):,.2f}")
#         print(f"  Press Ctrl+C to stop\n")
#         print("─" * 60)

#         # Main loop
#         try:
#             self._main_loop()
#         except KeyboardInterrupt:
#             print("\n\n  Stopping bot (Ctrl+C)...")
#         except Exception as e:
#             print(f"\n  FATAL ERROR: {e}")
#             traceback.print_exc()
#             self.notifier.error(f"Fatal: {str(e)[:200]}")
#         finally:
#             self._shutdown()

#     # ─── Main Loop ────────────────────────────────────────

#     def _main_loop(self):
#         """Main trading loop — runs until stopped."""
#         while self.running:
#             try:
#                 # Check for new M5 bar
#                 latest = self.mt5.fetch_latest_bar("M5")
#                 if latest is None:
#                     time.sleep(LOOP_INTERVAL_SECONDS)
#                     continue

#                 bar_time = latest["datetime"]
#                 if isinstance(bar_time, np.datetime64):
#                     bar_time = pd.Timestamp(bar_time).to_pydatetime()

#                 # Skip if we already processed this bar
#                 if self.last_m5_time is not None and bar_time <= self.last_m5_time:
#                     # Still check open positions
#                     self._monitor_positions()
#                     time.sleep(LOOP_INTERVAL_SECONDS)
#                     continue

#                 # New bar — process it
#                 self.last_m5_time = bar_time
#                 self._process_bar(bar_time)
#                 self.error_count = 0

#             except KeyboardInterrupt:
#                 raise
#             except Exception as e:
#                 self.error_count += 1
#                 print(f"  ⚠️ Error ({self.error_count}): {e}")
#                 traceback.print_exc()

#                 if self.error_count >= MAX_ERRORS_BEFORE_RESTART:
#                     print("  Too many errors. Attempting MT5 reconnect...")
#                     self.mt5.reconnect()
#                     self.error_count = 0

#                 time.sleep(LOOP_INTERVAL_SECONDS * 2)

#     # ─── Bar Processing ───────────────────────────────────

#     def _process_bar(self, bar_time: datetime):
#         """Process a new M5 bar — the core trading logic."""
#         now = datetime.now(UTC)
#         session = self.risk.get_session(now)
#         price_info = self.mt5.get_current_price()

#         # Status line
#         hour = now.hour
#         minute = now.minute
#         sess_counts = self.risk.session_trades
#         print(f"\n  [{hour:02d}:{minute:02d}] Price: {price_info.get('bid', 0):.2f} | "
#               f"{session}({sess_counts.get(session, 0)}/{self.risk.session_trades.get(session, 0)}) | "
#               f"DayPnL: ${self.risk.daily_pnl:+.2f}")

#         # Skip if position is open
#         if self.mt5.has_open_position():
#             print(f"  Position open — monitoring...")
#             self._monitor_positions()
#             return

#         # Risk pre-flight
#         can_trade, reason = self.risk.can_trade(session, now)
#         if not can_trade:
#             print(f"  Skip: {reason}")
#             return

#         # ── Fetch and compute features ────────────────────
#         df = self._build_feature_dataframe()
#         if df is None or len(df) < 100:
#             print(f"  Skip: insufficient data ({len(df) if df is not None else 0} bars)")
#             return

#         idx = len(df) - 1  # Last completed bar

#         # ── Detect regime ─────────────────────────────────
#         row = df.iloc[idx]
#         regime, adj = self.regime_detector.detect_and_adjust(row)

#         # ── Run A-D strategies ────────────────────────────
#         signals = []
#         for name, strat in [("A", self.strategy_a), ("B", self.strategy_b),
#                              ("C", self.strategy_c), ("D", self.strategy_d)]:
#             try:
#                 sig = strat.generate_signal(df, idx)
#                 if sig is not None:
#                     signals.append((name, sig))
#             except Exception as e:
#                 print(f"  Strategy {name} error: {e}")

#         # ── Get Strategy E prediction ─────────────────────
#         e_prob = None
#         if self.live_e.loaded:
#             try:
#                 df_debiased = self.live_e.debias_dataframe(df)
#                 e_prob = self.live_e.predict(df_debiased, idx)
#             except Exception as e:
#                 print(f"  Strategy E error: {e}")

#         # ── Decision logic ────────────────────────────────
#         signal = self._make_decision(signals, e_prob, adj, regime, row)

#         if signal is None:
#             sources = ", ".join(f"{n}={s.direction}" for n, s in signals) if signals else "none"
#             e_str = f"E={e_prob:.3f}" if e_prob else "E=N/A"
#             print(f"  No trade | Signals: {sources} | {e_str} | Regime: {regime}")
#             return

#         # ── Execute ───────────────────────────────────────
#         self._execute_signal(signal, session, regime, e_prob)

#     def _make_decision(self, ad_signals, e_prob, adj, regime, row):
#         """
#         Combined decision logic: A-D + E validator + E independent.
#         Same logic as run_phase8_lots.py CombinedABCDE_DynamicLot.
#         """
#         atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
#         if pd.isna(atr) or atr <= 0:
#             atr = 2.0

#         # ── Case 1: A-D signals exist ─────────────────────
#         if ad_signals:
#             # Direction check
#             dirs = set(s.direction for _, s in ad_signals)
#             if len(dirs) > 1:
#                 return None  # Conflict

#             best_name, best_signal = max(ad_signals, key=lambda x: x[1].confidence)
#             n_agree = len(ad_signals)
#             boost = (n_agree - 1) * 10
#             best_signal.confidence = min(100, best_signal.confidence + boost)

#             sources = "+".join(n for n, _ in ad_signals)

#             # E validation
#             if e_prob is not None:
#                 val_strong = adj.get("e_val_strong_agree", E_VAL_STRONG_AGREE)
#                 val_agree = adj.get("e_val_agree", E_VAL_AGREE)
#                 val_conflict = adj.get("e_val_conflict_skip", E_VAL_CONFLICT_SKIP)

#                 e_agrees = (e_prob > 0.5) if best_signal.direction == "buy" else (e_prob < 0.5)
#                 e_strength = e_prob if best_signal.direction == "buy" else (1 - e_prob)

#                 if e_agrees:
#                     if e_strength >= val_strong:
#                         best_signal.confidence = min(100, best_signal.confidence + E_STRONG_BOOST + adj.get("e_conf_boost", 0))
#                         sources += "+E_strong"
#                     elif e_strength >= val_agree:
#                         sources += "+E_agree"
#                     else:
#                         best_signal.confidence = max(0, best_signal.confidence + E_WEAK_PENALTY)
#                         sources += "+E_weak"
#                 else:
#                     if e_strength >= val_conflict:
#                         return None  # E vetoes
#                     else:
#                         best_signal.confidence = max(0, best_signal.confidence + E_WEAK_PENALTY)
#                         sources += "+E_disagree"

#             # Regime SL/TP adjustment
#             sl_mult = adj.get("sl_mult", 1.0)
#             tp_mult = adj.get("tp_mult", 1.0)
#             entry = best_signal.entry_price
#             if sl_mult != 1.0:
#                 sl_dist = abs(entry - best_signal.sl_price)
#                 if best_signal.direction == "buy":
#                     best_signal.sl_price = round(entry - sl_dist * sl_mult, 2)
#                 else:
#                     best_signal.sl_price = round(entry + sl_dist * sl_mult, 2)
#             if tp_mult != 1.0:
#                 tp_dist = abs(entry - best_signal.tp_price)
#                 if best_signal.direction == "buy":
#                     best_signal.tp_price = round(entry + tp_dist * tp_mult, 2)
#                 else:
#                     best_signal.tp_price = round(entry - tp_dist * tp_mult, 2)

#             # Dynamic lot
#             sl_distance = abs(entry - best_signal.sl_price)
#             lot_result = self.lot_sizer.compute(
#                 n_strategies=n_agree, e_prob=e_prob, regime=regime,
#                 daily_pnl=self.risk.daily_pnl, atr=atr, sl_distance=sl_distance,
#             )
#             best_signal.sub_conditions["dynamic_lot"] = lot_result["lot_size"]
#             best_signal.sub_conditions["source"] = sources
#             best_signal.sub_conditions["regime"] = regime

#             if best_signal.confidence < CONFIDENCE_THRESHOLD:
#                 return None

#             return best_signal

#         # ── Case 2: A-D silent → E independent ────────────
#         if e_prob is None or not self.live_e.loaded:
#             return None

#         if not adj.get("allow_e_independent", True):
#             return None

#         threshold = adj.get("e_indep_threshold", E_PROB_THRESHOLD)
#         if e_prob >= threshold:
#             direction = "buy"
#         elif e_prob <= (1 - threshold):
#             direction = "sell"
#         else:
#             return None

#         effective = max(e_prob, 1 - e_prob)
#         confidence = 50 + (effective - 0.5) * 100 + adj.get("e_conf_boost", 0)
#         confidence = max(50, min(100, confidence))

#         if confidence < E_INDEPENDENT_MIN_CONF + adj.get("e_conf_boost", 0):
#             return None

#         row = row  # Already have it
#         close = row["close"]
#         spread = SPREAD_SIMULATION_PIPS * PIP_VALUE

#         sl_dist = E_SL_ATR_MULT * atr * adj.get("sl_mult", 1.0)
#         rr = E_TP_RR_HIGH if confidence >= 75 else E_TP_RR_BASE
#         tp_dist = sl_dist * rr * adj.get("tp_mult", 1.0)

#         if direction == "buy":
#             entry = close + spread
#             sl, tp = entry - sl_dist, entry + tp_dist
#         else:
#             entry = close - spread
#             sl, tp = entry + sl_dist, entry - tp_dist

#         # Dynamic lot
#         lot_result = self.lot_sizer.compute(
#             n_strategies=0, e_prob=e_prob, regime=regime,
#             daily_pnl=self.risk.daily_pnl, atr=atr, sl_distance=sl_dist,
#         )

#         return TradeSignal(
#             datetime=row["datetime"],
#             direction=direction,
#             strategy_name="MIDAS_E",
#             confidence=confidence,
#             entry_price=round(entry, 2),
#             sl_price=round(sl, 2),
#             tp_price=round(tp, 2),
#             sub_conditions={
#                 "source": "E_independent",
#                 "regime": regime,
#                 "e_prob": round(e_prob, 4),
#                 "dynamic_lot": lot_result["lot_size"],
#                 "strategies_agreed": 0,
#             },
#         )

#     # ─── Execution ────────────────────────────────────────

#     def _execute_signal(self, signal, session, regime, e_prob):
#         """Execute a trade signal on MT5."""
#         lot = signal.sub_conditions.get("dynamic_lot", LOT_SIZE_SINGLE)
#         source = signal.sub_conditions.get("source", "unknown")
#         sl_distance = abs(signal.entry_price - signal.sl_price)

#         # Final risk validation
#         allowed, adjusted_lot, reason = self.risk.validate_trade_risk(sl_distance, lot)
#         if not allowed:
#             print(f"  ⛔ Risk blocked: {reason}")
#             self.notifier.signal_skipped(reason, source)
#             return

#         lot = adjusted_lot
#         risk_dollars = sl_distance * lot * XAUUSD_POINT_VALUE

#         print(f"\n  🔔 SIGNAL: {signal.direction.upper()} | Source: {source} | "
#               f"Conf: {signal.confidence:.0f}% | Regime: {regime}")
#         print(f"     Entry: {signal.entry_price:.2f} | SL: {signal.sl_price:.2f} | "
#               f"TP: {signal.tp_price:.2f} | Lot: {lot} | Risk: ${risk_dollars:.2f}")

#         if self.dry_run:
#             print(f"  🔸 DRY RUN — trade not executed")
#             self.logger.log_event("DRY_RUN_SIGNAL", {
#                 "direction": signal.direction, "source": source,
#                 "confidence": signal.confidence, "lot": lot,
#             })
#             return

#         # Execute on MT5
#         result = self.mt5.open_trade(
#             direction=signal.direction,
#             lot_size=lot,
#             sl_price=signal.sl_price,
#             tp_price=signal.tp_price,
#             comment=f"MIDAS_{source[:20]}",
#         )

#         if result is None:
#             print(f"  ❌ Execution failed!")
#             self.notifier.error(f"Execution failed: {source}")
#             return

#         # Record in risk manager
#         self.risk.record_trade_open(result["ticket"], session)

#         # Log
#         trade_info = {
#             "ticket": result["ticket"],
#             "direction": signal.direction,
#             "entry_price": result["entry_price"],
#             "sl_price": signal.sl_price,
#             "tp_price": signal.tp_price,
#             "lot_size": lot,
#             "source": source,
#             "confidence": signal.confidence,
#             "regime": regime,
#             "session": session,
#             "risk_dollars": round(risk_dollars, 2),
#             "e_prob": e_prob,
#         }
#         self.logger.log_trade_open(trade_info)
#         self.notifier.trade_opened(trade_info)

#     # ─── Position Monitoring ──────────────────────────────

#     def _monitor_positions(self):
#         """Check if any open position has been closed by MT5 (SL/TP hit)."""
#         if self.risk.open_trade_ticket is None:
#             return

#         positions = self.mt5.get_open_positions()

#         # Check if our tracked position is still open
#         our_tickets = [p["ticket"] for p in positions]
#         if self.risk.open_trade_ticket not in our_tickets:
#             # Position was closed (SL/TP hit by broker)
#             self._handle_position_closed()

#     def _handle_position_closed(self):
#         """Handle a position that was closed by the broker (SL/TP)."""
#         # Get the trade from MT5 history
#         import MetaTrader5 as mt5

#         ticket = self.risk.open_trade_ticket
#         deals = mt5.history_deals_get(position=ticket)

#         if deals and len(deals) >= 2:
#             close_deal = deals[-1]
#             pnl = close_deal.profit + close_deal.commission + close_deal.swap
#             exit_price = close_deal.price
#             reason = "tp_hit" if pnl > 0 else "sl_hit"
#         else:
#             # Fallback: get from account
#             acc = self.mt5.get_account_info()
#             pnl = 0  # Can't determine exact PnL
#             exit_price = 0
#             reason = "unknown"

#         balance = self.mt5.get_balance()

#         # Update risk manager
#         self.risk.record_trade_close(pnl, balance)
#         self.lot_sizer.record_result(pnl)

#         # Log
#         self.logger.log_trade_close(ticket, exit_price, pnl, reason, balance)

#         # Notify
#         self.notifier.trade_closed(pnl, reason, balance)

#         emoji = "✅" if pnl > 0 else "❌"
#         print(f"\n  {emoji} TRADE CLOSED | PnL: ${pnl:+.2f} | Reason: {reason} | "
#               f"Balance: ${balance:,.2f}")

#     # ─── Feature Computation ──────────────────────────────

#     def _build_feature_dataframe(self) -> pd.DataFrame:
#         """
#         Fetch live data from MT5 and compute all features.
#         Returns a merged M5 DataFrame with M15 context — same format
#         as load_and_prepare_data() in run_combined_abcd.py.
#         """
#         # Fetch M5 data
#         df_m5 = self.mt5.fetch_bars("M5", M5_HISTORY_BARS)
#         if df_m5 is None or len(df_m5) < 100:
#             return None

#         # Fetch M15 data
#         df_m15 = self.mt5.fetch_bars("M15", M15_HISTORY_BARS)
#         if df_m15 is None or len(df_m15) < 50:
#             return None

#         # Compute features on M15
#         df_m15 = self.feature_engine.compute_shared(df_m15, timeframe="M15")
#         df_m15 = self.feature_engine.compute_fvg(df_m15)
#         df_m15 = self.feature_engine.compute_smc(df_m15)

#         # Compute features on M5
#         df_m5 = self.feature_engine.compute_shared(df_m5, timeframe="M5")
#         df_m5 = self.feature_engine.compute_trend(df_m5)
#         df_m5 = self.feature_engine.compute_session_range(df_m5)

#         # Merge M15 into M5 (same logic as run_combined_abcd.py)
#         df_merged = self._merge_m15_into_m5(df_m5, df_m15)

#         return df_merged

#     def _merge_m15_into_m5(self, df_m5, df_m15):
#         """Merge M15 features into M5 — same as run_combined_abcd.merge_timeframes."""
#         df_m5 = df_m5.sort_values("datetime").reset_index(drop=True)
#         df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

#         # Select M15 columns to merge
#         m15_cols = ["datetime"]

#         fvg_cols = [c for c in df_m15.columns if c.startswith("fvg_")]
#         m15_cols.extend(fvg_cols)

#         smc_cols = [c for c in df_m15.columns if c.startswith(("bos_", "ob_", "equal_", "last_equal_"))]
#         m15_cols.extend(smc_cols)

#         m15_cols.extend(["open", "high", "low", "close"])

#         indicator_cols = [
#             "ema_9", "ema_21", "ema_50", "ema_200",
#             "ema_50_slope", "ema_stack",
#             "ema_partial_bull", "ema_partial_bear",
#             "above_ema200",
#             f"atr_{ATR_PERIOD}", f"rsi_{ATR_PERIOD}",
#             f"adx_{ATR_PERIOD}", "trend_strong", "trend_very_strong", "trend_weak",
#             "candle_range", "body", "is_bullish",
#             "last_swing_high", "last_swing_low",
#             "vol_regime",
#         ]
#         for col in indicator_cols:
#             if col in df_m15.columns:
#                 m15_cols.append(col)

#         m15_cols = list(dict.fromkeys(m15_cols))
#         df_m15_sub = df_m15[m15_cols].copy()

#         m15_orig_idx = df_m15_sub.index.values
#         rename_map = {col: f"m15_{col}" for col in df_m15_sub.columns if col != "datetime"}
#         df_m15_sub = df_m15_sub.rename(columns=rename_map)
#         df_m15_sub = df_m15_sub.rename(columns={"datetime": "m15_datetime_key"})
#         df_m15_sub["m15_idx"] = m15_orig_idx

#         df_merged = pd.merge_asof(
#             df_m5, df_m15_sub,
#             left_on="datetime", right_on="m15_datetime_key",
#             direction="backward",
#         )
#         df_merged = df_merged.rename(columns={"m15_datetime_key": "m15_datetime"})
#         return df_merged

#     # ─── Shutdown ─────────────────────────────────────────

#     def _shutdown(self):
#         """Graceful shutdown."""
#         print("\n  Shutting down...")

#         # Close any open positions
#         if self.mt5.has_open_position():
#             print("  ⚠️ Open position detected during shutdown!")
#             # Don't auto-close — let the trader decide

#         self.mt5.disconnect()
#         self.notifier.bot_stopped("Shutdown")
#         self.risk.print_status()

#         print("  👋 MIDAS v2 stopped.")


# # ─── Entry Point ──────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(description="MIDAS v2 — Live Trading Bot")
#     parser.add_argument("--dry-run", action="store_true",
#                         help="Generate signals but don't execute trades")
#     parser.add_argument("--mode", choices=["demo", "live"],
#                         default="demo", help="Account mode (default: demo)")
#     args = parser.parse_args()

#     if args.mode == "live" and not args.dry_run:
#         print("\n  ⚠️  LIVE MODE — Real money will be traded!")
#         confirm = input("  Type 'CONFIRM' to proceed: ")
#         if confirm != "CONFIRM":
#             print("  Aborted.")
#             return

#     bot = MidasBot(dry_run=args.dry_run, mode=args.mode)
#     bot.start()


# if __name__ == "__main__":
#     main()


##############################################################################
##############################################################################
##############################################################################
##############################################################################
                            # FINAL MAIN
##############################################################################
##############################################################################
##############################################################################
"""
Project MIDAS v2 — Main Live Trading Bot (Full Featured)
==========================================================
Architecture:
  - Strategies A-D+E run in parallel (ThreadPoolExecutor)
  - TSL monitors open positions and trails SL dynamically
  - Position updates sent to Telegram every 3 minutes
  - Telegram command listener (/help /stop /start /pause /summary etc.)
  - Risk manager enforces all The5ers compliance rules
  - State persists to disk (safe restart)

Usage:
    python main.py              # Live demo
    python main.py --dry-run    # Signal-only mode
    python main.py --mode live  # Real money (requires CONFIRM)
"""

import argparse
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE, MT5_SYMBOL,
    SPREAD_SIMULATION_PIPS, LOT_SIZE_SINGLE,
    MAX_TRADE_RISK, CONFIDENCE_THRESHOLD, SESSIONS,
)
from feature_engineering import FeatureEngine
from mt5_executor import MT5Executor
from risk_manager import RiskManager
from trade_logger import TradeLogger
from telegram_notifier import TelegramNotifier
from strategies.live_strategy_e import LiveStrategyE
from regime_detector import RegimeDetector, REGIME_ADJUSTMENTS
from dynamic_lot_sizer import DynamicLotSizer
from tsl_manager import TSLManager

from strategies.strategy_a import StrategyA
from strategies.strategy_b import StrategyB
from strategies.strategy_c import StrategyC
from strategies.strategy_d import StrategyD
from backtester import TradeSignal

# ─── Configuration ────────────────────────────────────────
M5_HISTORY_BARS = 500
M15_HISTORY_BARS = 200
LOOP_INTERVAL_SECONDS = 10
TSL_CHECK_INTERVAL_SECONDS = 5   # Check TSL more frequently than signal
MAX_ERRORS = 5

# Strategy E thresholds
E_PROB_THRESHOLD = 0.68
E_VAL_STRONG_AGREE = 0.60
E_VAL_AGREE = 0.52
E_VAL_CONFLICT_SKIP = 0.60
E_STRONG_BOOST = 10
E_WEAK_PENALTY = -10
E_SL_ATR_MULT = 1.5
E_TP_RR_BASE = 1.5
E_TP_RR_HIGH = 2.5
E_INDEPENDENT_MIN_CONF = 70


class MidasBot:
    """Full-featured MIDAS v2 trading bot."""

    def __init__(self, dry_run: bool = False, mode: str = "demo"):
        self.dry_run = dry_run
        self.mode = mode

        # Core components
        self.mt5 = MT5Executor()
        self.risk = RiskManager()
        self.logger = TradeLogger()
        self.notifier = TelegramNotifier()
        self.engine = FeatureEngine()
        self.regime = RegimeDetector()
        self.lot_sizer = DynamicLotSizer()
        self.tsl = TSLManager()
        self.live_e = LiveStrategyE()

        # Strategies (initialized once, reused)
        self.strategies = {
            "A": StrategyA(),
            "B": StrategyB(),
            "C": StrategyC(),
            "D": StrategyD(),
        }

        # State
        self.last_m5_time = None
        self.error_count = 0
        self.running = True
        self.paused = False
        self.pause_until = None

        # Active trade info (for position monitoring)
        self.active_trade_info = None  # Full trade details for updates
        self.active_trade_open_time = None

    # ═════════════════════════════════════════════════════
    # STARTUP
    # ═════════════════════════════════════════════════════

    def start(self):
        """Initialize everything and run."""
        self._print_banner()

        # Connect MT5
        print("\n  Connecting to MT5...")
        if not self.mt5.connect():
            print("  FATAL: Cannot connect to MT5")
            return

        acc = self.mt5.get_account_info()
        balance = acc.get("balance", 5000)
        self.risk.update_balance(balance)

        # Load Strategy E
        print("\n  Loading AI model...")
        if not self.live_e.loaded:
            print("  WARNING: Strategy E unavailable. Running A-D only.")

        # Register Telegram commands
        self._register_commands()
        self.notifier.start_polling()

        # Notify
        self.notifier.bot_started(balance, self.mode.upper())
        self.logger.log_event("BOT_STARTED", {"mode": self.mode, "balance": balance})

        print(f"\n  ✅ Bot running | Balance: ${balance:,.2f}")
        print(f"  Press Ctrl+C to stop\n")
        print("─" * 60)

        # Main loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n\n  Stopping (Ctrl+C)...")
        except Exception as e:
            print(f"\n  FATAL: {e}")
            traceback.print_exc()
            self.notifier.error(f"Fatal: {str(e)[:200]}")
        finally:
            self._shutdown()

    def _print_banner(self):
        print("=" * 60)
        print("  MIDAS v2 — XAUUSD AI Trading Bot")
        print("=" * 60)
        print(f"  Mode         : {self.mode.upper()}")
        print(f"  Dry Run      : {'YES' if self.dry_run else 'NO'}")
        print(f"  Strategies   : A+B+C+D+E (parallel)")
        print(f"  Regime       : ON")
        print(f"  Dynamic TSL  : ON")
        print(f"  Dynamic Lots : 0.01-0.05")
        print(f"  Risk Cap     : $40/trade, $100/day")
        print(f"  Session Lim  : 2 per session")
        print(f"  Cooldown     : 10 min")

    # ═════════════════════════════════════════════════════
    # MAIN LOOP
    # ═════════════════════════════════════════════════════

    def _main_loop(self):
        """Main event loop."""
        while self.running:
            try:
                # ── Pause check ───────────────────────────
                if self.paused:
                    if self.pause_until and datetime.now() >= self.pause_until:
                        self.paused = False
                        self.pause_until = None
                        self.notifier.bot_resumed()
                        print("  ▶️ Bot resumed")
                    else:
                        # Still monitor positions while paused
                        if self.risk.open_trade_ticket is not None:
                            self._monitor_and_trail()
                        time.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                # ── Position monitoring (always active) ───
                if self.risk.open_trade_ticket is not None:
                    self._monitor_and_trail()
                    time.sleep(TSL_CHECK_INTERVAL_SECONDS)
                    continue  # Don't look for new signals while position open

                # ── Check for new M5 bar ──────────────────
                latest = self.mt5.fetch_latest_bar("M5")
                if latest is None:
                    time.sleep(LOOP_INTERVAL_SECONDS)
                    continue

                bar_time = latest["datetime"]
                if isinstance(bar_time, np.datetime64):
                    bar_time = pd.Timestamp(bar_time).to_pydatetime()

                if self.last_m5_time is not None and bar_time <= self.last_m5_time:
                    time.sleep(LOOP_INTERVAL_SECONDS)
                    continue

                # ── New bar → process ─────────────────────
                self.last_m5_time = bar_time
                self._process_bar()
                self.error_count = 0

            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.error_count += 1
                print(f"  ⚠️ Error ({self.error_count}): {e}")
                if self.error_count >= MAX_ERRORS:
                    print("  Reconnecting MT5...")
                    self.mt5.reconnect()
                    self.error_count = 0
                time.sleep(LOOP_INTERVAL_SECONDS * 2)

    # ═════════════════════════════════════════════════════
    # BAR PROCESSING
    # ═════════════════════════════════════════════════════

    def _process_bar(self):
        """Process a new M5 bar — full signal pipeline."""
        now = datetime.now(UTC)
        session = self.risk.get_session(now)
        price = self.mt5.get_current_price()

        print(f"\n  [{now.strftime('%H:%M')}] {price.get('bid',0):.2f} | "
              f"{session} | DayPnL: ${self.risk.daily_pnl:+.2f}")

        # Risk pre-flight
        can_trade, reason = self.risk.can_trade(session, now)
        if not can_trade:
            print(f"  Skip: {reason}")
            return

        # Build features
        df = self._build_features()
        if df is None or len(df) < 100:
            print(f"  Skip: insufficient data")
            return

        idx = len(df) - 1
        row = df.iloc[idx]

        # Detect regime
        current_regime, adj = self.regime.detect_and_adjust(row)

        # ── Run A-D in parallel ───────────────────────────
        ad_signals = self._run_strategies_parallel(df, idx)

        # ── Get Strategy E prediction ─────────────────────
        e_prob = self._get_e_prediction(df, idx)

        # ── Decision ──────────────────────────────────────
        signal = self._make_decision(ad_signals, e_prob, adj, current_regime, row)

        if signal is None:
            sources = ", ".join(f"{n}" for n, _ in ad_signals) if ad_signals else "none"
            e_str = f"E={e_prob:.3f}" if e_prob else "E=N/A"
            print(f"  No trade | AD: {sources} | {e_str} | Regime: {current_regime}")
            return

        # ── Execute ───────────────────────────────────────
        self._execute(signal, session, current_regime, e_prob)

    def _run_strategies_parallel(self, df, idx) -> List[Tuple[str, TradeSignal]]:
        """Run all 4 strategies in parallel using ThreadPoolExecutor."""
        signals = []

        def run_one(name, strategy):
            try:
                sig = strategy.generate_signal(df, idx)
                return (name, sig) if sig is not None else None
            except Exception as e:
                print(f"  Strategy {name} error: {e}")
                return None

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="Strat") as pool:
            futures = {
                pool.submit(run_one, name, strat): name
                for name, strat in self.strategies.items()
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    signals.append(result)

        return signals

    def _get_e_prediction(self, df, idx) -> Optional[float]:
        """Get Strategy E's prediction on debiased data."""
        if not self.live_e.loaded:
            return None
        try:
            df_debiased = self.live_e.debias_dataframe(df)
            return self.live_e.predict(df_debiased, idx)
        except Exception as e:
            print(f"  Strategy E error: {e}")
            return None

    # ═════════════════════════════════════════════════════
    # DECISION LOGIC (same as Phase 8 runner)
    # ═════════════════════════════════════════════════════

    def _make_decision(self, ad_signals, e_prob, adj, regime, row):
        """Combined A-D+E decision with regime and dynamic lots."""
        atr = row.get(f"atr_{ATR_PERIOD}", 2.0)
        if pd.isna(atr) or atr <= 0:
            atr = 2.0

        if ad_signals:
            return self._decision_ad(ad_signals, e_prob, adj, regime, atr, row)
        return self._decision_e_independent(e_prob, adj, regime, atr, row)

    def _decision_ad(self, signals, e_prob, adj, regime, atr, row):
        """A-D fired → validate with E, adjust with regime + dynamic lots."""
        dirs = set(s.direction for _, s in signals)
        if len(dirs) > 1:
            return None

        best_name, best_signal = max(signals, key=lambda x: x[1].confidence)
        n_agree = len(signals)
        best_signal.confidence = min(100, best_signal.confidence + (n_agree - 1) * 10)
        sources = "+".join(n for n, _ in signals)

        # E validation
        if e_prob is not None:
            val_strong = adj.get("e_val_strong_agree", E_VAL_STRONG_AGREE)
            val_agree = adj.get("e_val_agree", E_VAL_AGREE)
            val_conflict = adj.get("e_val_conflict_skip", E_VAL_CONFLICT_SKIP)

            e_agrees = (e_prob > 0.5) if best_signal.direction == "buy" else (e_prob < 0.5)
            e_strength = e_prob if best_signal.direction == "buy" else (1 - e_prob)

            if e_agrees:
                if e_strength >= val_strong:
                    best_signal.confidence = min(100, best_signal.confidence + E_STRONG_BOOST + adj.get("e_conf_boost", 0))
                    sources += "+E_strong"
                elif e_strength >= val_agree:
                    sources += "+E_agree"
                else:
                    best_signal.confidence = max(0, best_signal.confidence + E_WEAK_PENALTY)
                    sources += "+E_weak"
            else:
                if e_strength >= val_conflict:
                    return None
                else:
                    best_signal.confidence = max(0, best_signal.confidence + E_WEAK_PENALTY)
                    sources += "+E_disagree"

        # Regime SL/TP
        entry = best_signal.entry_price
        sl_mult = adj.get("sl_mult", 1.0)
        tp_mult = adj.get("tp_mult", 1.0)
        if sl_mult != 1.0:
            sd = abs(entry - best_signal.sl_price)
            best_signal.sl_price = round(entry - sd * sl_mult if best_signal.direction == "buy" else entry + sd * sl_mult, 2)
        if tp_mult != 1.0:
            td = abs(entry - best_signal.tp_price)
            best_signal.tp_price = round(entry + td * tp_mult if best_signal.direction == "buy" else entry - td * tp_mult, 2)

        # Dynamic lot
        sl_distance = abs(entry - best_signal.sl_price)
        lot_result = self.lot_sizer.compute(
            n_strategies=n_agree, e_prob=e_prob, regime=regime,
            daily_pnl=self.risk.daily_pnl, atr=atr, sl_distance=sl_distance,
        )
        best_signal.sub_conditions["dynamic_lot"] = lot_result["lot_size"]
        best_signal.sub_conditions["source"] = sources
        best_signal.sub_conditions["regime"] = regime

        if best_signal.confidence < CONFIDENCE_THRESHOLD:
            return None
        return best_signal

    def _decision_e_independent(self, e_prob, adj, regime, atr, row):
        """E independent signal when A-D are silent."""
        if e_prob is None or not self.live_e.loaded:
            return None
        if not adj.get("allow_e_independent", True):
            return None

        threshold = adj.get("e_indep_threshold", E_PROB_THRESHOLD)
        if e_prob >= threshold:
            direction = "buy"
        elif e_prob <= (1 - threshold):
            direction = "sell"
        else:
            return None

        effective = max(e_prob, 1 - e_prob)
        confidence = 50 + (effective - 0.5) * 100 + adj.get("e_conf_boost", 0)
        confidence = max(50, min(100, confidence))

        if confidence < E_INDEPENDENT_MIN_CONF + adj.get("e_conf_boost", 0):
            return None

        close = row["close"]
        spread = SPREAD_SIMULATION_PIPS * PIP_VALUE
        sl_dist = E_SL_ATR_MULT * atr * adj.get("sl_mult", 1.0)
        rr = E_TP_RR_HIGH if confidence >= 75 else E_TP_RR_BASE
        tp_dist = sl_dist * rr * adj.get("tp_mult", 1.0)

        if direction == "buy":
            entry, sl, tp = close + spread, close + spread - sl_dist, close + spread + tp_dist
        else:
            entry, sl, tp = close - spread, close - spread + sl_dist, close - spread - tp_dist

        lot_result = self.lot_sizer.compute(
            n_strategies=0, e_prob=e_prob, regime=regime,
            daily_pnl=self.risk.daily_pnl, atr=atr, sl_distance=sl_dist,
        )

        return TradeSignal(
            datetime=row["datetime"], direction=direction,
            strategy_name="MIDAS_E", confidence=confidence,
            entry_price=round(entry, 2), sl_price=round(sl, 2), tp_price=round(tp, 2),
            sub_conditions={
                "source": "E_independent", "regime": regime,
                "e_prob": round(e_prob, 4),
                "dynamic_lot": lot_result["lot_size"],
                "strategies_agreed": 0,
            },
        )

    # ═════════════════════════════════════════════════════
    # EXECUTION
    # ═════════════════════════════════════════════════════

    def _execute(self, signal, session, regime, e_prob):
        """Execute trade on MT5 with full logging and notification."""
        lot = signal.sub_conditions.get("dynamic_lot", LOT_SIZE_SINGLE)
        source = signal.sub_conditions.get("source", "unknown")
        sl_distance = abs(signal.entry_price - signal.sl_price)

        # Risk validation
        allowed, adjusted_lot, reason = self.risk.validate_trade_risk(sl_distance, lot)
        if not allowed:
            print(f"  ⛔ Risk blocked: {reason}")
            self.notifier.signal_skipped(reason, source)
            return

        lot = adjusted_lot
        risk_dollars = sl_distance * lot * XAUUSD_POINT_VALUE

        print(f"\n  🔔 SIGNAL: {signal.direction.upper()} | {source} | "
              f"Conf: {signal.confidence:.0f}% | Regime: {regime}")
        print(f"     Entry: {signal.entry_price:.2f} | SL: {signal.sl_price:.2f} | "
              f"TP: {signal.tp_price:.2f} | Lot: {lot} | Risk: ${risk_dollars:.2f}")

        if self.dry_run:
            print(f"  🔸 DRY RUN — not executed")
            self.logger.log_event("DRY_RUN", {"source": source, "direction": signal.direction})
            return

        # Execute on MT5
        result = self.mt5.open_trade(
            direction=signal.direction, lot_size=lot,
            sl_price=signal.sl_price, tp_price=signal.tp_price,
            comment=f"MIDAS_{source[:20]}",
        )

        if result is None:
            print(f"  ❌ Execution failed!")
            self.notifier.error(f"Execution failed: {source}")
            return

        # Record
        self.risk.record_trade_open(result["ticket"], session)
        self.active_trade_open_time = datetime.now()

        trade_info = {
            "ticket": result["ticket"],
            "direction": signal.direction,
            "entry_price": result["entry_price"],
            "sl_price": signal.sl_price,
            "tp_price": signal.tp_price,
            "lot_size": lot,
            "source": source,
            "confidence": signal.confidence,
            "regime": regime,
            "session": session,
            "risk_dollars": round(risk_dollars, 2),
            "e_prob": e_prob,
        }
        self.active_trade_info = trade_info
        self.logger.log_trade_open(trade_info)
        self.notifier.trade_opened(trade_info)

        # Start TSL tracking
        self.tsl.start_tracking({
            "ticket": result["ticket"],
            "direction": signal.direction,
            "entry_price": result["entry_price"],
            "sl_price": signal.sl_price,
            "tp_price": signal.tp_price,
            "lot_size": lot,
        })

    # ═════════════════════════════════════════════════════
    # POSITION MONITORING + TSL
    # ═════════════════════════════════════════════════════

    def _monitor_and_trail(self):
        """Monitor open position: check if closed, update TSL, send position updates."""
        if self.risk.open_trade_ticket is None:
            return

        # Check if position still open
        positions = self.mt5.get_open_positions()
        our_tickets = [p["ticket"] for p in positions]

        if self.risk.open_trade_ticket not in our_tickets:
            self._handle_closed()
            return

        # Get current position
        pos = next(p for p in positions if p["ticket"] == self.risk.open_trade_ticket)
        current_price = pos["current_price"]
        unrealized_pnl = pos["profit"]

        # ── TSL check ─────────────────────────────────────
        tsl_result = self.tsl.update(current_price)
        if tsl_result is not None:
            new_sl = tsl_result["new_sl"]
            old_sl = pos["sl"]

            # Apply to MT5
            if not self.dry_run:
                success = self.mt5.modify_sl_tp(
                    self.risk.open_trade_ticket, new_sl=new_sl
                )
                if success:
                    print(f"  📐 TSL: {old_sl:.2f} → {new_sl:.2f} | {tsl_result['reason']}")
                    self.logger.log_tsl_change(
                        self.risk.open_trade_ticket, old_sl, new_sl, tsl_result["reason"]
                    )
                    self.notifier.tsl_updated(
                        old_sl, new_sl, tsl_result["reason"],
                        current_price, self.tsl.get_progress(current_price)
                    )

        # ── 3-minute position update ──────────────────────
        if self.active_trade_info:
            progress = self.tsl.get_progress(current_price)
            tsl_status = self.tsl.get_status()
            self.notifier.position_update(
                self.active_trade_info, current_price,
                unrealized_pnl, progress, tsl_status
            )

    def _handle_closed(self):
        """Handle a position closed by broker (SL/TP/TSL hit)."""
        ticket = self.risk.open_trade_ticket

        try:
            import MetaTrader5 as mt5
            deals = mt5.history_deals_get(position=ticket)
            if deals and len(deals) >= 2:
                close_deal = deals[-1]
                pnl = close_deal.profit + close_deal.commission + close_deal.swap
                exit_price = close_deal.price
                reason = "tp_hit" if pnl > 0 else "sl_hit"
            else:
                pnl, exit_price, reason = 0, 0, "unknown"
        except Exception:
            pnl, exit_price, reason = 0, 0, "unknown"

        balance = self.mt5.get_balance()
        duration = 0
        if self.active_trade_open_time:
            duration = int((datetime.now() - self.active_trade_open_time).total_seconds() / 60)

        # Update systems
        self.risk.record_trade_close(pnl, balance)
        self.lot_sizer.record_result(pnl)
        self.tsl.stop_tracking()

        direction = self.active_trade_info.get("direction", "") if self.active_trade_info else ""
        entry = self.active_trade_info.get("entry_price", 0) if self.active_trade_info else 0

        self.logger.log_trade_close(ticket, exit_price, pnl, reason, balance, duration)
        self.notifier.trade_closed(pnl, reason, balance, direction, entry, exit_price, duration)

        self.active_trade_info = None
        self.active_trade_open_time = None

        emoji = "✅" if pnl > 0 else "❌"
        print(f"\n  {emoji} CLOSED | PnL: ${pnl:+.2f} | {reason} | "
              f"Duration: {duration}m | Balance: ${balance:,.2f}")

    # ═════════════════════════════════════════════════════
    # FEATURE COMPUTATION
    # ═════════════════════════════════════════════════════

    def _build_features(self) -> Optional[pd.DataFrame]:
        """Fetch live data and compute all features."""
        df_m5 = self.mt5.fetch_bars("M5", M5_HISTORY_BARS)
        df_m15 = self.mt5.fetch_bars("M15", M15_HISTORY_BARS)

        if df_m5 is None or len(df_m5) < 100 or df_m15 is None or len(df_m15) < 50:
            return None

        df_m15 = self.engine.compute_shared(df_m15, "M15")
        df_m15 = self.engine.compute_fvg(df_m15)
        df_m15 = self.engine.compute_smc(df_m15)

        df_m5 = self.engine.compute_shared(df_m5, "M5")
        df_m5 = self.engine.compute_trend(df_m5)
        df_m5 = self.engine.compute_session_range(df_m5)

        return self._merge_m15(df_m5, df_m15)

    def _merge_m15(self, df_m5, df_m15):
        """Merge M15 features into M5 (same as run_combined_abcd.py)."""
        df_m5 = df_m5.sort_values("datetime").reset_index(drop=True)
        df_m15 = df_m15.sort_values("datetime").reset_index(drop=True)

        m15_cols = ["datetime"]
        m15_cols += [c for c in df_m15.columns if c.startswith("fvg_")]
        m15_cols += [c for c in df_m15.columns if c.startswith(("bos_", "ob_", "equal_", "last_equal_"))]
        m15_cols += ["open", "high", "low", "close"]

        for col in ["ema_9", "ema_21", "ema_50", "ema_200", "ema_50_slope", "ema_stack",
                     "ema_partial_bull", "ema_partial_bear", "above_ema200",
                     f"atr_{ATR_PERIOD}", f"rsi_{ATR_PERIOD}", f"adx_{ATR_PERIOD}",
                     "trend_strong", "trend_very_strong", "trend_weak",
                     "candle_range", "body", "is_bullish",
                     "last_swing_high", "last_swing_low", "vol_regime"]:
            if col in df_m15.columns:
                m15_cols.append(col)

        m15_cols = list(dict.fromkeys(m15_cols))
        sub = df_m15[m15_cols].copy()
        orig_idx = sub.index.values
        rename = {c: f"m15_{c}" for c in sub.columns if c != "datetime"}
        sub = sub.rename(columns=rename).rename(columns={"datetime": "m15_datetime_key"})
        sub["m15_idx"] = orig_idx

        merged = pd.merge_asof(df_m5, sub, left_on="datetime",
                                right_on="m15_datetime_key", direction="backward")
        return merged.rename(columns={"m15_datetime_key": "m15_datetime"})

    # ═════════════════════════════════════════════════════
    # TELEGRAM COMMANDS
    # ═════════════════════════════════════════════════════

    def _register_commands(self):
        """Register all Telegram command handlers."""

        self.notifier.register_command("/stop", lambda _: self._cmd_stop())
        self.notifier.register_command("/start", lambda _: self._cmd_resume())
        self.notifier.register_command("/pause", lambda args: self._cmd_pause(args))
        self.notifier.register_command("/status", lambda _: self._cmd_status())
        self.notifier.register_command("/summary", lambda _: self._cmd_summary())
        self.notifier.register_command("/trades", lambda _: self._cmd_trades())
        self.notifier.register_command("/risk", lambda _: self._cmd_risk())
        self.notifier.register_command("/balance", lambda _: self._cmd_balance())
        self.notifier.register_command("/regime", lambda _: self._cmd_regime())
        self.notifier.register_command("/positions", lambda _: self._cmd_positions())

    def _cmd_stop(self) -> str:
        self.running = False
        return "🛑 Stopping bot after current cycle..."

    def _cmd_resume(self) -> str:
        self.paused = False
        self.pause_until = None
        return "▶️ Bot resumed!"

    def _cmd_pause(self, args: str) -> str:
        try:
            minutes = int(args) if args.strip() else 30
        except ValueError:
            minutes = 30
        self.paused = True
        self.pause_until = datetime.now() + timedelta(minutes=minutes)
        return f"⏸️ Paused for {minutes} minutes (until {self.pause_until.strftime('%H:%M')})"

    def _cmd_status(self) -> str:
        acc = self.mt5.get_account_info()
        regime_stats = self.regime.get_stats()
        top_regime = max(regime_stats, key=lambda r: regime_stats[r]["pct"]) if regime_stats else "N/A"

        state = "PAUSED" if self.paused else "RUNNING"
        if self.risk.open_trade_ticket:
            state = "IN TRADE"

        return (
            f"📊 <b>STATUS</b>\n"
            f"State     : {state}\n"
            f"Balance   : ${acc.get('balance', 0):,.2f}\n"
            f"Equity    : ${acc.get('equity', 0):,.2f}\n"
            f"Day PnL   : ${self.risk.daily_pnl:+.2f}\n"
            f"Day Trades: {self.risk.daily_trades}\n"
            f"Regime    : {top_regime}\n"
            f"E loaded  : {'Yes' if self.live_e.loaded else 'No'}\n"
            f"Dry Run   : {'Yes' if self.dry_run else 'No'}"
        )

    def _cmd_summary(self) -> str:
        s = self.logger.get_today_summary()
        return (
            f"📊 <b>TODAY'S SUMMARY</b>\n"
            f"Trades  : {s['trades']} ({s['wins']}W / {s['losses']}L)\n"
            f"Win Rate: {s.get('win_rate', 0):.0f}%\n"
            f"PnL     : ${s['pnl']:+.2f}\n"
            f"Best    : ${s.get('best_trade', 0):+.2f}\n"
            f"Worst   : ${s.get('worst_trade', 0):+.2f}"
        )

    def _cmd_trades(self) -> str:
        trades = self.logger.get_last_n_trades(5)
        if not trades:
            return "No trades recorded yet."

        lines = ["<b>Last 5 Trades:</b>"]
        for t in trades:
            emoji = "✅" if t.get("pnl", 0) > 0 else "❌"
            lines.append(f"{emoji} ${t.get('pnl', 0):+.2f} | {t.get('exit_reason', '')} | "
                         f"{t.get('time', '')[:16]}")
        return "\n".join(lines)

    def _cmd_risk(self) -> str:
        s = self.risk.get_status()
        return (
            f"🛡️ <b>RISK STATUS</b>\n"
            f"Daily PnL    : ${s['daily_pnl']:+.2f}\n"
            f"Remaining    : ${s['daily_remaining']:.2f}\n"
            f"Daily Trades : {s['daily_trades']}\n"
            f"Sessions     : {s['session_trades']}\n"
            f"Locked       : {'YES ⛔' if s['is_locked'] else 'No'}\n"
            f"Balance      : ${s['balance']:,.2f}\n"
            f"Drawdown     : {s['dd_from_start']:.1f}%"
        )

    def _cmd_balance(self) -> str:
        acc = self.mt5.get_account_info()
        return f"💰 Balance: ${acc.get('balance', 0):,.2f} | Equity: ${acc.get('equity', 0):,.2f}"

    def _cmd_regime(self) -> str:
        stats = self.regime.get_stats()
        if not stats:
            return "No regime data yet."
        lines = ["<b>Regime Distribution:</b>"]
        for r, s in sorted(stats.items(), key=lambda x: -x[1]["pct"]):
            lines.append(f"  {r}: {s['pct']:.1f}% ({s['count']} bars)")
        return "\n".join(lines)

    def _cmd_positions(self) -> str:
        positions = self.mt5.get_open_positions()
        if not positions:
            return "No open positions."
        lines = ["<b>Open Positions:</b>"]
        for p in positions:
            lines.append(
                f"{p['direction'].upper()} {p['volume']} @ {p['entry_price']:.2f}\n"
                f"  PnL: ${p['profit']:+.2f} | SL: {p['sl']:.2f} | TP: {p['tp']:.2f}"
            )
        return "\n".join(lines)

    # ═════════════════════════════════════════════════════
    # SHUTDOWN
    # ═════════════════════════════════════════════════════

    def _shutdown(self):
        """Graceful shutdown."""
        print("\n  Shutting down...")
        self.notifier.stop_polling()

        if self.mt5.has_open_position():
            print("  ⚠️ Open position — NOT auto-closing. Manage manually.")

        # Daily summary
        summary = self.logger.get_today_summary()
        if summary.get("trades", 0) > 0:
            balance = self.mt5.get_balance()
            summary["balance"] = balance
            summary["asian"] = self.risk.session_trades.get("asian", 0)
            summary["london"] = self.risk.session_trades.get("london", 0)
            summary["ny"] = self.risk.session_trades.get("new_york", 0)
            self.notifier.daily_summary(summary)

        self.mt5.disconnect()
        self.notifier.bot_stopped("Shutdown")
        self.logger.log_event("BOT_STOPPED")
        self.risk.print_status()
        print("  👋 MIDAS v2 stopped.")


# ═════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MIDAS v2 — Live Trading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals without executing trades")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo",
                        help="Account mode (default: demo)")
    args = parser.parse_args()

    if args.mode == "live" and not args.dry_run:
        print("\n  ⚠️  LIVE MODE — Real money!")
        if input("  Type 'CONFIRM': ") != "CONFIRM":
            print("  Aborted.")
            return

    bot = MidasBot(dry_run=args.dry_run, mode=args.mode)
    bot.start()


if __name__ == "__main__":
    main()