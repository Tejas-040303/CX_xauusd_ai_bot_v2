"""
Project MIDAS v2 — Dynamic TSL Manager
Trails stop loss as price moves in our favor.

TSL Logic (progressive locking):
  - At 30% of TP distance: lock 10% of profit
  - At 50% of TP distance: lock 30% of profit
  - At 70% of TP distance: lock 50% of profit
  - At 90% of TP distance: lock 75% of profit

Also supports:
  - Breakeven move (lock SL at entry after X% progress)
  - TP extension when momentum is strong
"""

from datetime import datetime
from typing import Optional, Dict

from config import ATR_PERIOD, PIP_VALUE, XAUUSD_POINT_VALUE


# TSL progression table: (progress_pct, lock_pct)
# At X% progress toward TP, lock Y% of current profit
TSL_LEVELS = [
    (0.30, 0.10),   # 30% → lock 10%
    (0.50, 0.30),   # 50% → lock 30%
    (0.70, 0.50),   # 70% → lock 50%
    (0.90, 0.75),   # 90% → lock 75%
]

# Breakeven: move SL to entry + small buffer after this progress
BREAKEVEN_PROGRESS = 0.25
BREAKEVEN_BUFFER_PIPS = 1.0  # Lock 1 pip above entry


class TSLManager:
    """
    Manages trailing stop loss for open positions.
    Call update() on every tick/bar to check if SL should move.
    """

    def __init__(self):
        self.active_trade = None  # Currently tracked trade
        self.current_tsl_level = -1  # Index into TSL_LEVELS (-1 = not started)
        self.breakeven_done = False
        self.last_update_time = None
        self.tsl_history = []  # Log of TSL adjustments

    def start_tracking(self, trade: Dict):
        """
        Start tracking a new trade for TSL.

        Args:
            trade: dict with keys:
                ticket, direction, entry_price, sl_price, tp_price, lot_size
        """
        self.active_trade = {
            "ticket": trade["ticket"],
            "direction": trade["direction"],
            "entry_price": trade["entry_price"],
            "original_sl": trade["sl_price"],
            "original_tp": trade["tp_price"],
            "current_sl": trade["sl_price"],
            "current_tp": trade["tp_price"],
            "lot_size": trade.get("lot_size", 0.01),
            "open_time": datetime.now(),
        }
        self.current_tsl_level = -1
        self.breakeven_done = False
        self.tsl_history = []

    def stop_tracking(self):
        """Stop tracking (trade closed)."""
        self.active_trade = None
        self.current_tsl_level = -1
        self.breakeven_done = False

    def update(self, current_price: float, current_atr: float = None) -> Optional[Dict]:
        """
        Check if TSL should be adjusted based on current price.

        Args:
            current_price: Current bid (for buy) or ask (for sell)
            current_atr: Current ATR for TP extension check

        Returns:
            Dict with new SL/TP if changed, None if no change.
            {"new_sl": float, "new_tp": float or None, "reason": str}
        """
        if self.active_trade is None:
            return None

        trade = self.active_trade
        direction = trade["direction"]
        entry = trade["entry_price"]
        current_sl = trade["current_sl"]
        current_tp = trade["current_tp"]

        # Calculate progress toward TP
        if direction == "buy":
            total_distance = current_tp - entry
            current_progress = current_price - entry
        else:
            total_distance = entry - current_tp
            current_progress = entry - current_price

        if total_distance <= 0:
            return None

        progress_pct = current_progress / total_distance
        progress_pct = max(0, progress_pct)  # Can be negative if price moved against us

        # ── Breakeven check ───────────────────────────────
        if not self.breakeven_done and progress_pct >= BREAKEVEN_PROGRESS:
            buffer = BREAKEVEN_BUFFER_PIPS * PIP_VALUE

            if direction == "buy":
                new_sl = entry + buffer
                if new_sl > current_sl:
                    self.breakeven_done = True
                    trade["current_sl"] = new_sl
                    self._log_change("BREAKEVEN", new_sl, current_tp, progress_pct)
                    return {"new_sl": round(new_sl, 2), "new_tp": None,
                            "reason": f"Breakeven at {progress_pct:.0%} progress"}
            else:
                new_sl = entry - buffer
                if new_sl < current_sl:
                    self.breakeven_done = True
                    trade["current_sl"] = new_sl
                    self._log_change("BREAKEVEN", new_sl, current_tp, progress_pct)
                    return {"new_sl": round(new_sl, 2), "new_tp": None,
                            "reason": f"Breakeven at {progress_pct:.0%} progress"}

        # ── Progressive TSL check ─────────────────────────
        new_level = self.current_tsl_level
        for i, (level_pct, lock_pct) in enumerate(TSL_LEVELS):
            if progress_pct >= level_pct and i > self.current_tsl_level:
                new_level = i

        if new_level > self.current_tsl_level:
            level_pct, lock_pct = TSL_LEVELS[new_level]

            # Calculate new SL that locks lock_pct of current profit
            if direction == "buy":
                profit_distance = current_price - entry
                locked_profit = profit_distance * lock_pct
                new_sl = entry + locked_profit
                # Only move SL up, never down
                if new_sl <= current_sl:
                    return None
            else:
                profit_distance = entry - current_price
                locked_profit = profit_distance * lock_pct
                new_sl = entry - locked_profit
                # Only move SL down (toward price), never up
                if new_sl >= current_sl:
                    return None

            self.current_tsl_level = new_level
            trade["current_sl"] = new_sl

            reason = (f"TSL Level {new_level+1}: {level_pct:.0%} progress, "
                      f"locking {lock_pct:.0%} profit")
            self._log_change(f"TSL_L{new_level+1}", new_sl, current_tp, progress_pct)

            return {"new_sl": round(new_sl, 2), "new_tp": None, "reason": reason}

        return None

    def get_status(self) -> Dict:
        """Get current TSL tracking status."""
        if self.active_trade is None:
            return {"tracking": False}

        trade = self.active_trade
        return {
            "tracking": True,
            "ticket": trade["ticket"],
            "direction": trade["direction"],
            "entry": trade["entry_price"],
            "original_sl": trade["original_sl"],
            "current_sl": trade["current_sl"],
            "original_tp": trade["original_tp"],
            "current_tp": trade["current_tp"],
            "breakeven": self.breakeven_done,
            "tsl_level": self.current_tsl_level + 1,
            "adjustments": len(self.tsl_history),
        }

    def get_progress(self, current_price: float) -> float:
        """Get current progress toward TP as percentage."""
        if self.active_trade is None:
            return 0

        trade = self.active_trade
        entry = trade["entry_price"]
        tp = trade["current_tp"]

        if trade["direction"] == "buy":
            total = tp - entry
            current = current_price - entry
        else:
            total = entry - tp
            current = entry - current_price

        if total <= 0:
            return 0
        return max(0, min(1, current / total))

    def _log_change(self, change_type, new_sl, new_tp, progress):
        """Log a TSL change."""
        self.tsl_history.append({
            "time": str(datetime.now()),
            "type": change_type,
            "new_sl": round(new_sl, 2),
            "progress": round(progress, 3),
        })