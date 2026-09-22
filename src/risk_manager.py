"""
Project MIDAS v2 — Live Risk Manager
Real-time risk tracking for The5ers compliance.

Tracks:
  - Daily P&L vs $100 cap (safety buffer below The5ers $250)
  - Session trade counts (2 per session)
  - Cooldown between trades (10 min)
  - Per-trade risk cap ($40)
  - Account drawdown vs The5ers -10% limit
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict

from config import (
    MAX_DAILY_LOSS, MAX_TRADE_RISK, COOLDOWN_SECONDS,
    SESSION_LIMITS, SESSIONS, ACCOUNT_BALANCE,
    MAX_LOSS_LEVEL, DAILY_LOSS_LIMIT_5ERS,
    LOT_SIZE_MIN, LOT_SIZE_MAX, XAUUSD_POINT_VALUE, PIP_VALUE,
    LOGS_DIR, DAILY_STATE_FILE,
)


class RiskManager:
    """
    Live risk management engine.
    Enforces The5ers rules + our safety buffer.
    State persists to disk so bot can restart safely.
    """

    def __init__(self, state_file: str = None):
        self.state_file = Path(state_file or DAILY_STATE_FILE)

        # Daily tracking
        self.current_date = None
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.session_trades = {
            "asian": 0, "london": 0, "new_york": 0,
            "london_ny_overlap": 0, "off": 0,
        }
        self.is_daily_locked = False

        # Trade tracking
        self.last_trade_time = None
        self.open_trade_ticket = None

        # Account tracking
        self.starting_balance = ACCOUNT_BALANCE
        self.current_balance = ACCOUNT_BALANCE

        # Load persisted state
        self._load_state()

    # ─── Pre-Trade Checks ─────────────────────────────────

    def can_trade(self, session: str, current_time: datetime = None) -> tuple:
        """
        Full pre-flight risk check.

        Returns:
            (allowed: bool, reason: str)
        """
        now = current_time or datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Reset daily state if new day
        if self.current_date != today:
            self._reset_daily(today)

        # Check daily lock
        if self.is_daily_locked:
            return False, f"Daily loss limit hit (${self.daily_pnl:.2f})"

        # Check daily PnL
        if self.daily_pnl <= -MAX_DAILY_LOSS:
            self.is_daily_locked = True
            return False, f"Daily loss limit: ${self.daily_pnl:.2f} <= -${MAX_DAILY_LOSS}"

        # Check account drawdown (The5ers -10%)
        if self.current_balance <= MAX_LOSS_LEVEL:
            return False, f"Account drawdown limit: ${self.current_balance:.2f} <= ${MAX_LOSS_LEVEL}"

        # Check session
        if session == "off":
            return False, "No trading outside sessions"

        # Session limit
        sess_key = session if session in SESSION_LIMITS else "off"
        limit = SESSION_LIMITS.get(sess_key, 0)
        current = self.session_trades.get(sess_key, 0)
        if current >= limit:
            return False, f"Session limit: {sess_key} ({current}/{limit})"

        # Cooldown
        if self.last_trade_time is not None:
            elapsed = (now - self.last_trade_time).total_seconds()
            if elapsed < COOLDOWN_SECONDS:
                remaining = int(COOLDOWN_SECONDS - elapsed)
                return False, f"Cooldown: {remaining}s remaining"

        # Open position check
        if self.open_trade_ticket is not None:
            return False, "Position already open"

        return True, "Clear to trade"

    def validate_trade_risk(self, sl_distance: float, lot_size: float) -> tuple:
        """
        Validate that a specific trade is within risk limits.

        Returns:
            (allowed: bool, adjusted_lot: float, reason: str)
        """
        # Calculate dollar risk
        risk = sl_distance * lot_size * XAUUSD_POINT_VALUE

        if risk > MAX_TRADE_RISK:
            # Try to reduce lot
            max_lot = MAX_TRADE_RISK / (XAUUSD_POINT_VALUE * sl_distance)
            adjusted_lot = max(LOT_SIZE_MIN, round(max_lot * 100) / 100)

            new_risk = sl_distance * adjusted_lot * XAUUSD_POINT_VALUE
            if new_risk > MAX_TRADE_RISK:
                return False, 0, f"Risk ${new_risk:.2f} exceeds ${MAX_TRADE_RISK} even at min lot"

            return True, adjusted_lot, f"Lot adjusted {lot_size}→{adjusted_lot} for risk cap"

        # Check remaining daily budget
        remaining = MAX_DAILY_LOSS + self.daily_pnl
        if risk > remaining:
            return False, 0, f"Risk ${risk:.2f} exceeds daily remaining ${remaining:.2f}"

        return True, lot_size, "OK"

    # ─── Trade Recording ──────────────────────────────────

    def record_trade_open(self, ticket: int, session: str, current_time: datetime = None):
        """Record that a trade was opened."""
        now = current_time or datetime.now()

        self.open_trade_ticket = ticket
        self.last_trade_time = now
        self.daily_trades += 1

        sess_key = session if session in self.session_trades else "off"
        self.session_trades[sess_key] = self.session_trades.get(sess_key, 0) + 1

        self._save_state()

    def record_trade_close(self, pnl: float, balance: float = None):
        """Record that a trade was closed."""
        self.daily_pnl += pnl
        self.open_trade_ticket = None

        if balance is not None:
            self.current_balance = balance

        # Check if daily limit now hit
        if self.daily_pnl <= -MAX_DAILY_LOSS:
            self.is_daily_locked = True

        self._save_state()

    def update_balance(self, balance: float):
        """Update current account balance."""
        self.current_balance = balance
        self._save_state()

    # ─── Session Detection ────────────────────────────────

    def get_session(self, dt: datetime = None) -> str:
        """Determine current trading session from UTC hour."""
        hour = (dt or datetime.utcnow()).hour

        if 12 <= hour < 14:
            return "london_ny_overlap"
        for name, times in SESSIONS.items():
            if times["start"] <= hour < times["end"]:
                return name
        return "off"

    # ─── State Management ─────────────────────────────────

    def _reset_daily(self, date_str: str):
        """Reset daily counters."""
        self.current_date = date_str
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.is_daily_locked = False
        self.session_trades = {k: 0 for k in self.session_trades}
        self._save_state()

    def _save_state(self):
        """Persist state to disk."""
        state = {
            "current_date": self.current_date,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "session_trades": self.session_trades,
            "is_daily_locked": self.is_daily_locked,
            "last_trade_time": str(self.last_trade_time) if self.last_trade_time else None,
            "open_trade_ticket": self.open_trade_ticket,
            "current_balance": round(self.current_balance, 2),
            "updated_at": str(datetime.now()),
        }

        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"  WARNING: Failed to save risk state: {e}")

    def _load_state(self):
        """Load persisted state from disk."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            today = datetime.now().strftime("%Y-%m-%d")

            # Only restore if same day
            if state.get("current_date") == today:
                self.current_date = state["current_date"]
                self.daily_pnl = state.get("daily_pnl", 0)
                self.daily_trades = state.get("daily_trades", 0)
                self.session_trades = state.get("session_trades", self.session_trades)
                self.is_daily_locked = state.get("is_daily_locked", False)
                self.open_trade_ticket = state.get("open_trade_ticket")
                self.current_balance = state.get("current_balance", ACCOUNT_BALANCE)

                if state.get("last_trade_time"):
                    try:
                        self.last_trade_time = datetime.fromisoformat(
                            state["last_trade_time"].replace("Z", ""))
                    except (ValueError, AttributeError):
                        pass

                print(f"  Risk state restored: date={today}, "
                      f"daily_pnl=${self.daily_pnl:.2f}, trades={self.daily_trades}")
            else:
                print(f"  Risk state from different day ({state.get('current_date')}), resetting")
                self.current_balance = state.get("current_balance", ACCOUNT_BALANCE)

        except Exception as e:
            print(f"  WARNING: Failed to load risk state: {e}")

    def get_status(self) -> Dict:
        """Get current risk status for display."""
        return {
            "date": self.current_date,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "daily_remaining": round(MAX_DAILY_LOSS + self.daily_pnl, 2),
            "session_trades": dict(self.session_trades),
            "is_locked": self.is_daily_locked,
            "balance": round(self.current_balance, 2),
            "dd_from_start": round((self.starting_balance - self.current_balance) / self.starting_balance * 100, 2),
        }

    def print_status(self):
        """Print formatted risk status."""
        s = self.get_status()
        print(f"\n  RISK STATUS:")
        print(f"    Daily PnL:     ${s['daily_pnl']:+.2f} (remaining: ${s['daily_remaining']:.2f})")
        print(f"    Daily Trades:  {s['daily_trades']}")
        print(f"    Sessions:      {s['session_trades']}")
        print(f"    Balance:       ${s['balance']:,.2f} (DD: {s['dd_from_start']:.1f}%)")
        if s['is_locked']:
            print(f"    ⛔ DAILY LOCKED")


# Make Dict available for type hints
from typing import Dict