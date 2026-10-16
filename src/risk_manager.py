"""
Project MIDAS v2 — Live Risk Manager
Real-time risk tracking. Hard limits follow The5ers-style prop rules
(we don't trade a The5ers account; the rules are kept as a framework).

Every dollar limit is derived from the live MT5 account, not from config:

  initial balance   account's deposits (or config ACCOUNT_BALANCE)
                    → fixed max-loss floor: initial × (1 - THE5ERS_MAX_LOSS_PCT)
  day's base        higher of balance / equity at broker midnight, the way
                    The5ers measures its daily loss; saved to disk so a
                    restart mid-day can't reset it
                    → daily stop      = base × MAX_DAILY_LOSS_PCT
                    → per-trade risk  = base × MAX_TRADE_RISK_PCT
  day's loss        base - current equity (floating P/L counts, as at The5ers)

Also tracks session trade counts and the cooldown between trades.
"""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict

from config import (
    COOLDOWN_SECONDS, SESSION_LIMITS, SESSIONS,
    ACCOUNT_BALANCE, BACKTEST_INITIAL_CAPITAL,
    MAX_DAILY_LOSS_PCT, MAX_TRADE_RISK_PCT,
    THE5ERS_DAILY_LOSS_PCT, THE5ERS_MAX_LOSS_PCT,
    LOT_SIZE_MIN, XAUUSD_POINT_VALUE, DAILY_STATE_FILE,
)
from broker_time import expected_offset_now


def _to_utc(dt: Optional[datetime]) -> datetime:
    """Return dt as a timezone-aware UTC datetime (now if None).

    Naive datetimes are treated as machine-local time (what datetime.now()
    returns) — this also covers last_trade_time values persisted by older
    versions of the state file.
    """
    if dt is None:
        return datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc)


class RiskManager:
    """
    Live risk management engine.
    Enforces the hard (The5ers-style) rules + our safety buffer.
    State persists to disk so bot can restart safely.
    """

    def __init__(self, state_file: str = None):
        self.state_file = Path(state_file or DAILY_STATE_FILE)

        # Daily tracking
        self.current_date = None      # Trading day (broker calendar), "YYYY-MM-DD"
        self.day_start_value = None   # The day's base (higher of balance/equity at day start)
        self.daily_pnl = 0.0          # Closed-trade PnL today (for reports)
        self.daily_trades = 0
        self.session_trades = {
            "asian": 0, "london": 0, "new_york": 0,
            "london_ny_overlap": 0, "off": 0,
        }
        self.is_daily_locked = False

        # Trade tracking
        self.last_trade_time = None
        self.open_trade_ticket = None

        # Account tracking (replaced with MT5 values at startup)
        self.initial_balance = ACCOUNT_BALANCE or BACKTEST_INITIAL_CAPITAL
        self.current_balance = self.initial_balance
        self.current_equity = None
        self.account_login = None     # MT5 account the saved state belongs to
        # Trading day = broker calendar day (The5ers resets at server midnight)
        self.day_offset_hours = expected_offset_now()

        # Load persisted state
        self._load_state()

    # ─── Limits (derived from the account) ────────────────

    @property
    def starting_balance(self) -> float:
        return self.initial_balance

    @property
    def day_base(self) -> float:
        return self.day_start_value or self.current_balance

    @property
    def max_daily_loss(self) -> float:
        """Our daily stop in $ for today."""
        return round(self.day_base * MAX_DAILY_LOSS_PCT, 2)

    @property
    def max_trade_risk(self) -> float:
        """Max $ loss per trade (if SL is hit) for today."""
        return round(self.day_base * MAX_TRADE_RISK_PCT, 2)

    @property
    def the5ers_daily_limit(self) -> float:
        return round(self.day_base * THE5ERS_DAILY_LOSS_PCT, 2)

    @property
    def max_loss_level(self) -> float:
        """Fixed floor: The5ers max loss from the initial balance."""
        return round(self.initial_balance * (1 - THE5ERS_MAX_LOSS_PCT), 2)

    @property
    def day_loss(self) -> float:
        """How much today has lost so far, floating P/L included (>= 0)."""
        equity = self.current_equity if self.current_equity is not None else self.current_balance
        return max(0.0, self.day_base - equity, -self.daily_pnl)

    @property
    def day_pnl(self) -> float:
        """Today's result vs. the day's base (negative = loss)."""
        if self.day_loss > 0:
            return -self.day_loss
        equity = self.current_equity if self.current_equity is not None else self.current_balance
        return equity - self.day_base

    # ─── Account updates ──────────────────────────────────

    def set_initial_balance(self, balance: float):
        """The account's starting balance (for the fixed max-loss floor)."""
        if balance and balance > 0:
            self.initial_balance = float(balance)

    def update_account(self, balance: float, equity: float = None, now: datetime = None):
        """Feed in the latest MT5 balance/equity (call before can_trade)."""
        if balance is None:
            return
        self.current_balance = float(balance)
        self.current_equity = float(equity) if equity is not None else float(balance)
        self._ensure_day(_to_utc(now))
        if self.day_start_value is None:
            # Day started before we had account data (or old state file)
            self.day_start_value = max(self.current_balance, self.current_equity)
            self._save_state()

    def update_balance(self, balance: float):
        """Update current account balance (kept for older callers)."""
        self.update_account(balance)

    def _day_key(self, now: datetime) -> str:
        return (now + timedelta(hours=self.day_offset_hours)).strftime("%Y-%m-%d")

    def _ensure_day(self, now: datetime):
        today = self._day_key(now)
        if self.current_date != today:
            self._reset_daily(today)

    # ─── Pre-Trade Checks ─────────────────────────────────

    def can_trade(self, session: str, current_time: datetime = None) -> tuple:
        """
        Full pre-flight risk check.

        Returns:
            (allowed: bool, reason: str)
        """
        now = _to_utc(current_time)
        self._ensure_day(now)

        # Check daily lock
        if self.is_daily_locked:
            return False, f"Daily loss limit hit (-${self.day_loss:.2f})"

        # Check today's loss (equity-based, like The5ers)
        if self.day_loss >= self.max_daily_loss:
            self.is_daily_locked = True
            self._save_state()
            return False, (f"Daily loss limit: -${self.day_loss:.2f} >= ${self.max_daily_loss:.2f} "
                           f"({MAX_DAILY_LOSS_PCT:.0%} of ${self.day_base:,.2f})")

        # Check account drawdown (The5ers max loss, fixed floor)
        low = min(self.current_balance, self.current_equity or self.current_balance)
        if low <= self.max_loss_level:
            return False, f"Account drawdown limit: ${low:.2f} <= ${self.max_loss_level:.2f}"

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
        # NaN compares False everywhere below, so it would sail through every check
        if not (math.isfinite(sl_distance) and sl_distance > 0
                and math.isfinite(lot_size) and lot_size > 0):
            return False, 0, f"Invalid SL distance/lot ({sl_distance}, {lot_size})"

        max_trade_risk = self.max_trade_risk
        risk = sl_distance * lot_size * XAUUSD_POINT_VALUE

        reason = "OK"
        if risk > max_trade_risk:
            # Reduce the lot. Round DOWN to the 0.01 step: rounding to nearest
            # (0.015 → 0.02) could land back above the cap and block a trade
            # that fits at 0.01.
            max_lot = max_trade_risk / (XAUUSD_POINT_VALUE * sl_distance)
            adjusted_lot = math.floor(max_lot * 100 + 1e-9) / 100
            if adjusted_lot < LOT_SIZE_MIN:
                min_risk = sl_distance * LOT_SIZE_MIN * XAUUSD_POINT_VALUE
                return False, 0, f"Risk ${min_risk:.2f} exceeds ${max_trade_risk:.2f} even at min lot"

            reason = f"Lot adjusted {lot_size}→{adjusted_lot} for risk cap"
            lot_size = adjusted_lot
            risk = sl_distance * lot_size * XAUUSD_POINT_VALUE
            # ...and still subject to the daily budget check below

        # Check remaining daily budget
        remaining = self.max_daily_loss - self.day_loss
        if risk > remaining:
            return False, 0, f"Risk ${risk:.2f} exceeds daily remaining ${remaining:.2f}"

        return True, lot_size, reason

    # ─── Trade Recording ──────────────────────────────────

    def record_trade_open(self, ticket: int, session: str, current_time: datetime = None):
        """Record that a trade was opened."""
        now = _to_utc(current_time)

        self.open_trade_ticket = ticket
        self.last_trade_time = now
        self.daily_trades += 1

        sess_key = session if session in self.session_trades else "off"
        self.session_trades[sess_key] = self.session_trades.get(sess_key, 0) + 1

        self._save_state()

    def bind_account(self, login) -> bool:
        """Tie the saved state to the MT5 account it belongs to.

        If the terminal is now logged into a different account, today's PnL,
        base, trade counts, cooldown and open ticket belong to the old
        account: reset them. Returns True when that happened.
        """
        if login is None:
            return False
        switched = self.account_login is not None and self.account_login != login
        if switched:
            print(f"  MT5 account changed ({self.account_login} → {login}): "
                  f"resetting today's risk counters")
            self.current_date = None
            self.day_start_value = None
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.session_trades = {k: 0 for k in self.session_trades}
            self.is_daily_locked = False
            self.last_trade_time = None
            self.open_trade_ticket = None
        self.account_login = login
        self._save_state()
        return switched

    def adopt_open_trade(self, ticket: int):
        """Track a position that is open on MT5 but missing from saved state
        (e.g. bot restarted on a new day while a trade was running)."""
        self.open_trade_ticket = ticket
        self._save_state()

    def record_trade_close(self, pnl: float, balance: float = None):
        """Record that a trade was closed."""
        self.daily_pnl += pnl
        self.open_trade_ticket = None

        if balance is not None:
            # One trade at a time: with it closed, equity == balance
            self.current_balance = self.current_equity = balance

        # Check if daily limit now hit
        if self.day_loss >= self.max_daily_loss:
            self.is_daily_locked = True

        self._save_state()

    # ─── Session Detection ────────────────────────────────

    def get_session(self, dt: datetime = None) -> str:
        """Determine current trading session from UTC hour."""
        hour = _to_utc(dt).hour

        if 12 <= hour < 14:
            return "london_ny_overlap"
        for name, times in SESSIONS.items():
            if times["start"] <= hour < times["end"]:
                return name
        return "off"

    # ─── State Management ─────────────────────────────────

    def _reset_daily(self, date_str: str):
        """New trading day: reset counters and take the day's base."""
        self.current_date = date_str
        self.day_start_value = None
        if self.current_balance:
            equity = self.current_equity if self.current_equity is not None else self.current_balance
            self.day_start_value = max(self.current_balance, equity)
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.is_daily_locked = False
        self.session_trades = {k: 0 for k in self.session_trades}
        self._save_state()

    def _save_state(self):
        """Persist state to disk."""
        state = {
            "current_date": self.current_date,
            "day_start_value": round(self.day_start_value, 2) if self.day_start_value else None,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "session_trades": self.session_trades,
            "is_daily_locked": self.is_daily_locked,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "open_trade_ticket": self.open_trade_ticket,
            "current_balance": round(self.current_balance, 2),
            "account_login": self.account_login,
            "updated_at": datetime.now(timezone.utc).isoformat(),
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

            today = self._day_key(datetime.now(timezone.utc))
            self.account_login = state.get("account_login")
            self.current_balance = state.get("current_balance", self.current_balance)

            # Only restore if same day
            if state.get("current_date") == today:
                self.current_date = state["current_date"]
                self.day_start_value = state.get("day_start_value")
                self.daily_pnl = state.get("daily_pnl", 0)
                self.daily_trades = state.get("daily_trades", 0)
                self.session_trades = state.get("session_trades", self.session_trades)
                self.is_daily_locked = state.get("is_daily_locked", False)
                self.open_trade_ticket = state.get("open_trade_ticket")

                if state.get("last_trade_time"):
                    try:
                        self.last_trade_time = _to_utc(datetime.fromisoformat(
                            state["last_trade_time"].replace("Z", "+00:00")))
                    except (ValueError, AttributeError):
                        pass

                print(f"  Risk state restored: date={today}, "
                      f"daily_pnl=${self.daily_pnl:.2f}, trades={self.daily_trades}")
            else:
                print(f"  Risk state from different day ({state.get('current_date')}), resetting")

        except Exception as e:
            print(f"  WARNING: Failed to load risk state: {e}")

    def get_status(self) -> Dict:
        """Get current risk status for display."""
        return {
            "date": self.current_date,
            "daily_pnl": round(self.daily_pnl, 2),
            "day_loss": round(self.day_loss, 2),
            "daily_trades": self.daily_trades,
            "daily_remaining": round(self.max_daily_loss - self.day_loss, 2),
            "day_base": round(self.day_base, 2),
            "max_daily_loss": self.max_daily_loss,
            "max_trade_risk": self.max_trade_risk,
            "the5ers_daily_limit": self.the5ers_daily_limit,
            "max_loss_level": self.max_loss_level,
            "session_trades": dict(self.session_trades),
            "is_locked": self.is_daily_locked,
            "balance": round(self.current_balance, 2),
            "dd_from_start": round((self.initial_balance - self.current_balance) / self.initial_balance * 100, 2),
        }

    def print_status(self):
        """Print formatted risk status."""
        s = self.get_status()
        print(f"\n  RISK STATUS:")
        print(f"    Daily PnL:     ${s['daily_pnl']:+.2f} (remaining: ${s['daily_remaining']:.2f} "
              f"of ${s['max_daily_loss']:.2f})")
        print(f"    Daily Trades:  {s['daily_trades']}")
        print(f"    Sessions:      {s['session_trades']}")
        print(f"    Balance:       ${s['balance']:,.2f} (DD: {s['dd_from_start']:.1f}%)")
        if s['is_locked']:
            print(f"    ⛔ DAILY LOCKED")
