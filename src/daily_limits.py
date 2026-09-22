"""
Project MIDAS v2 — Daily Limits Tracker
Tracks daily P&L, trade counts, and session counts.
Persists to disk (DAILY_STATE_FILE) so state survives bot restarts.

Enforces:
  - MAX_DAILY_LOSS: our internal buffer ($100)
  - DAILY_LOSS_LIMIT_5ERS: The5ers hard limit ($250)
  - Session limits (2 trades per session)
  - Cooldown (10 min between trades)
"""

import json
from datetime import datetime, date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DAILY_STATE_FILE, MAX_DAILY_LOSS, DAILY_LOSS_LIMIT_5ERS,
    SESSION_LIMITS, COOLDOWN_SECONDS,
)


class DailyLimits:
    """
    Persistent daily state tracker.
    Resets automatically at the start of each new trading day.
    Saves to disk after every update.
    """

    def __init__(self):
        self._state = self._load_or_init()

    # ─── State Init ───────────────────────────────────────

    def _empty_state(self, today: str) -> dict:
        return {
            "date": today,
            "daily_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "session_counts": {
                "asian": 0, "london": 0,
                "new_york": 0, "london_ny_overlap": 0,
            },
            "last_trade_time": None,
            "open_tickets": [],
        }

    def _load_or_init(self) -> dict:
        today = date.today().isoformat()
        if DAILY_STATE_FILE.exists():
            try:
                with open(DAILY_STATE_FILE) as f:
                    state = json.load(f)
                if state.get("date") == today:
                    return state
            except Exception:
                pass
        state = self._empty_state(today)
        self._save(state)
        return state

    def _save(self, state: dict = None):
        if state is None:
            state = self._state
        try:
            with open(DAILY_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"WARNING: Could not save daily state: {e}")

    # ─── Day Reset ────────────────────────────────────────

    def check_day_rollover(self):
        """Call at the start of each bar. Resets state if new day."""
        today = date.today().isoformat()
        if self._state["date"] != today:
            self._state = self._empty_state(today)
            self._save()
            return True  # New day
        return False

    # ─── Trade Recording ──────────────────────────────────

    def record_trade_opened(self, ticket: int, session: str, trade_time: datetime):
        """Record that a trade was opened."""
        self._state["trade_count"] += 1
        self._state["last_trade_time"] = trade_time.isoformat()

        if session in self._state["session_counts"]:
            self._state["session_counts"][session] += 1

        if ticket not in self._state["open_tickets"]:
            self._state["open_tickets"].append(ticket)

        self._save()

    def record_trade_closed(self, ticket: int, pnl: float):
        """Record that a trade was closed with a given P&L."""
        self._state["daily_pnl"] = round(self._state["daily_pnl"] + pnl, 2)
        if pnl > 0:
            self._state["win_count"] += 1
        if ticket in self._state["open_tickets"]:
            self._state["open_tickets"].remove(ticket)
        self._save()

    # ─── Limit Checks ─────────────────────────────────────

    def can_trade(self, session: str, current_time: datetime) -> tuple[bool, str]:
        """
        Check if a new trade is allowed right now.
        Returns (allowed: bool, reason: str).
        reason is empty string if allowed.
        """
        # Our internal daily loss buffer
        if self._state["daily_pnl"] <= -MAX_DAILY_LOSS:
            return False, (
                f"Internal daily loss limit hit "
                f"(${self._state['daily_pnl']:.2f} <= -${MAX_DAILY_LOSS})"
            )

        # The5ers hard limit (emergency stop)
        if self._state["daily_pnl"] <= -DAILY_LOSS_LIMIT_5ERS:
            return False, (
                f"The5ers daily loss limit hit "
                f"(${self._state['daily_pnl']:.2f} <= -${DAILY_LOSS_LIMIT_5ERS})"
            )

        # Off-session
        if session == "off":
            return False, "No trading outside defined sessions"

        # Session trade limit
        limit = SESSION_LIMITS.get(session, 0)
        current = self._state["session_counts"].get(session, 0)
        if current >= limit:
            return False, f"Session limit: {session} ({current}/{limit} trades used)"

        # Cooldown
        last = self._state.get("last_trade_time")
        if last:
            last_dt = datetime.fromisoformat(last)
            elapsed = (current_time - last_dt).total_seconds()
            if elapsed < COOLDOWN_SECONDS:
                remaining = int(COOLDOWN_SECONDS - elapsed)
                return False, f"Cooldown: {remaining}s remaining"

        return True, ""

    # ─── Getters ──────────────────────────────────────────

    @property
    def daily_pnl(self) -> float:
        return self._state["daily_pnl"]

    @property
    def trade_count(self) -> int:
        return self._state["trade_count"]

    @property
    def win_count(self) -> int:
        return self._state["win_count"]

    @property
    def open_tickets(self) -> list:
        return list(self._state["open_tickets"])

    @property
    def session_counts(self) -> dict:
        return dict(self._state["session_counts"])

    def get_summary(self) -> dict:
        return {
            "date": self._state["date"],
            "daily_pnl": self._state["daily_pnl"],
            "trade_count": self._state["trade_count"],
            "win_count": self._state["win_count"],
            "session_counts": self._state["session_counts"],
            "last_trade_time": self._state["last_trade_time"],
            "open_tickets": self._state["open_tickets"],
        }