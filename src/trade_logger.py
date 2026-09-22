"""
Project MIDAS v2 — Trade Logger (Full Featured)
=================================================
Logs:
  - Every trade open/close with full details
  - Daily summaries with PnL, win rate, sessions
  - Running cumulative stats
  - System events (startup, shutdown, errors)

Files:
  logs/trades.json       — All trade events (append)
  logs/daily_YYYY-MM-DD.json — Per-day trade log
  logs/cumulative.json   — Running stats across all days
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import LOGS_DIR, TRADE_LOG_FILE


class TradeLogger:
    """Comprehensive trade and event logging."""

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.trade_log_file = Path(TRADE_LOG_FILE)
        self.cumulative_file = LOGS_DIR / "cumulative.json"
        self._trades = self._load_json(self.trade_log_file, default=[])
        self._cumulative = self._load_json(self.cumulative_file, default={
            "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0, "best_trade": 0, "worst_trade": 0,
            "start_date": str(datetime.now().date()),
            "trading_days": 0,
        })

    # ─── Trade Logging ────────────────────────────────────

    def log_trade_open(self, trade: Dict):
        """Log a trade opening with full details."""
        entry = {
            "event": "OPEN",
            "time": str(datetime.now()),
            "ticket": trade.get("ticket"),
            "direction": trade.get("direction"),
            "lot_size": trade.get("lot_size"),
            "entry_price": trade.get("entry_price"),
            "sl_price": trade.get("sl_price"),
            "tp_price": trade.get("tp_price"),
            "source": trade.get("source", ""),
            "confidence": trade.get("confidence", 0),
            "regime": trade.get("regime", ""),
            "session": trade.get("session", ""),
            "risk_dollars": trade.get("risk_dollars", 0),
            "e_prob": trade.get("e_prob"),
            "lot_factors": trade.get("lot_factors"),
        }
        self._append(entry)
        self._append_daily(entry)
        print(f"  📝 Logged: OPEN {entry['direction']} {entry['lot_size']} @ {entry['entry_price']}")

    def log_trade_close(self, ticket: int, exit_price: float, pnl: float,
                        reason: str, balance: float = 0, duration_mins: int = 0):
        """Log a trade closure with PnL."""
        entry = {
            "event": "CLOSE",
            "time": str(datetime.now()),
            "ticket": ticket,
            "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "exit_reason": reason,
            "balance_after": round(balance, 2),
            "duration_mins": duration_mins,
        }
        self._append(entry)
        self._append_daily(entry)

        # Update cumulative
        self._cumulative["total_trades"] += 1
        self._cumulative["total_pnl"] = round(self._cumulative["total_pnl"] + pnl, 2)
        if pnl > 0:
            self._cumulative["wins"] += 1
        else:
            self._cumulative["losses"] += 1
        self._cumulative["best_trade"] = max(self._cumulative["best_trade"], pnl)
        self._cumulative["worst_trade"] = min(self._cumulative["worst_trade"], pnl)
        self._save_json(self.cumulative_file, self._cumulative)

        emoji = "✅" if pnl > 0 else "❌"
        print(f"  📝 Logged: CLOSE {emoji} PnL=${pnl:+.2f} ({reason})")

    def log_tsl_change(self, ticket: int, old_sl: float, new_sl: float, reason: str):
        """Log a TSL adjustment."""
        entry = {
            "event": "TSL_UPDATE",
            "time": str(datetime.now()),
            "ticket": ticket,
            "old_sl": round(old_sl, 2),
            "new_sl": round(new_sl, 2),
            "reason": reason,
        }
        self._append(entry)
        self._append_daily(entry)

    def log_event(self, event: str, details: Dict = None):
        """Log a system event."""
        entry = {
            "event": event,
            "time": str(datetime.now()),
            "details": details or {},
        }
        self._append(entry)

    # ─── Daily Queries ────────────────────────────────────

    def get_today_trades(self) -> List[Dict]:
        """Get all closed trades from today."""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_file = LOGS_DIR / f"daily_{today}.json"
        events = self._load_json(daily_file, default=[])
        return [e for e in events if e.get("event") == "CLOSE"]

    def get_today_summary(self) -> Dict:
        """Get today's trading summary."""
        trades = self.get_today_trades()
        if not trades:
            return {"date": datetime.now().strftime("%Y-%m-%d"),
                    "trades": 0, "pnl": 0, "wins": 0, "losses": 0}

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        pnl = sum(t.get("pnl", 0) for t in trades)

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "pnl": round(pnl, 2),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "best_trade": round(max(t.get("pnl", 0) for t in trades), 2),
            "worst_trade": round(min(t.get("pnl", 0) for t in trades), 2),
            "avg_trade": round(pnl / len(trades), 2),
        }

    def get_last_n_trades(self, n: int = 5) -> List[Dict]:
        """Get the last N closed trades across all days."""
        closes = [e for e in self._trades if e.get("event") == "CLOSE"]
        return closes[-n:]

    def get_cumulative(self) -> Dict:
        """Get cumulative stats."""
        c = self._cumulative.copy()
        total = c.get("total_trades", 0)
        wins = c.get("wins", 0)
        c["win_rate"] = round(wins / total * 100, 1) if total > 0 else 0
        c["avg_pnl"] = round(c["total_pnl"] / total, 2) if total > 0 else 0
        return c

    # ─── File Operations ──────────────────────────────────

    def _append(self, entry: Dict):
        """Append to main trade log."""
        self._trades.append(entry)
        self._save_json(self.trade_log_file, self._trades)

    def _append_daily(self, entry: Dict):
        """Append to today's daily log."""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_file = LOGS_DIR / f"daily_{today}.json"
        events = self._load_json(daily_file, default=[])
        events.append(entry)
        self._save_json(daily_file, events)

    def _load_json(self, path: Path, default=None):
        """Load JSON file safely."""
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception):
                pass
        return default if default is not None else {}

    def _save_json(self, path: Path, data):
        """Save JSON file safely."""
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"  WARNING: Failed to save {path.name}: {e}")