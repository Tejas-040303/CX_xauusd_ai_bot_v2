"""
Project MIDAS v2 — Telegram Bot (Full Featured)
=================================================
Two-way: sends alerts AND receives commands.

Commands:
  /help      — List all commands
  /start     — Resume trading
  /stop      — Stop trading (graceful)
  /pause N   — Pause for N minutes (default 30)
  /status    — Current bot status
  /summary   — Today's trading summary
  /trades    — Last 5 trades
  /risk      — Risk manager state
  /balance   — Account balance
  /regime    — Current market regime
  /positions — Open positions

Alerts:
  - Trade opened (with full details)
  - Trade closed (PnL, reason)
  - TSL updates (every SL change)
  - Position updates (every 3 minutes while trade open)
  - Daily summary (end of day)
  - Errors and warnings
"""

import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

import requests

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


class TelegramNotifier:
    """
    Full-featured Telegram bot with command handling.
    Runs a polling listener in a background thread.
    """

    def __init__(self):
        self.token = TELEGRAM_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled = bool(self.token and self.chat_id
                            and self.token != "" and self.chat_id != "")

        # Command handlers: {"/cmd": callback_fn}
        self._command_handlers = {}
        self._last_update_id = 0
        self._polling_thread = None
        self._polling_active = False

        # Position update tracking
        self._last_position_update = None
        self.POSITION_UPDATE_INTERVAL = 180  # 3 minutes

        if not self.enabled:
            print("  ⚠️ Telegram: disabled (no token/chat_id)")

    # ─── Command Registration ─────────────────────────────

    def register_command(self, command: str, handler: Callable):
        """
        Register a command handler.
        handler receives (args: str) and returns a response string.
        """
        self._command_handlers[command] = handler

    def start_polling(self):
        """Start listening for commands in a background thread."""
        if not self.enabled:
            return

        self._polling_active = True
        self._polling_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="TelegramPoller"
        )
        self._polling_thread.start()
        print("  Telegram command listener started")

    def stop_polling(self):
        """Stop the command listener."""
        self._polling_active = False
        if self._polling_thread:
            self._polling_thread.join(timeout=5)

    def _poll_loop(self):
        """Background thread: poll for new messages."""
        while self._polling_active:
            try:
                resp = requests.get(
                    f"{self.base_url}/getUpdates",
                    params={"offset": self._last_update_id + 1, "timeout": 10},
                    timeout=15,
                )
                if resp.status_code != 200:
                    time.sleep(5)
                    continue

                data = resp.json()
                if not data.get("ok"):
                    time.sleep(5)
                    continue

                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    # Only respond to our chat
                    if chat_id != str(self.chat_id):
                        continue

                    if text.startswith("/"):
                        self._handle_command(text)

            except requests.exceptions.Timeout:
                continue
            except Exception as e:
                print(f"  Telegram poll error: {e}")
                time.sleep(10)

    def _handle_command(self, text: str):
        """Parse and execute a command."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Built-in /help
        if cmd == "/help":
            self._send_help()
            return

        handler = self._command_handlers.get(cmd)
        if handler:
            try:
                response = handler(args)
                if response:
                    self.send(response)
            except Exception as e:
                self.send(f"⚠️ Command error: {e}")
                traceback.print_exc()
        else:
            self.send(f"Unknown command: {cmd}\nType /help for available commands.")

    def _send_help(self):
        """Send help message with all available commands."""
        commands = [
            "/help      — This help message",
            "/start     — Resume trading",
            "/stop      — Stop bot gracefully",
            "/pause N   — Pause for N minutes (default 30)",
            "/status    — Bot status + current regime",
            "/summary   — Today's trading summary",
            "/trades    — Last 5 trades",
            "/risk      — Risk manager state",
            "/balance   — Account balance",
            "/regime    — Current market regime",
            "/positions — Open positions",
        ]
        msg = "<b>MIDAS v2 Commands:</b>\n\n" + "\n".join(commands)
        self.send(msg)

    # ─── Send Methods ─────────────────────────────────────

    def send(self, message: str) -> bool:
        """Send a message to Telegram."""
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"  Telegram send error: {e}")
            return False

    # ─── Trade Alerts ─────────────────────────────────────

    def trade_opened(self, trade: dict):
        """Detailed trade opened alert."""
        d = trade.get("direction", "").upper()
        emoji = "🟢" if d == "BUY" else "🔴"
        source = trade.get("source", "N/A")
        regime = trade.get("regime", "N/A")

        entry = trade.get("entry_price", 0)
        sl = trade.get("sl_price", 0)
        tp = trade.get("tp_price", 0)
        lot = trade.get("lot_size", 0)
        risk = trade.get("risk_dollars", 0)
        conf = trade.get("confidence", 0)
        e_prob = trade.get("e_prob")

        # Calculate RR
        sl_dist = abs(entry - sl)
        tp_dist = abs(entry - tp)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        msg = (
            f"{emoji} <b>TRADE OPENED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Direction  : {d}\n"
            f"Entry      : {entry:.2f}\n"
            f"SL         : {sl:.2f} ({sl_dist/PIP_VALUE:.1f} pips)\n"
            f"TP         : {tp:.2f} ({tp_dist/PIP_VALUE:.1f} pips)\n"
            f"RR         : 1:{rr:.1f}\n"
            f"Lot        : {lot}\n"
            f"Risk       : ${risk:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Source     : {source}\n"
            f"Confidence : {conf:.0f}%\n"
            f"Regime     : {regime}\n"
        )
        if e_prob is not None:
            e_dir = "BUY" if e_prob > 0.5 else "SELL"
            msg += f"E predict  : {e_dir} ({e_prob:.3f})\n"

        msg += f"Time       : {datetime.now().strftime('%H:%M:%S')}"
        self.send(msg)

        # Reset position update timer
        self._last_position_update = datetime.now()

    def trade_closed(self, pnl: float, reason: str, balance: float,
                     direction: str = "", entry: float = 0, exit_price: float = 0,
                     duration_mins: int = 0):
        """Detailed trade closed alert."""
        emoji = "💰" if pnl > 0 else "💔"
        result = "WIN" if pnl > 0 else "LOSS"

        msg = (
            f"{emoji} <b>TRADE CLOSED — {result}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Direction : {direction.upper()}\n"
            f"Entry     : {entry:.2f}\n"
            f"Exit      : {exit_price:.2f}\n"
            f"PnL       : <b>${pnl:+.2f}</b>\n"
            f"Reason    : {reason}\n"
            f"Duration  : {duration_mins}m\n"
            f"Balance   : ${balance:,.2f}\n"
            f"Time      : {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send(msg)

    def tsl_updated(self, old_sl: float, new_sl: float, reason: str,
                    current_price: float, progress_pct: float):
        """TSL change notification."""
        direction_emoji = "⬆️" if new_sl > old_sl else "⬇️"
        msg = (
            f"📐 <b>TSL UPDATED</b> {direction_emoji}\n"
            f"Old SL    : {old_sl:.2f}\n"
            f"New SL    : {new_sl:.2f}\n"
            f"Price     : {current_price:.2f}\n"
            f"Progress  : {progress_pct:.0%} toward TP\n"
            f"Reason    : {reason}"
        )
        self.send(msg)

    def position_update(self, trade: dict, current_price: float,
                        unrealized_pnl: float, progress_pct: float,
                        tsl_status: dict):
        """
        Periodic position update (every 3 minutes).
        Shows current state of open trade.
        """
        now = datetime.now()
        if (self._last_position_update and
                (now - self._last_position_update).total_seconds() < self.POSITION_UPDATE_INTERVAL):
            return  # Not time yet

        self._last_position_update = now

        d = trade.get("direction", "").upper()
        entry = trade.get("entry_price", 0)
        sl = trade.get("current_sl", trade.get("sl_price", 0))
        tp = trade.get("current_tp", trade.get("tp_price", 0))

        pnl_emoji = "📈" if unrealized_pnl >= 0 else "📉"

        msg = (
            f"{pnl_emoji} <b>POSITION UPDATE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Direction : {d}\n"
            f"Entry     : {entry:.2f}\n"
            f"Current   : {current_price:.2f}\n"
            f"SL        : {sl:.2f}\n"
            f"TP        : {tp:.2f}\n"
            f"Progress  : {progress_pct:.0%}\n"
            f"PnL       : ${unrealized_pnl:+.2f}\n"
        )

        if tsl_status.get("tracking"):
            msg += (
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"TSL Level : {tsl_status.get('tsl_level', 0)}/4\n"
                f"BE done   : {'Yes' if tsl_status.get('breakeven') else 'No'}\n"
                f"Adjusts   : {tsl_status.get('adjustments', 0)}\n"
            )

        msg += f"Time      : {now.strftime('%H:%M:%S')}"
        self.send(msg)

    # ─── System Alerts ────────────────────────────────────

    def daily_summary(self, stats: dict):
        """End-of-day summary."""
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0

        msg = (
            f"📊 <b>DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Date      : {stats.get('date', '')}\n"
            f"Trades    : {total} ({wins}W / {losses}L)\n"
            f"Win Rate  : {wr:.0f}%\n"
            f"PnL       : <b>${stats.get('pnl', 0):+.2f}</b>\n"
            f"Balance   : ${stats.get('balance', 0):,.2f}\n"
            f"Max DD    : {stats.get('max_dd', 0):.1f}%\n"
            f"Sessions  : A={stats.get('asian', 0)} L={stats.get('london', 0)} "
            f"NY={stats.get('ny', 0)}"
        )
        self.send(msg)

    def bot_started(self, balance: float, mode: str = "DEMO"):
        """Bot startup alert."""
        msg = (
            f"🚀 <b>MIDAS v2 STARTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Mode       : {mode}\n"
            f"Balance    : ${balance:,.2f}\n"
            f"Strategies : A+B+C+D+E\n"
            f"Regime     : ON\n"
            f"Dynamic TSL: ON\n"
            f"Lot Range  : 0.01-0.05\n"
            f"Risk Cap   : $40/trade, $100/day\n"
            f"Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send(msg)

    def bot_stopped(self, reason: str = "Manual"):
        self.send(f"🛑 <b>MIDAS v2 STOPPED</b>\nReason: {reason}\n"
                  f"Time: {datetime.now().strftime('%H:%M:%S')}")

    def bot_paused(self, minutes: int):
        self.send(f"⏸️ <b>BOT PAUSED</b> for {minutes} minutes\n"
                  f"Resume at: {(datetime.now() + timedelta(minutes=minutes)).strftime('%H:%M:%S')}")

    def bot_resumed(self):
        self.send(f"▶️ <b>BOT RESUMED</b>\nTime: {datetime.now().strftime('%H:%M:%S')}")

    def error(self, message: str):
        self.send(f"⚠️ <b>ERROR</b>\n{message}\n"
                  f"Time: {datetime.now().strftime('%H:%M:%S')}")

    def signal_skipped(self, reason: str, source: str = ""):
        self.send(f"⏭️ Signal skipped\nSource: {source}\nReason: {reason}")


# Import here to avoid circular
from config import PIP_VALUE