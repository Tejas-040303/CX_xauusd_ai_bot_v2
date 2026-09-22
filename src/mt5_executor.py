"""
Project MIDAS v2 — MT5 Executor
Handles all MetaTrader 5 trade operations.

Operations:
  - Open market orders (BUY/SELL) with SL/TP
  - Close positions
  - Modify SL/TP
  - Fetch account info
  - Fetch live bars
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from config import MT5_SYMBOL, PIP_VALUE, SPREAD_SIMULATION_PIPS


class MT5Executor:
    """MT5 trade execution engine."""

    def __init__(self, symbol: str = None):
        self.symbol = symbol or MT5_SYMBOL
        self.connected = False
        self._symbol_info = None

    # ─── Connection ───────────────────────────────────────

    def connect(self) -> bool:
        """Initialize MT5 connection."""
        if mt5 is None:
            print("  ERROR: MetaTrader5 not installed. pip install MetaTrader5")
            return False

        if not mt5.initialize():
            print(f"  MT5 initialization failed: {mt5.last_error()}")
            return False

        info = mt5.terminal_info()
        account = mt5.account_info()
        self._symbol_info = mt5.symbol_info(self.symbol)

        if self._symbol_info is None:
            print(f"  Symbol {self.symbol} not found")
            mt5.shutdown()
            return False

        if not self._symbol_info.visible:
            mt5.symbol_select(self.symbol, True)

        self.connected = True
        print(f"  MT5 connected: {info.name} | Build: {info.build}")
        print(f"  Account: {account.login} | Balance: ${account.balance:,.2f} | {account.currency}")
        print(f"  Symbol: {self.symbol} | Spread: {self._symbol_info.spread} points")
        return True

    def disconnect(self):
        """Shutdown MT5 connection."""
        if mt5 is not None and self.connected:
            mt5.shutdown()
            self.connected = False
            print("  MT5 disconnected")

    def reconnect(self) -> bool:
        """Reconnect to MT5."""
        self.disconnect()
        time.sleep(2)
        return self.connect()

    # ─── Account Info ─────────────────────────────────────

    def get_account_info(self) -> Dict:
        """Get current account details."""
        acc = mt5.account_info()
        if acc is None:
            return {}
        return {
            "login": acc.login,
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free,
            "profit": acc.profit,
            "currency": acc.currency,
            "leverage": acc.leverage,
        }

    def get_balance(self) -> float:
        """Get current account balance."""
        acc = mt5.account_info()
        return acc.balance if acc else 0.0

    def get_current_price(self) -> Dict:
        """Get current bid/ask for the symbol."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return {"bid": 0, "ask": 0, "spread": 0}
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round((tick.ask - tick.bid) / PIP_VALUE, 1),
            "time": datetime.fromtimestamp(tick.time),
        }

    # ─── Data Fetching ────────────────────────────────────

    def fetch_bars(self, timeframe_name: str, count: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch recent bars from MT5.

        Args:
            timeframe_name: "M1", "M3", "M5", "M15"
            count: Number of bars to fetch

        Returns:
            DataFrame with [datetime, open, high, low, close, volume]
        """
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M3": mt5.TIMEFRAME_M3,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
        }

        tf = tf_map.get(timeframe_name)
        if tf is None:
            print(f"  Invalid timeframe: {timeframe_name}")
            return None

        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            print(f"  No data for {timeframe_name}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={
            "time": "datetime",
            "tick_volume": "volume",
        })
        cols = ["datetime", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]

        return df

    def fetch_latest_bar(self, timeframe_name: str) -> Optional[pd.Series]:
        """Fetch the most recent completed bar."""
        df = self.fetch_bars(timeframe_name, count=2)
        if df is None or len(df) < 2:
            return None
        # Return second-to-last (last completed bar)
        return df.iloc[-2]

    # ─── Trade Execution ──────────────────────────────────

    def open_trade(
        self,
        direction: str,
        lot_size: float,
        sl_price: float,
        tp_price: float,
        comment: str = "MIDAS_v2",
        magic: int = 202504,
    ) -> Optional[Dict]:
        """
        Open a market order.

        Args:
            direction: "buy" or "sell"
            lot_size: Position size (0.01 - 0.05)
            sl_price: Stop loss price
            tp_price: Take profit price
            comment: Trade comment
            magic: Magic number for identification

        Returns:
            Dict with order details or None on failure
        """
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            print("  ERROR: Cannot get current price")
            return None

        if direction == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif direction == "sell":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            print(f"  ERROR: Invalid direction: {direction}")
            return None

        # Validate SL/TP
        if direction == "buy":
            if sl_price >= price:
                print(f"  ERROR: BUY SL ({sl_price}) must be below entry ({price})")
                return None
            if tp_price <= price:
                print(f"  ERROR: BUY TP ({tp_price}) must be above entry ({price})")
                return None
        else:
            if sl_price <= price:
                print(f"  ERROR: SELL SL ({sl_price}) must be above entry ({price})")
                return None
            if tp_price >= price:
                print(f"  ERROR: SELL TP ({tp_price}) must be below entry ({price})")
                return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": round(sl_price, 2),
            "tp": round(tp_price, 2),
            "deviation": 20,  # Max slippage in points
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            print(f"  ERROR: order_send returned None: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"  ORDER FAILED: {result.retcode} — {result.comment}")
            return None

        trade_info = {
            "ticket": result.order,
            "direction": direction,
            "entry_price": result.price,
            "lot_size": lot_size,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "time": datetime.now(),
            "comment": comment,
        }

        print(f"  ✅ ORDER OPENED: {direction.upper()} {lot_size} @ {result.price:.2f} "
              f"SL={sl_price:.2f} TP={tp_price:.2f}")
        return trade_info

    def close_position(self, ticket: int = None, comment: str = "MIDAS_close") -> bool:
        """
        Close an open position.
        If ticket is None, closes the first open position for the symbol.
        """
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None or len(positions) == 0:
            return False

        if ticket:
            pos = next((p for p in positions if p.ticket == ticket), None)
        else:
            pos = positions[0]

        if pos is None:
            return False

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False

        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": 202504,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ POSITION CLOSED: ticket={pos.ticket} @ {price:.2f}")
            return True

        print(f"  ERROR closing position: {result.retcode if result else 'None'}")
        return False

    def modify_sl_tp(self, ticket: int, new_sl: float = None, new_tp: float = None) -> bool:
        """Modify SL and/or TP of an open position."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": round(new_sl, 2) if new_sl else pos.sl,
            "tp": round(new_tp, 2) if new_tp else pos.tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        return False

    def get_open_positions(self) -> list:
        """Get all open positions for the symbol."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []

        return [{
            "ticket": p.ticket,
            "direction": "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
            "volume": p.volume,
            "entry_price": p.price_open,
            "current_price": p.price_current,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit,
            "time": datetime.fromtimestamp(p.time),
            "comment": p.comment,
            "magic": p.magic,
        } for p in positions]

    def has_open_position(self) -> bool:
        """Check if there's an active position."""
        positions = mt5.positions_get(symbol=self.symbol)
        return positions is not None and len(positions) > 0