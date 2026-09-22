"""
Project MIDAS v2 — Central Configuration
All settings in one place. Loaded from .env where needed.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load .env ───────────────────────────────────────────
load_dotenv()

# ─── Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(r"E:\GenAI\Projects\CX_xauusd_ai_bot_v2")
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
BACKTEST_DIR = PROJECT_ROOT / "backtest_results"

# Create dirs if they don't exist
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, LOGS_DIR, BACKTEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── MT5 Settings ────────────────────────────────────────
MT5_SYMBOL = "XAUUSD"
MT5_TIMEFRAMES = {
    "M1": 1,      # TIMEFRAME_M1
    "M3": 3,      # TIMEFRAME_M3
    "M5": 5,      # TIMEFRAME_M5
    "M15": 15,    # TIMEFRAME_M15
}

# Data download range (5 years)
DATA_START_YEAR = 2021
DATA_END_YEAR = 2026  # Up to current

# ─── The5ers Rules ───────────────────────────────────────
ACCOUNT_BALANCE = 5000
PROFIT_TARGET = 5500          # +10%
MAX_LOSS_LEVEL = 4500         # -10% total drawdown
DAILY_LOSS_LIMIT_5ERS = 250   # -5% per day ($250)
MIN_PROFITABLE_DAYS = 3
MAX_INACTIVE_DAYS = 30

# ─── Risk Settings (Our Safety Buffer) ──────────────────
MAX_DAILY_LOSS = 100          # Below The5ers $250 limit
MAX_TRADE_RISK = 55           # Per trade max loss in $
LOT_SIZE_MIN = 0.01
LOT_SIZE_MAX = 0.05
LOT_SIZE_SINGLE = 0.01       # 1 strategy triggers
LOT_SIZE_MULTI = 0.02        # 2+ strategies agree
COOLDOWN_SECONDS = 600        # 10 min between trades
SPREAD_SIMULATION_PIPS = 2.5  # Simulated slippage for backtests

# ─── Session Definitions (UTC) ───────────────────────────
# Broker is UTC+2 (MetaQuotes), adjust if needed
SESSIONS = {
    "asian":  {"start": 0, "end": 6},     # 00:00 - 06:00 UTC
    "london": {"start": 6, "end": 14},    # 06:00 - 14:00 UTC
    "new_york": {"start": 12, "end": 20}, # 12:00 - 20:00 UTC
}

# Session trade limits
SESSION_LIMITS = {
    "asian": 2,
    "london": 2,
    "new_york": 2,
    "off": 0,  # No trades outside sessions
}

# ─── Strategy Confidence Threshold ───────────────────────
CONFIDENCE_THRESHOLD = 65  # Minimum confidence to consider a strategy signal
# Note: This will be tuned in Phase 5

# ─── Shared Feature Parameters ───────────────────────────
# EMAs
EMA_PERIODS = [9, 21, 50, 200]

# ATR
ATR_PERIOD = 14

# RSI
RSI_PERIOD = 14

# ADX (for trend strength)
ADX_PERIOD = 14

# Swing point lookback
SWING_LOOKBACK = 5  # Candles on each side to confirm swing high/low

# ─── Backtester Settings ─────────────────────────────────
# Train/test split
BACKTEST_TRAIN_END = "2024-06-30"     # Train: 2021-01-01 to 2024-06-30
BACKTEST_FORWARD_START = "2024-07-01"  # Forward: 2024-07-01 to 2026-present

# Starting capital for backtest
BACKTEST_INITIAL_CAPITAL = 5000

# Point value for XAUUSD (1 lot = 100 oz, $1 move = $100 per lot)
XAUUSD_POINT_VALUE = 100  # $ per lot per $1 price move
# So 0.01 lot, 10 pip ($1) move = 0.01 * 100 * 1 = $1.00
# And 0.01 lot, 20 pip ($2) move = 0.01 * 100 * 2 = $2.00

# Pip definition for gold (1 pip = $0.10 price movement)
PIP_VALUE = 0.10  # $0.10 in price = 1 pip for XAUUSD

# ─── Telegram Settings ───────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── Claude API Settings ─────────────────────────────────
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ─── Logging ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
TRADE_LOG_FILE = LOGS_DIR / "trades.json"
DAILY_STATE_FILE = LOGS_DIR / "daily_state.json"


def get_mt5_timeframe(tf_name: str):
    """Convert timeframe name to MT5 constant."""
    import MetaTrader5 as mt5
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M3": mt5.TIMEFRAME_M3,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return mapping.get(tf_name)


def print_config():
    """Print current configuration for verification."""
    print("=" * 60)
    print("PROJECT MIDAS v2 — Configuration")
    print("=" * 60)
    print(f"Symbol: {MT5_SYMBOL}")
    print(f"Timeframes: {list(MT5_TIMEFRAMES.keys())}")
    print(f"Data range: {DATA_START_YEAR} - {DATA_END_YEAR}")
    print(f"Account balance: ${ACCOUNT_BALANCE}")
    print(f"Max daily loss: ${MAX_DAILY_LOSS}")
    print(f"Max trade risk: ${MAX_TRADE_RISK}")
    print(f"Lot range: {LOT_SIZE_MIN} - {LOT_SIZE_MAX}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"Session limits: {SESSION_LIMITS}")
    print(f"Backtest train end: {BACKTEST_TRAIN_END}")
    print(f"Backtest forward start: {BACKTEST_FORWARD_START}")
    print(f"Data dir: {DATA_RAW}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()