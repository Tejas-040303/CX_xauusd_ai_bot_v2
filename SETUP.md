# MIDAS v2 — Cold Start / Re-Run Runbook

For picking the project back up after months away: reissuing every credential,
rebuilding everything the repo does not carry, and getting back to live trading
without risking the account on a stale config.

Work top to bottom. Each step has a **verify** command — if it fails, fix it
before moving on. A quiet failure here shows up later as a bad trade.

**Time:** ~60–90 minutes, most of it the data download and the model retrain.

---

## What you actually need to renew

Short version, because it is less than you'd think:

| Credential | Needed? | Where it lives | Why |
|---|---|---|---|
| **Telegram bot token** | **Yes** | `.env` → `TELEGRAM_TOKEN` | Alerts + `/stop` kill switch |
| **Telegram chat id** | **Yes** | `.env` → `TELEGRAM_CHAT_ID` | Bot only answers this chat |
| **MT5 trading account** | **Yes** | **Not in `.env`** — the MT5 terminal itself | `mt5.initialize()` attaches to the logged-in terminal |
| **Claude API key** | **No** | `.env` → `CLAUDE_API_KEY` | Read by config, used by nothing. `claude_brain.py` is empty. |

The one that surprises people: **there are no broker credentials in this
project.** `src/mt5_executor.py:44` calls `mt5.initialize()` with no arguments,
which attaches to whatever MT5 terminal is already running and logged in. You
change accounts by logging into the terminal, not by editing a file. That also
means the bot will happily trade whatever account the terminal happens to be on
— **check the terminal before you start it.**

---

## Step 0 — Prerequisites

- **Windows 64-bit.** Non-negotiable: `MetaTrader5` publishes `win_amd64` wheels
  only. No Linux, no macOS, no WSL.
- **Python 3.12, 64-bit.** Not 3.11 — one file is a `SyntaxError` below 3.12
  (README gotcha 2). Not 3.14 — lightgbm/pyarrow wheel coverage is thinner.
- **MetaTrader 5 terminal** installed, with your broker's server reachable.

```bat
python --version
python -c "import struct; print(struct.calcsize('P')*8, 'bit')"
```

Expect `Python 3.12.x` and `64 bit`. A 32-bit interpreter cannot import
`MetaTrader5` at all.

---

## Step 1 — Get the code and install dependencies

```bat
git clone <your-repo-url> CX_xauusd_ai_bot_v2
cd CX_xauusd_ai_bot_v2

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Verify:**

```bat
python -c "import MetaTrader5, pandas, numpy, lightgbm, sklearn, matplotlib, requests, dotenv, pyarrow; print('all imports OK')"
```

If `MetaTrader5` fails here, you are on 32-bit Python or a non-Windows box.

---

## Step 2 — Fix `PROJECT_ROOT` (do this before anything else)

`src/config.py` line 14 still points at the machine this was built on:

```python
PROJECT_ROOT = Path(r"E:\GenAI\Projects\CX_xauusd_ai_bot_v2")
```

Every data, model and log path derives from it. Leave it wrong and the bot
creates empty folders on a drive you don't care about, then reports "Strategy E
model not found" forever.

**Pick one.** Either hardcode your real path:

```python
PROJECT_ROOT = Path(r"C:\Users\<you>\projects\CX_xauusd_ai_bot_v2")
```

Or make it portable, so this never bites again (`config.py` lives in `src/`, so
one `.parent` reaches the repo root):

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

**Verify:**

```bat
python src\config.py
```

The printed `Data dir` must be inside your actual project folder. It also creates
`data/`, `models/`, `logs/`, `backtest_results/` — confirm they appeared where you
expect, not on `E:\`.

---

## Step 3 — Renew the trading account (MT5 terminal)

No files involved. This is done entirely in the terminal.

1. Open **MetaTrader 5**.
2. **File → Login to Trade Account.**
3. Enter the new login, password and **exact** server name from your broker or
   The5ers dashboard. Server names are precise (`The5ers-Server`,
   `MetaQuotes-Demo`) — a near-miss silently fails to connect.
4. Confirm the account number in the terminal title bar is the one you intend to
   trade. **The bot trades whatever is logged in here.**
5. **Tools → Options → Expert Advisors → tick "Allow algorithmic trading".**
   Without it every `order_send` returns `10027` and no trade ever opens.
6. Find XAUUSD in **Market Watch** (Ctrl+M). Right-click → Show All if it is
   hidden. **Note the exact symbol name** — many brokers use a suffix like
   `XAUUSD.r`, `XAUUSDm`, `XAUUSD_i`. If yours is not plain `XAUUSD`, update
   `MT5_SYMBOL` in `src/config.py`.
7. Leave the terminal **running and logged in**. The bot cannot start it.

**Verify:**

```bat
python -c "import MetaTrader5 as mt5; mt5.initialize(); a=mt5.account_info(); print(a.login, a.server, a.balance, a.currency); print('XAUUSD:', mt5.symbol_info('XAUUSD') is not None); mt5.shutdown()"
```

You want the new account number, the right server, a sane balance, and
`XAUUSD: True`. If that last one is `False`, go back to 6.

---

## Step 4 — Renew the Telegram bot

You can reuse the old bot or create a fresh one. **Reissue the token either
way** — if the old one has been sitting in a `.env` on an old machine, in a chat
log, or anywhere else, treat it as compromised.

### 4a. Token

Reissuing an existing bot (keeps the same `@username` and chat history):

1. Message **@BotFather** → `/mybots` → pick your bot
2. **API Token** → **Revoke current token**
3. Copy the new token

Creating a new bot:

1. **@BotFather** → `/newbot`
2. Give it a display name, then a username ending in `bot`
3. Copy the token from the confirmation message

Format: `8123456789:AAF...` — digits, a colon, then ~35 characters.

> Revoking instantly kills the old token. Anything still running on it goes
> silent, which is the point.

### 4b. Chat id

1. Open a chat with your bot and send **`/start`** — a bot cannot message you
   until you message it first. Skipping this is the most common reason alerts
   never arrive.
2. Message **@userinfobot** → it replies with your numeric id.

Personal chat ids are positive (`123456789`). Group ids are negative
(`-1001234567890`) — include the minus sign.

### 4c. Write `.env`

Put it in the **project root**, beside `src/` and `.env.example`.
`config.py` calls bare `load_dotenv()`, which walks up from `src/` and finds a
root-level `.env` fine. A `.env` inside `src/` would also be picked up — the risk
is ending up with *two* of them and editing the one that loses.

```bat
copy .env.example .env
notepad .env
```

```ini
TELEGRAM_TOKEN=8123456789:AAF-your-new-token-here
TELEGRAM_CHAT_ID=123456789
CLAUDE_API_KEY=
```

No quotes, no spaces around `=`, no trailing whitespace. `CLAUDE_API_KEY` stays
empty — nothing reads it.

**Verify** — this sends a real message:

```bat
python -c "import sys; sys.path.insert(0,'src'); from telegram_notifier import TelegramNotifier; n=TelegramNotifier(); print('enabled:', n.enabled); n.send('MIDAS v2 setup test')"
```

`enabled: True` **and** the message arrives on your phone. If `enabled` is
`True` but nothing arrives, the token is fine and the **chat id is wrong**.

---

## Step 5 — Re-download market data

The repo ships no data — `data/` is gitignored. Your old CSVs are months stale
regardless, and Strategy E should be retrained on data that includes recent
regimes.

MT5 must be running and logged in (Step 3).

```bat
python src\data_downloader.py
```

Downloads M1, M3, M5, M15 from `DATA_START_YEAR` (2021) to today, in 6-month
chunks, into `data/raw/`. M1 is the slow one. Gaps reported for weekends and
holidays are normal.

**Only M5 and M15 are needed** for training and live trading. To save time:

```bat
python src\data_downloader.py --tf M5
python src\data_downloader.py --tf M15
```

**Verify:**

```bat
python src\data_downloader.py --verify
```

Check the end date reaches roughly today and the candle counts are in the
millions for M1 / hundreds of thousands for M5. If the range stops short, your
broker limits history depth — raise `DATA_START_YEAR` in `config.py` to whatever
they actually serve and re-run.

---

## Step 6 — Rebuild Strategy E

`models/` is gitignored too, so the LightGBM model must be retrained. Without it
the bot still runs A–D but prints `Strategy E unavailable`.

```bat
python src\strategy_e_data_prep.py
```

Computes every feature stage on M5, merges M15 context, builds ATR-filtered
labels, and writes `data/processed/strategy_e_train_ready.parquet`. Slow — it
computes features over years of M5 bars. The intermediate
`XAUUSD_M5_all_features.parquet` is cached, so a re-run is much faster.

```bat
python src\strategy_e_clean_retrain.py
```

This is the one that matters. It debiases the swing-derived features (shifting
them by `SWING_LOOKBACK` to remove lookahead), retrains walk-forward, and
backtests on clean data. It writes:

- `models/strategy_e_clean_final.txt` ← what the live bot loads
- `models/strategy_e_clean_metadata.json` ← the feature list
- `backtest_results/strategy_e_clean_backtest.json`

> Use `strategy_e_clean_retrain.py`, not `strategy_e_train.py`.
> `LiveStrategyE` prefers the `_clean_` model and falls back to the
> non-debiased `strategy_e_final.txt`, whose backtest numbers are inflated by
> lookahead bias. Produce the clean one.

**Verify:**

```bat
python -c "import sys; sys.path.insert(0,'src'); from strategies.live_strategy_e import LiveStrategyE; e=LiveStrategyE(); print('loaded:', e.loaded, '| features:', len(e.feature_cols or []))"
```

Expect `loaded: True` and a feature count in the low hundreds.

---

## Step 7 — Dry run

Never go straight to orders after a config change.

```bat
python src\main.py --dry-run
```

Confirm, in order:

- [ ] Banner prints
- [ ] `MT5 connected` with the **new** account number and balance
- [ ] `Strategy E loaded: strategy_e_clean_final.txt (N features)`
- [ ] `Telegram command listener started` (not `Telegram: disabled`)
- [ ] A "bot started" message arrives on your phone
- [ ] A new `[HH:MM] price | session | DayPnL` line every ~5 minutes
- [ ] `/status` in Telegram replies

Let it run **at least 30 minutes** through an active session (London or NY —
during the Asian session it may legitimately do nothing). Any signal prints
`DRY RUN — not executed`. No orders are placed.

Kill it with Ctrl+C or `/stop`.

---

## Step 8 — Demo, then live

**Demo first**, for at least a few sessions:

```bat
python src\main.py
```

`--mode demo` is the default. Real orders, fake money — this is where you find
out whether SL/TP distances clear your broker's stop level and whether fills
behave.

Only once demo looks right:

```bat
python src\main.py --mode live
```

Prompts for a typed `CONFIRM`. Before you type it:

- [ ] The MT5 terminal is logged into the **intended** account
- [ ] `ACCOUNT_BALANCE` and the limits in `config.py` match that account
- [ ] Telegram works — you need `/stop` reachable from your phone
- [ ] You have read README §7; **Strategy E's live predictions are degraded
      until gotcha 3 is fixed**

---

## Quick reference

```bat
:: every session
.venv\Scripts\activate
python src\main.py --dry-run

:: refresh data + model (monthly-ish)
python src\data_downloader.py --tf M5
python src\data_downloader.py --tf M15
python src\strategy_e_data_prep.py
python src\strategy_e_clean_retrain.py

:: health checks
python src\config.py
python src\data_downloader.py --verify
```

---

## If something breaks

Full symptom table in **README §8**. The ones that bite on a cold start:

| Symptom | Cause | Fix |
|---|---|---|
| `Strategy E model not found` | `PROJECT_ROOT` wrong, or Step 6 skipped | Step 2, then Step 6 |
| `Telegram: disabled` | `.env` missing or values blank | Step 4c — `.env` goes in the project root |
| `enabled: True` but no messages | wrong chat id | Step 4b — and send `/start` to your bot |
| `MT5 initialization failed` | terminal closed or logged out | Step 3 |
| `Symbol XAUUSD not found` | broker uses a suffix | set `MT5_SYMBOL` in `config.py` |
| `ORDER FAILED: 10027` | algo trading disabled | Step 3.5 |
| `SyntaxError: f-string` | Python < 3.12 | Step 0 |
| Data stops years short | broker history limit | raise `DATA_START_YEAR` |

---

## Security

- `.env` is gitignored. Keep it that way; commit `.env.example` instead.
- Revoke the old Telegram token via @BotFather rather than leaving it live.
- The bot only accepts commands from `TELEGRAM_CHAT_ID`, so a leaked token means
  an attacker can read your alerts and impersonate the bot, but not trade
  through it. Revoke it anyway.
- Nothing in this project stores broker credentials — they live in the MT5
  terminal. Anyone with access to that machine has access to the account.
