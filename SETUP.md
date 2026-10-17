# MIDAS v2 — Cold Start / Re-Run Runbook

For picking the project back up after months away: reissuing every credential,
rebuilding everything the repo does not carry, and getting back to live trading
without risking the account on a stale config.

Work top to bottom. Each step has a **verify** command — if it fails, fix it
before moving on. A quiet failure here shows up later as a bad trade.

**Time:** ~60–90 minutes, most of it the first data download and model training (done automatically on first start).

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

## Step 2 — Check where data, models and logs go

`PROJECT_ROOT` in `src/config.py` is the repo root, worked out from where
`config.py` lives. Every data, model and log path derives from it, so a fresh
clone works wherever you put it.

If your existing `data/` and `models/` live somewhere else (for example an old
`E:\GenAI\Projects\CX_xauusd_ai_bot_v2`), point the bot there in `.env`:

```ini
MIDAS_PROJECT_ROOT=E:\GenAI\Projects\CX_xauusd_ai_bot_v2
```

**Verify:**

```bat
python src\config.py
```

The printed `Data dir` must be the folder you expect. It also creates
`data/`, `models/`, `logs/`, `backtest_results/` there.

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
8. **Nothing to edit for account size.** Risk limits are percentages that the
   bot sizes from this account at startup and every broker midnight (on $500:
   up to $15 per trade, $20/day, below the hard 5% = $25). The fixed max-loss floor uses
   the account's deposit history; set `ACCOUNT_BALANCE` in `src/config.py` only
   if that history is wrong. Check the startup lines `Initial balance ...` and
   `Today's base ...`.

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

## Step 5 — Market data and Strategy E (automatic)

**`python src\main.py` does this for you.** On every start it checks Strategy
E's model and, if the model's training data is more than
`E_MODEL_MAX_AGE_DAYS` (7) days old, was built with an older feature pipeline,
or doesn't exist yet:

1. appends the M5/M15 bars missing since the last download to `data/raw/`
   (a full download from `DATA_START_YEAR` the first time),
2. retrains on the last `E_TRAIN_YEARS` (4) years, ending at the newest bar,
3. checks the new model on the latest `E_TEST_MONTHS` (6) months it has never
   seen, and only swaps it in if AUC ≥ `E_MIN_TEST_AUC` (0.52),
4. archives the previous model to `models/archive/<time>/`.

You get a Telegram message when it starts and when it finishes (or is
rejected/fails — then the old model stays). Left running, the bot repeats the
check on Saturdays while the market is closed. Settings are in `config.py`
under "Strategy E auto-retrain".

The first run takes a while (full download + feature build + three training
runs). To do it ahead of time instead of at bot start:

```bat
python src\retrain_e.py --check     :: is the model fresh?
python src\retrain_e.py             :: sync + retrain if stale (MT5 running)
python src\retrain_e.py --force     :: retrain now
```

`python src\main.py --skip-retrain` starts with whatever model is saved;
`--retrain` forces a retrain first.

**History depth.** MT5 only serves as many bars as the terminal keeps:
**Tools → Options → Charts → Max bars in chart** (set it to Unlimited). If the
first download stops short of `DATA_START_YEAR`, raise that setting or the year.

**Verify:**

```bat
python src\data_downloader.py --verify
python src\retrain_e.py --check
```

Expect data ending roughly today and `"state": "fresh"`.

### Research path (optional)

Backtests for Strategy E and for all strategies combined:

```bat
python src\strategy_e_data_prep.py      :: E dataset from data/raw (same features as live)
python src\strategy_e_clean_retrain.py  :: walk-forward models + E-only threshold backtest
python src\run_combined_abcde.py        :: A-D baseline vs +E validator / +E independent / full A-E
```

They use the same feature pipeline (`src/strategy_e_features.py`) as the live
bot, and write their models to `models/research/`, so they never replace the
live model (`models/strategy_e_clean_final.txt`, from `retrain_e.py`), and the
bot's retraining never leaks into a backtest. Every backtest segment is
predicted by a model trained only on earlier data.

---

## Step 6 — Dry run

Never go straight to orders after a config change.

```bat
python src\main.py --dry-run
```

Confirm, in order:

- [ ] Banner prints
- [ ] `MT5 connected` with the **new** account number and balance
- [ ] `Strategy E model is up to date (...)` — or a retrain that ends in `✅ Strategy E retrained`
- [ ] `Strategy E loaded: strategy_e_clean_final.txt (N features, pipeline v4, data through <recent date>)`
- [ ] `Broker server time: UTC+3` in summer / `UTC+2` in winter, with no warning
      after it (a warning means the broker's clock rule differs — README §8)
- [ ] `Telegram command listener started` (not `Telegram: disabled`)
- [ ] A "bot started" message arrives on your phone
- [ ] A new `[HH:MM] price | session | DayPnL` line every ~5 minutes
- [ ] `/status` in Telegram replies

Let it run **at least 30 minutes** through an active session (London or NY —
during the Asian session it may legitimately do nothing). Any signal prints
`DRY RUN — not executed`. No orders are placed.

Kill it with Ctrl+C or `/stop`.

---

## Step 7 — Demo, then live

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
- [ ] Startup's `Initial balance ...` and `Today's base ...` lines show that
      account's numbers (and the per-trade / daily limits you expect)
- [ ] Telegram works — you need `/stop` reachable from your phone
- [ ] You have read README §7

---

## Quick reference

```bat
:: every session (also refreshes Strategy E when it is >7 days stale)
.venv\Scripts\activate
python src\main.py --dry-run

:: refresh data + model by hand
python src\retrain_e.py --force

:: health checks
python src\config.py
python src\data_downloader.py --verify
python src\retrain_e.py --check
```

---

## If something breaks

Full symptom table in **README §8**. The ones that bite on a cold start:

| Symptom | Cause | Fix |
|---|---|---|
| `Strategy E model not found` | data/models elsewhere, or the first retrain failed | Step 2, then `python src\\retrain_e.py --force` |
| `Strategy E retrain rejected` | new model failed the out-of-sample check | old model kept; see the printed AUC, retry after more data |
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
