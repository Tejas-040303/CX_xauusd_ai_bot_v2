# Project MIDAS v2 — XAUUSD AI Trading Bot

An automated gold (XAUUSD) trading bot for MetaTrader 5. Five strategies vote on
every M5 candle: four rule-based (`A`–`D`) and one LightGBM model (`E`) that acts
as both a **validator** of the others and an **independent signal generator**. A
regime detector retunes the thresholds per market condition, a dynamic lot sizer
maps conviction to position size, and a risk manager enforces prop-firm-style
rules (modeled on The5ers). Control and monitoring run through a Telegram bot.

> **Coming back after a break?** Go straight to **[SETUP.md](SETUP.md)** — it is
> the step-by-step runbook for reissuing tokens and getting the bot running again
> from a cold start. This README explains *what the system is*; SETUP.md explains
> *how to run it*.

---

## 1. Status at a glance

| | |
|---|---|
| **Platform** | Windows 64-bit **only** (`MetaTrader5` ships `win_amd64` wheels exclusively) |
| **Python** | **3.12** required — see [§7](#7-known-issues--gotchas) |
| **Instrument** | XAUUSD, M5 decisions with M15 structural context |
| **Account model** | Prop-style rules modeled on The5ers (not a The5ers account); limits sized from the live MT5 account (+10% target / −10% max drawdown) |
| **Secrets needed** | `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` — that's it |
| **Claude API** | Declared in config, **never used**. `claude_brain.py` is empty. |
| **MT5 credentials** | Not in `.env`. Inherited from the running MT5 terminal. |
| **Model + data in repo?** | **No.** `data/`, `models/`, `logs/` are gitignored and must be rebuilt. |

---

## 2. How a trade happens

The live loop (`src/main.py`) polls MT5 every 10 seconds and acts once per new
completed M5 bar.

```text
                        ┌────────────────────────┐
                        │  MT5 terminal (logged  │
                        │  in, XAUUSD in Market  │
                        │  Watch)                │
                        └───────────┬────────────┘
                                    │ copy_rates_from_pos
                                    ▼
                    ┌───────────────────────────────┐
                    │ 1. Fetch 500x M5 + 200x M15   │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 2. FeatureEngine              │
                    │    M15: shared + FVG + SMC    │
                    │    M5 : shared + trend        │
                    │          + session_range      │
                    │    merge_asof M15 -> m15_*    │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 3. RegimeDetector             │
                    │    -> TRENDING_BULL / BEAR /  │
                    │       VOLATILE / QUIET /      │
                    │       RANGING + adjustments   │
                    └───────────────┬───────────────┘
                                    ▼
        ┌───────────────────────────┴───────────────────────────┐
        │ 4. Strategies A-D in parallel (ThreadPoolExecutor)    │
        │    A: SMC order blocks   B: fair value gaps           │
        │    C: session breakout   D: trend continuation        │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 5. Strategy E (LightGBM)      │
                    │    debias -> predict -> p(BUY)│
                    └───────────────┬───────────────┘
                                    ▼
              ┌─────────────────────┴─────────────────────┐
              │ 6. Decision                                │
              │    A-D fired?  -> E validates, adjusts     │
              │                   confidence, may veto     │
              │    A-D silent? -> E may fire independently │
              └─────────────────────┬─────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 7. DynamicLotSizer 0.01-0.05  │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 8. RiskManager — FINAL VETO   │
                    │    daily cap, session cap,    │
                    │    cooldown, per-trade risk   │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 9. MT5Executor.open_trade()   │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ 10. TSLManager trails the SL  │
                    │     TradeLogger -> logs/      │
                    │     TelegramNotifier -> phone │
                    └───────────────────────────────┘
```

**One position at a time.** While a trade is open the loop stops looking for new
signals entirely and switches to monitoring/trailing every 5 seconds.

### The two decision paths

**Path 1 — A–D fired.** Conflicting directions abort immediately. Otherwise the
highest-confidence signal wins, gets `+10` confidence per additional agreeing
strategy, then E validates it:

| E's view | Condition | Effect |
|---|---|---|
| Agrees strongly | `e_strength >= 0.60` | `+10` confidence (+ regime boost) |
| Agrees | `e_strength >= 0.52` | no change |
| Agrees weakly | below that | `−10` confidence |
| **Conflicts strongly** | `e_strength >= 0.60` against | **trade vetoed** |
| Conflicts weakly | below that | `−10` confidence |

Survivors need `confidence >= CONFIDENCE_THRESHOLD` (65).

**Path 2 — A–D silent.** E may fire alone if `p >= 0.68` (buy) or `p <= 0.32`
(sell), *and* the derived confidence clears `E_INDEPENDENT_MIN_CONF` (70), *and*
the regime permits it. SL is `1.5 x ATR`; TP is `1.5R`, or `2.5R` when confidence
is `>= 75`.

### Regime adjustments

`RegimeDetector` classifies each bar and rewrites E's thresholds:

| Regime | E indep. threshold | Conf. boost | SL mult | TP mult | E independent? |
|---|---|---|---|---|---|
| `TRENDING_BULL` | 0.65 | +5 | 1.0 | 1.2 | yes |
| `TRENDING_BEAR` | 0.65 | +5 | 1.0 | 1.2 | yes |
| `VOLATILE` | 0.72 | −5 | 1.3 | 1.0 | yes (stricter) |
| `QUIET` | 0.75 | −10 | 0.8 | 0.8 | **no** |
| `RANGING` | 0.70 | 0 | 1.0 | 0.9 | yes |

### Trailing stop progression

`TSLManager` locks in profit as price advances toward TP. Breakeven moves at 25%
progress (entry + 1 pip), then:

| Progress to TP | Profit locked |
|---|---|
| 30% | 10% |
| 50% | 30% |
| 70% | 50% |
| 90% | 75% |

---

## 3. Risk model

Two layers: hard rules modeled on The5ers' prop rules (we don't trade a The5ers
account — the rules are kept as the framework), and a tighter internal buffer
that trips first.
All limits are **percentages**; `RiskManager` turns them into dollars from the
MT5 account itself, so nothing needs editing when the account size changes:

- **Initial balance** = the account's deposits in MT5 history (override with
  `ACCOUNT_BALANCE` in `config.py`) → fixed max-loss floor.
- **Day's base** = higher of balance/equity at **broker midnight** (how The5ers-style rules
  measures daily loss) → daily stop and per-trade risk, recomputed every day
  and saved, so a restart mid-day can't reset it.
- **Day's loss** = base − current equity (floating P/L counts).

| Control | Internal (`config.py`) | Hard limit (The5ers-style) | On a $500 day |
|---|---|---|---|
| Daily loss | `MAX_DAILY_LOSS_PCT` 4% of day's base | 5% of day's base | $20 (hard $25) |
| Per-trade risk | `MAX_TRADE_RISK_PCT` 1.5% of day's base | — | $7.50 |
| Total drawdown | — | 10% below initial balance (fixed) | floor $450 |
| Lot-size reduction | at −20/−50/−80% of the day's stop; no new trades past −80% | — | −$4/−$10/−$16 |
| Lot size | 0.01 – 0.05 (capped by per-trade risk) | — | |
| Trades per session | 2 (`asian` / `london` / `london_ny_overlap` / `new_york`) | — | |
| Cooldown | `COOLDOWN_SECONDS` = 300s between trades | — | |

`config.py` refuses to load unless per-trade % ≤ daily stop % < the hard 5%.
Startup prints the sized limits, warns if `ACCOUNT_BALANCE` is set and doesn't
match the account's deposits, and resets the day's counters if MT5 is logged
into a different account than the saved state belongs to. `/risk` in Telegram
shows the live numbers. Backtests use `BACKTEST_INITIAL_CAPITAL` ($500) for the
same percentages.

On a small account the minimum lot dominates: 0.01 lot loses $1 per $1 gold
move, so a $3–7 stop already risks $3–7. Trades that exceed the per-trade cap
even at 0.01 lot are skipped and logged.

`RiskManager` persists to `logs/daily_state.json`, so limits survive a restart —
a crash-loop cannot reset your daily loss counter.


---

## 4. Repository layout

```text
CX_xauusd_ai_bot_v2/
├── README.md              <- you are here
├── SETUP.md               <- the re-run runbook
├── requirements.txt
├── .env.example           <- copy to .env
└── src/
    ├── config.py              Central config + .env loading (MIDAS_PROJECT_ROOT to relocate).
    ├── main.py                Live bot. Active code starts at line 3571.
    ├── feature_engineering.py FeatureEngine. Active code starts at line 1050.
    │
    ├── LIVE RUNTIME ────────────────────────────────────────────
    ├── mt5_executor.py        Orders, positions, bars, account info
    ├── risk_manager.py        Hard + buffer risk limits, persisted state
    ├── broker_time.py         Broker server time <-> UTC (DST-aware)
    ├── tsl_manager.py         Progressive trailing stop
    ├── dynamic_lot_sizer.py   Conviction -> 0.01-0.05 lots
    ├── regime_detector.py     Online regime classification
    ├── telegram_notifier.py   Two-way Telegram (alerts + commands)
    ├── trade_logger.py        logs/trades.json, daily, cumulative
    ├── strategies/
    │   ├── strategy_a.py      SMC order blocks + liquidity sweeps
    │   ├── strategy_b.py      Fair value gaps
    │   ├── strategy_c.py      Session breakout
    │   ├── strategy_d.py      Trend continuation
    │   ├── strategy_e.py      LightGBM (backtest-side)
    │   └── live_strategy_e.py LightGBM (live inference)
    │
    ├── DATA + TRAINING PIPELINE ────────────────────────────────
    ├── data_downloader.py         MT5 -> data/raw/*.csv (--update appends new bars)
    ├── strategy_e_features.py     E features: ONE pipeline for training AND live
    ├── retrain_e.py               keeps E current: sync -> retrain -> gate -> swap
    ├── strategy_e_data_prep.py    features + labels -> parquet (research)
    ├── strategy_e_train.py        walk-forward LightGBM
    ├── strategy_e_clean_retrain.py fixed-window retrain + backtest (research)
    ├── backtester.py              engine + metrics + equity curves
    │
    ├── RESEARCH RUNNERS (optional, not needed to trade) ─────────
    ├── run_strategy_{a,b,c,d,e}.py
    ├── run_combined_{bd,bcd,abcd,abcde}.py
    ├── run_phase5_tuning.py       threshold sweep
    ├── run_phase6_metalearner.py  stacking experiment
    ├── run_phase8_lots.py         lot-sizing validation
    ├── run_phase9_regime.py       regime validation
    ├── run_strategy_e_debiased.py
    ├── strategy_e_execution_test.py
    ├── diagnose_strategy_b.py
    │
    └── UNUSED / LEGACY ──────────────────────────────────────────
        ├── decision_engine.py   superseded by main.py's own logic
        ├── daily_limits.py      superseded by risk_manager.py
        ├── claude_brain.py      EMPTY (0 bytes)
        ├── meta_learner.py      EMPTY (0 bytes)
        └── rl/{agent,environment,train_rl}.py   EMPTY (0 bytes)
```

Generated at runtime, gitignored, **not in the repo**:

```text
data/raw/          XAUUSD_{M1,M3,M5,M15}.csv     from data_downloader.py
data/processed/    *_features.csv, *.parquet      from the E pipeline
models/            strategy_e_clean_final.txt     LIVE model, from retrain_e
                   strategy_e_clean_metadata.json  feature list, pipeline version, trained_through
                   archive/<time>/                 previous models
                   research/                       walk-forward models from clean_retrain (backtests only)
logs/              trades.json, daily_state.json, cumulative.json
backtest_results/  *.json, *.png
```

---

## 5. Telegram control

`TelegramNotifier` runs a polling thread and only answers your `TELEGRAM_CHAT_ID`.

| Command | Action |
|---|---|
| `/help` | list commands |
| `/status` | state, balance, equity, day P&L, regime, is E loaded |
| `/stop` | graceful stop after the current cycle |
| `/start` | resume |
| `/pause N` | pause N minutes (default 30) |
| `/summary` | today's trades, win rate, P&L, best/worst |
| `/trades` | last 5 trades |
| `/risk` | daily P&L, remaining budget, session counts, lock state |
| `/balance` | balance + equity |
| `/regime` | regime distribution |
| `/positions` | open positions with live P&L |

Pushed automatically: trade opened/closed, every TSL move, a position update
every 3 minutes while in a trade, the daily summary, and errors.

**`/stop` and `/pause` are your remote kill switch.** If Telegram is
misconfigured you lose them — the only way to stop the bot is at the machine.

---

## 6. Commands

```bash
# Live, demo account, signals only — no orders. Start here.
python src/main.py --dry-run

# Live on a demo account (default mode)
python src/main.py

# Real money — prompts for a typed CONFIRM
python src/main.py --mode live

# Data
python src/data_downloader.py            # all timeframes
python src/data_downloader.py --tf M5    # one timeframe
python src/data_downloader.py --verify   # integrity check, no download

# Strategy E (main.py also does this at startup when the model is >7 days stale)
python src/retrain_e.py --check          # model status
python src/retrain_e.py                  # sync new bars + retrain if stale
python src/retrain_e.py --force          # retrain now
python src/main.py --skip-retrain        # start without checking the model

# Sanity check config
python src/config.py
```

---

## 7. Known issues & gotchas

These are real and were verified against the code. Read them before trusting a
run.

**1. ~~`PROJECT_ROOT` is hardcoded to another machine.~~ Fixed.**
It is now the repo root (derived from `config.py`'s location); set
`MIDAS_PROJECT_ROOT` in `.env` to keep data/models elsewhere.

**2. Python 3.12+ is required, not optional.**
`src/run_combined_bcd.py:317-321` uses PEP 701 nested same-quote f-strings
(`f"{f'${mf['total_pnl']:+.0f}':>10}"`). On 3.11 and below that file is a hard
`SyntaxError`. `main.py` separately needs 3.11+ for `datetime.UTC`. Verified:
every file compiles on 3.12 and 3.13; `run_combined_bcd.py` fails on 3.10/3.11.
Only that one research runner is affected — the live path works on 3.11 — but
3.12 avoids the whole question.

**3. ~~Strategy E has a train/serve feature skew.~~ Fixed.**
Live inference skipped M5 FVG/SMC, the ATR-relative features and the
categorical encodings, so 28 of 133 model inputs were NaN. Training and live
now both call `strategy_e_features.build_e_features()`. The same change fixed
two leaks in training: M15 context was joined by bar *open* time (up to 10 min
of future data) and the ATR-relative OB/equal-level distances were computed
before debiasing. Feature pipeline is versioned (`PIPELINE_VERSION`); a model
from an older version is retrained automatically.

**4. Large blocks of dead code.**
`main.py` is 4,396 lines of which the first 3,570 are a commented-out earlier
version; the live bot starts at line 3571. `feature_engineering.py` is 2,193
lines with the first 994 commented out; the real `FeatureEngine` starts at line
1050. Search for the *last* definition of a symbol, not the first.

**5. `CLAUDE_API_KEY` does nothing.**
`config.py` reads it and sets `CLAUDE_MODEL = "claude-sonnet-4-20250514"`, but no
module imports either. `claude_brain.py` is 0 bytes. You do **not** need to
reissue this key to run the bot.

**6. Five empty files.**
`claude_brain.py`, `meta_learner.py`, and all three of `rl/`. They are 0 bytes —
placeholders for unbuilt features, not missing files.

**7. Two legacy modules that look live but aren't.**
`decision_engine.py` and `daily_limits.py` are not imported by `main.py`
(superseded by its inline `_make_decision` and by `risk_manager.py`). Editing
them changes nothing at runtime.

**8. ~~Banner/config drift.~~ Fixed.** Banners now print `MAX_TRADE_RISK` / `MAX_DAILY_LOSS` from config.

---

## 8. Debugging

Symptom-first, since the failure modes are distinctive.

| Symptom | Likely cause | Where to look |
|---|---|---|
| `MT5 initialization failed` | terminal not running, or not logged in | open MT5, log in, retry |
| `Symbol XAUUSD not found` | broker uses a suffix (`XAUUSD.r`, `XAUUSDm`) | set `MT5_SYMBOL` in `config.py` to the exact Market Watch name |
| `Strategy E model not found. E will be disabled.` | first retrain failed, or models elsewhere | `python src/retrain_e.py --force`; `MIDAS_PROJECT_ROOT` (SETUP Step 2) |
| `Strategy E retrain rejected` | new model failed its out-of-sample check | old model kept; inspect printed AUC (SETUP Step 5) |
| `Broker server time is UTC+X, but config's rule ... gives UTC+Y` | broker doesn't follow the configured clock rule | set `BROKER_TIME_ZONE` / `BROKER_TIME_SHIFT_HOURS` in `config.py` (default: New York + 7h = UTC+2 winter / UTC+3 summer; EET servers: `Europe/Athens`, 0) |
| `Telegram: disabled (no token/chat_id)` | `.env` missing, blank, or unreadable | `.env` goes in the project root, beside `src/` |
| Bot runs, no Telegram messages | token valid, wrong chat id | `/status` in the chat; notifier ignores other chat ids |
| `ORDER FAILED: 10027` | algo trading disabled in terminal | MT5 → Tools → Options → Expert Advisors |
| `ORDER FAILED: 10019` | insufficient margin | check balance/leverage |
| `ORDER FAILED: 10016` | SL/TP inside the broker's stop level | widen SL, or check `SPREAD_SIMULATION_PIPS` |
| Signals in dry-run, none live | risk veto | `/risk` — daily cap, session cap, or cooldown |
| Bot idle with a position open | working as designed | one trade at a time; `/positions` |
| `SyntaxError: f-string` | Python < 3.12 | see gotcha 2 |

**Trace a decision.** `_process_bar()` prints one line per bar:

```text
[14:35] 2654.30 | london | DayPnL: $-12.50
No trade | AD: B, D | E=0.612 | Regime: RANGING
```

That tells you which strategies fired, E's raw probability, and the regime —
enough to replay the decision table in [§2](#2-how-a-trade-happens) by hand.

**Persistent state.** `logs/daily_state.json` holds the daily counters. If the
bot thinks it is locked out and you disagree, inspect that file — deleting it
resets the day's risk tracking, which is exactly as dangerous as it sounds.

---

## 9. Safety notes

- **Always `--dry-run` first** after any credential change, machine change, or
  code edit. It exercises the full pipeline and places no orders.
- **`--mode demo` is the default** and requires no confirmation. `--mode live`
  demands a typed `CONFIRM`.
- **Shutdown does not close positions.** On exit the bot prints a warning and
  leaves an open trade for you to manage manually. SL/TP stay live at the broker.
- **One instance only.** Two bots on one terminal will both see "no position",
  both trade, and both blow the session limits.
