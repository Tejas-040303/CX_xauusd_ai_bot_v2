# Project MIDAS v2 — XAUUSD AI Trading Bot

An automated gold (XAUUSD) trading bot for MetaTrader 5. Five strategies vote on
every M5 candle: four rule-based (`A`–`D`) and one LightGBM model (`E`) that acts
as both a **validator** of the others and an **independent signal generator**. A
regime detector retunes the thresholds per market condition, a dynamic lot sizer
maps conviction to position size, and a risk manager enforces The5ers prop-firm
rules. Control and monitoring run through a Telegram bot.

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
| **Account model** | The5ers $5,000 evaluation (+10% target / −10% max drawdown) |
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

Two layers: The5ers' hard rules, and a tighter internal buffer that trips first.

| Control | Internal (`config.py`) | The5ers limit |
|---|---|---|
| Daily loss | `MAX_DAILY_LOSS` = **$100** | $250 (−5%) |
| Total drawdown | — | $4,500 floor (−10%) |
| Profit target | — | $5,500 (+10%) |
| Per-trade risk | `MAX_TRADE_RISK` = **$55** | — |
| Lot size | 0.01 – 0.05 | — |
| Trades per session | 2 (`asian` / `london` / `new_york`) | — |
| Cooldown | 600s between trades | — |

`RiskManager` persists to `logs/daily_state.json`, so limits survive a restart —
a crash-loop cannot reset your daily loss counter.

> The startup banner prints `Risk Cap: $40/trade`, but the enforced value is
> `MAX_TRADE_RISK = 55`. The banner string is stale; `config.py` is the truth.

---

## 4. Repository layout

```text
CX_xauusd_ai_bot_v2/
├── README.md              <- you are here
├── SETUP.md               <- the re-run runbook
├── requirements.txt
├── .env.example           <- copy to .env
└── src/
    ├── config.py              Central config + .env loading. EDIT PROJECT_ROOT.
    ├── main.py                Live bot. Active code starts at line 3571.
    ├── feature_engineering.py FeatureEngine. Active code starts at line 1050.
    │
    ├── LIVE RUNTIME ────────────────────────────────────────────
    ├── mt5_executor.py        Orders, positions, bars, account info
    ├── risk_manager.py        The5ers compliance, persisted state
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
    │   └── live_strategy_e.py LightGBM (live inference + debiasing)
    │
    ├── DATA + TRAINING PIPELINE ────────────────────────────────
    ├── data_downloader.py         MT5 -> data/raw/*.csv
    ├── strategy_e_data_prep.py    features + labels -> parquet
    ├── strategy_e_train.py        walk-forward LightGBM
    ├── strategy_e_clean_retrain.py debias -> retrain -> backtest  [USE THIS]
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
models/            strategy_e_clean_final.txt     from clean_retrain
                   strategy_e_clean_metadata.json
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

# Rebuild Strategy E
python src/strategy_e_data_prep.py
python src/strategy_e_clean_retrain.py

# Sanity check config
python src/config.py
```

---

## 7. Known issues & gotchas

These are real and were verified against the code. Read them before trusting a
run.

**1. `PROJECT_ROOT` is hardcoded to another machine.**
`src/config.py:14` is `Path(r"E:\GenAI\Projects\CX_xauusd_ai_bot_v2")`. On any
other path this silently creates `data/`, `models/`, `logs/` in the wrong place
and Strategy E will not be found. **Must be edited before first run** — see
SETUP.md Step 2.

**2. Python 3.12+ is required, not optional.**
`src/run_combined_bcd.py:317-321` uses PEP 701 nested same-quote f-strings
(`f"{f'${mf['total_pnl']:+.0f}':>10}"`). On 3.11 and below that file is a hard
`SyntaxError`. `main.py` separately needs 3.11+ for `datetime.UTC`. Verified:
every file compiles on 3.12 and 3.13; `run_combined_bcd.py` fails on 3.10/3.11.
Only that one research runner is affected — the live path works on 3.11 — but
3.12 avoids the whole question.

**3. Strategy E has a train/serve feature skew.**
`strategy_e_data_prep.py` computes `compute_fvg()` **and** `compute_smc()` on the
M5 frame, so the model trains on ~30 M5-level `fvg_*`, `bos_*`, `ob_*` and
`equal_*` columns. But `main.py:_build_features()` only calls `compute_shared`,
`compute_trend` and `compute_session_range` on M5 — it runs FVG/SMC on **M15
only**. At inference `LiveStrategyE.predict()` resolves the missing columns via
`row.get(col, np.nan)`, so those features arrive as `NaN`.

LightGBM accepts `NaN` natively and routes it down the default branch, so
**nothing crashes and no warning is printed** — E just predicts from a degraded
feature set that does not match its training distribution. Live behaviour will
not reproduce the backtest numbers. Strategies A–D are unaffected (they read the
`m15_*` columns, which are computed correctly).

Fix, if you want E's live predictions to match its backtest: add the two missing
calls in `main.py:_build_features()`:

```python
df_m5 = self.engine.compute_shared(df_m5, "M5")
df_m5 = self.engine.compute_fvg(df_m5)      # <- add
df_m5 = self.engine.compute_smc(df_m5)      # <- add
df_m5 = self.engine.compute_trend(df_m5)
df_m5 = self.engine.compute_session_range(df_m5)
```

Re-run a backtest afterwards — this changes E's inputs, so its behaviour will
shift. Until it is fixed, `--dry-run` output for `E_independent` signals should
be treated as unvalidated.

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

**8. Banner/config drift.** The banner says `$40/trade`; `MAX_TRADE_RISK` is 55.

---

## 8. Debugging

Symptom-first, since the failure modes are distinctive.

| Symptom | Likely cause | Where to look |
|---|---|---|
| `MT5 initialization failed` | terminal not running, or not logged in | open MT5, log in, retry |
| `Symbol XAUUSD not found` | broker uses a suffix (`XAUUSD.r`, `XAUUSDm`) | set `MT5_SYMBOL` in `config.py` to the exact Market Watch name |
| `Strategy E model not found. E will be disabled.` | `models/` empty, or wrong `PROJECT_ROOT` | `config.py:14`, then rebuild (SETUP Step 6) |
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
- Strategy E's live predictions are degraded until gotcha 3 is fixed.
