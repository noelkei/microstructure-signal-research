
# Short-Horizon Crypto Microstructure
### Signal Research & Event-Driven Backtesting (Binance Spot REST, 1s)

## Overview
This repository is a **research prototype** for exploring **short-horizon (10–60s) trading signals** built from **public Binance Spot REST market data**.  
It implements a reproducible pipeline:

**Binance REST → 1s bars + aggTrades → causal features → event-driven, non-overlapping backtest (with costs) → OOS evaluation + Monte Carlo baseline + diagnostics**

📓 **Notebook report:** [`notebooks/research_report.ipynb`](./notebooks/research_report.ipynb)

The focus is **realism and auditability**: assumptions are explicit, execution is modeled, and results are reported with failure modes.

> Not a production trading system.

---

## What’s in here
### Data (public only)
**Source:** Binance Spot REST API  
**Instrument:** BTCUSDT (extendable)

**Endpoints used**
- `GET /api/v3/klines` — 1-second OHLCV bars (core)
- `GET /api/v3/aggTrades` — aggregated trades (optional, used for microstructure proxies)

No private keys required.

### Features (strictly causal)
All features are engineered with **no lookahead** (rolling windows use past-only data; decisions at time `t` execute at `t+1s`).

Examples of implemented microstructure-style proxies:
- **SVI (from klines):** taker-buy ratio × volume (signed volume proxy)
- **OFI proxy (from aggTrades):** taker buy base − taker sell base (tape-level flow proxy)
- **CFI / count-flow variants (from aggTrades 1s):** buy count − sell count, ratios, rolling z-scores
- **max_share:** concentration within the second (dominance of largest trade)

### Backtest (event-driven, non-overlapping)
Trades are simulated directly from execution prices (no need for a precomputed future-return label):
- Signal observed at **decision time** `t` (features up to `t`)
- Entry executes at **next-second open**: `open[t+1]`
- Exit executes at:
  - **time exit**: `open[t+1+H]`, or
  - **take-profit**: monitored on `close` during the holding window, filled on the next `open`

**Costs**
- Constant round-trip cost per trade:  
  <b>cost</b> = 2·(fee<sub>bps</sub> + slippage<sub>bps</sub>) / 10<sup>4</sup>

(defaults: `fee_bps=10`, `slippage_bps=5` → ~30 bps round-trip)

### Evaluation hygiene
- **Time split**: train/val/test (default `0.6/0.2/0.2`)
- **Tuning**: grid-search on **val only**, test untouched
- **Baseline**: Monte Carlo **random-sign** baseline that preserves the same entry times/holding/take-profit logic and pays the same costs
- **Diagnostics**: cost dominance, low-N configs, boundary selections, and imputation sensitivity

---

## Repository structure (high level)
Typical flow (paths are defaults; see script args):
- `src/fetch_binance.py` — fetch raw klines (1s)
- `src/fetch_aggtrades.py` — fetch raw aggTrades
- `src/build_bars.py` — build clean 1s bars
- `src/build_trades_1s.py` — aggregate aggTrades to 1s tape features
- `src/features.py` — merge + engineer features → `data/processed/features_1s.parquet`
- `src/backtest.py` — run grid-tune + OOS backtest per run config → `reports/tables/<run>/`
- `src/report.py` — build leaderboards + figures from all runs → `reports/figures/`

---

## Quickstart (end-to-end)
### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2) Build dataset (1s bars + optional aggTrades tape)

Run the scripts with your desired date range / output paths (see `--help` in each file).
Example (illustrative; adjust args to your setup):

```bash
python src/fetch_binance.py --symbol BTCUSDT ...
python src/fetch_aggtrades.py --symbol BTCUSDT ...
python src/build_bars.py ...
python src/build_trades_1s.py ...
python src/features.py ...
```

### 3) Run backtests (one “run” = one signal + gating configuration)

`src/backtest.py` writes per-run outputs under `reports/tables/<run_name>/`.

Example:

```bash
python src/backtest.py \
  --features data/processed/features_1s.parquet \
  --out_dir reports/tables \
  --run_name cfi_base \
  --signal_col z_cfi_60_600_lag1 \
  --H_list 10,30,60 \
  --q_list 1.0,1.5,2.0 \
  --pt_bps_list 0,5,10,25 \
  --split 0.6,0.2,0.2 \
  --baseline_trials 500
```

### 4) Build aggregate report (leaderboards + figures across all runs)

```bash
python src/report.py --runs_root reports/tables --out_fig_dir reports/figures --out_tables_dir reports/tables
```

---

## Outputs

Per run (`reports/tables/<run_name>/`):

* `grid_tune_val.csv` (or `grid_tune_train.csv`)
* `trades_test_H{H}.csv` for each H
* `best_oos.csv` (one row per H)
* `config.json`

Aggregate:

* `reports/tables/leaderboard_compact.csv`
* `reports/figures/` (equity curves, leaderboards, parameter heatmaps, etc.)

---

## Key assumptions & limitations

* **Public REST only** (no historical L2 / order book replay)
* Execution is simplified:

  * constant fees + slippage
  * no queue priority, no partial fills, no latency modeling
* Take-profit is path-dependent but still simplified (monitor close, fill next open)
* Results are **indicative** and primarily useful for **research iteration**, not deployment

---

## Notebook report

A notebook in this repo documents:

* data validation (missing seconds, duplicates, imputation rates)
* feature definitions + causality checks
* signal comparisons (e.g., SVI≈OFI replication, CFI behaving differently)
* OOS results + “edge vs cost” evidence
* Monte Carlo baseline definition and alpha computation
* failure modes and next steps

---

## Disclaimer

This project is for **educational and research purposes only**.
It does **not** constitute trading advice and is **not production-ready**.

---

## Author

**Noel Pedrosa Alba**
BSc Mathematical Engineering in Data Science
GitHub: [https://github.com/noelkei](https://github.com/noelkei)
