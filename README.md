
# Short-Horizon Crypto Microstructure  
### Signal Research & Backtesting Prototype

## Overview
This project explores **short-horizon (10–60s) trading signals** derived from **crypto market microstructure** using **public Binance Spot REST market data**.  
The goal is to build a **clean, reproducible research pipeline** for signal ideation, evaluation, and robustness analysis on **high-frequency (1s) data**, without relying on private or proprietary feeds.

This is a **research prototype**, not a production trading system.

---

## Objectives
- Ingest and preprocess **tick-level / high-frequency market data** from Binance Spot.
- Engineer **microstructure-inspired features** that capture short-term order flow and activity.
- Design and evaluate **simple, interpretable short-horizon signals**.
- Implement a **lightweight event-driven backtesting framework** with realistic assumptions.
- Document assumptions, limitations, and robustness checks clearly.

---

## Data
**Source:** Binance Spot REST API (public endpoints)  
**Instruments:** BTCUSDT (extendable to other pairs)

**Endpoints used:**
- `GET /api/v3/klines` — 1-second OHLCV bars  
- *(Optional)* `GET /api/v3/aggTrades` — aggregated trades for finer microstructure proxies

No private API keys are required.

---

## Feature Engineering
The project focuses on **simple but meaningful microstructure proxies**, such as:
- Taker-buy volume ratio
- Trade intensity (trades / second)
- Short-horizon realized volatility
- Volume and volatility imbalances
- Rolling z-scores and regime-normalized features

Features are engineered with **strict time-causality** (no lookahead).

---

## Signal Design
Signals are designed for **very short holding periods (10–60 seconds)** and follow a clear research loop:
1. Hypothesis (e.g. order-flow imbalance precedes short-term price moves)
2. Feature construction
3. Threshold-based or score-based signal definition
4. Backtest evaluation
5. Sensitivity & robustness checks

No machine learning models are assumed by default; the emphasis is on **statistical validity and interpretability**.

---

## Backtesting Framework
A minimal **event-driven backtesting harness** is implemented with:
- Explicit entry/exit logic
- Fixed holding horizons
- Transaction fees and slippage assumptions
- Turnover and exposure tracking

Outputs include:
- PnL curves
- Hit rate and average return per trade
- Sensitivity to thresholds and holding time
- Basic diagnostics to detect overfitting

---

## Reproducibility
The project is structured for reproducibility:
- Modular Python code
- Config-driven experiments
- Fixed random seeds where applicable
- Notebook-based reports summarizing results
- Clear documentation of assumptions and limitations

---

## Limitations
- Uses **public REST data only** (no full order book).
- Latency, queue position, and execution priority are not modeled.
- Results are indicative and **not deployable** as-is.

---

## Disclaimer
This project is for **educational and research purposes only**.  
It does **not** constitute trading advice or a production-ready strategy.

---

## Author
**Noel Pedrosa Alba**  
BSc Mathematical Engineering in Data Science  
GitHub: https://github.com/noelkei

