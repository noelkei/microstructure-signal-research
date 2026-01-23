# src/backtest.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    price_col: str = "open"  # execution uses bar open
    signal_col: str = "z_svi_60_600_lag1"  # conservative default (lagged)
    intensity_col: str = "intensity_60"

    # gating options
    use_intensity_gate: bool = False
    intensity_q_low: float = 0.5      # median default
    intensity_q_high: Optional[float] = None  # e.g. 0.99 if you want an upper bound

    # costs
    fee_bps: float = 10.0
    slippage_bps: float = 2.0

    # model selection hygiene
    min_trades_tune: int = 30

    # randomness
    seed: int = 7


# ----------------------------
# Splits
# ----------------------------
def parse_split(split_str: str) -> List[float]:
    """
    split_str examples:
      "0.7,0.3" -> train,test
      "0.6,0.2,0.2" -> train,val,test
    """
    parts = [float(x.strip()) for x in split_str.split(",") if x.strip()]
    if len(parts) not in (2, 3):
        raise ValueError("--split must have 2 or 3 comma-separated floats, e.g. '0.7,0.3' or '0.6,0.2,0.2'")
    s = sum(parts)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"--split fractions must sum to 1.0, got sum={s}")
    for p in parts:
        if p <= 0:
            raise ValueError("All split fractions must be > 0.")
    return parts


def split_by_time(df: pd.DataFrame, fracs: List[float]) -> Dict[str, pd.DataFrame]:
    """
    Time-ordered split.
    Returns dict with keys:
      - if 2 fracs: train, test
      - if 3 fracs: train, val, test
    """
    n = len(df)
    if n < 10:
        raise ValueError("Not enough rows to split meaningfully.")

    if len(fracs) == 2:
        f_train, f_test = fracs
        cut = int(np.floor(f_train * n))
        return {
            "train": df.iloc[:cut].copy(),
            "test": df.iloc[cut:].copy(),
        }

    f_train, f_val, f_test = fracs
    cut1 = int(np.floor(f_train * n))
    cut2 = int(np.floor((f_train + f_val) * n))
    return {
        "train": df.iloc[:cut1].copy(),
        "val": df.iloc[cut1:cut2].copy(),
        "test": df.iloc[cut2:].copy(),
    }


# ----------------------------
# Costs
# ----------------------------
def compute_cost_per_trade(cfg: BacktestConfig) -> float:
    """
    Round-trip cost (entry+exit), expressed as return.
    Example: fee=10bps, slip=2bps => per-side=12bps => round-trip=24bps => 0.0024
    """
    return 2.0 * (cfg.fee_bps + cfg.slippage_bps) / 1e4


# ----------------------------
# Gating thresholds (computed ONLY on TRAIN)
# ----------------------------
def pick_intensity_bounds(train_df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[float, float]:
    """
    Returns (low, high) bounds for intensity gating, computed ONLY from TRAIN.
    If gating disabled: returns (-inf, +inf).
    If intensity_q_high is None: high=+inf.
    """
    if not cfg.use_intensity_gate:
        return -np.inf, np.inf

    col = cfg.intensity_col
    x = train_df[col].dropna()
    if len(x) == 0:
        return -np.inf, np.inf

    low = float(x.quantile(cfg.intensity_q_low))

    if cfg.intensity_q_high is None:
        high = np.inf
    else:
        high = float(x.quantile(cfg.intensity_q_high))
        # Safety: if user sets a high quantile too low by mistake, ensure high >= low
        if high < low:
            high = low

    return low, high


# ----------------------------
# Trade simulation
# ----------------------------
def simulate_non_overlapping_trades(
    df: pd.DataFrame,
    H: int,
    q: float,
    cfg: BacktestConfig,
    intensity_low: float = -np.inf,
    intensity_high: float = np.inf,
) -> pd.DataFrame:
    """
    Non-overlapping fixed-hold trades.

    Decision time: t
    Entry time: t + 1 second (next bar open)
    Exit time: entry_time + H seconds (bar open)

    Rule:
      if signal(t) > q => long
      if signal(t) < -q => short
      else no trade

    Intensity gate (optional):
      require intensity_low <= intensity(t) <= intensity_high
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("df index must be DatetimeIndex (timestamps).")

    cost = compute_cost_per_trade(cfg)

    price = df[cfg.price_col]
    signal = df[cfg.signal_col]
    intensity = df[cfg.intensity_col] if cfg.use_intensity_gate else None

    one_sec = pd.Timedelta(seconds=1)
    hold = pd.Timedelta(seconds=H)

    trades = []
    next_allowed_t = df.index[0]

    for t in df.index:
        if t < next_allowed_t:
            continue

        s = signal.at[t]
        if np.isnan(s):
            continue

        if cfg.use_intensity_gate:
            it = intensity.at[t]
            if np.isnan(it):
                continue
            if it < intensity_low or it > intensity_high:
                continue

        if s > q:
            side = 1
        elif s < -q:
            side = -1
        else:
            continue

        entry_time = t + one_sec
        exit_time = entry_time + hold

        if entry_time not in df.index or exit_time not in df.index:
            continue

        entry_px = float(price.at[entry_time])
        exit_px = float(price.at[exit_time])

        if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
            continue

        gross_ret = side * (exit_px / entry_px - 1.0)
        pnl = gross_ret - cost

        trades.append(
            {
                "decision_time": t,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": side,
                "signal": float(s),
                "q": float(q),
                "H": int(H),
                "entry_px": entry_px,
                "exit_px": exit_px,
                "gross_ret": gross_ret,
                "cost": cost,
                "pnl": pnl,
            }
        )

        # prevent overlap: next decision after the exit second
        next_allowed_t = exit_time + one_sec

    return pd.DataFrame(trades)


# ----------------------------
# Equity curve + drawdown
# ----------------------------
def trades_to_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """
    wealth_0 = 1
    wealth_k = wealth_{k-1} * (1 + pnl_k)

    drawdown = wealth / peak - 1, where peak is the running maximum of wealth.
    """
    if trades.empty:
        return pd.DataFrame(columns=["exit_time", "wealth", "drawdown"])

    t = trades.sort_values("exit_time").copy()
    t["wealth"] = (1.0 + t["pnl"]).cumprod()

    # cummax: running maximum so far
    t["peak"] = t["wealth"].cummax()

    # drawdown: how far below the peak we are (negative when below peak)
    t["drawdown"] = t["wealth"] / t["peak"] - 1.0
    return t[["exit_time", "wealth", "drawdown"]]


# ----------------------------
# Metrics
# ----------------------------
def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    """
    Small metrics set.
    sharpe_trades is "trade-level sharpe-ish": mean/std * sqrt(n_trades)
    """
    if trades.empty:
        return {
            "n_trades": 0,
            "avg_pnl": np.nan,
            "median_pnl": np.nan,
            "win_rate": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_trades": np.nan,
        }

    pnl = trades["pnl"].to_numpy(dtype="float64")
    n = len(pnl)

    avg = float(np.mean(pnl))
    med = float(np.median(pnl))
    win = float(np.mean(pnl > 0.0))

    eq = trades_to_equity_curve(trades)
    total_ret = float(eq["wealth"].iloc[-1] - 1.0)
    max_dd = float(eq["drawdown"].min())

    std = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
    sharpe = (avg / std * np.sqrt(n)) if (std > 0 and n > 1) else np.nan

    return {
        "n_trades": int(n),
        "avg_pnl": avg,
        "median_pnl": med,
        "win_rate": win,
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "sharpe_trades": float(sharpe),
    }


# ----------------------------
# Baseline (Monte Carlo)
# ----------------------------
def random_sign_baseline_mc(trades: pd.DataFrame, cfg: BacktestConfig, n_trials: int = 500) -> Dict[str, float]:
    """
    Baseline: keep SAME entry/exit times, randomize side (+1/-1), repeat n_trials times.
    Reports mean and p05/p95 for avg_pnl, total_return, sharpe_trades.

    This answers: "is your direction better than random, given the same timing?"
    """
    if trades.empty:
        return {
            "baseline_trials": int(n_trials),
            "baseline_avg_pnl_mean": np.nan,
            "baseline_avg_pnl_p05": np.nan,
            "baseline_avg_pnl_p95": np.nan,
            "baseline_total_return_mean": 0.0,
            "baseline_total_return_p05": 0.0,
            "baseline_total_return_p95": 0.0,
            "baseline_sharpe_mean": np.nan,
            "baseline_sharpe_p05": np.nan,
            "baseline_sharpe_p95": np.nan,
        }

    rng = np.random.default_rng(cfg.seed)
    cost = compute_cost_per_trade(cfg)

    # base (unsigned) returns per trade (exit/entry - 1)
    base_ret = (trades["exit_px"].to_numpy(dtype="float64") / trades["entry_px"].to_numpy(dtype="float64")) - 1.0
    n = len(base_ret)

    avg_pnls = np.empty(n_trials, dtype="float64")
    total_returns = np.empty(n_trials, dtype="float64")
    sharpes = np.empty(n_trials, dtype="float64")

    for i in range(n_trials):
        side = rng.choice([-1.0, 1.0], size=n)
        pnl = side * base_ret - cost

        avg = np.mean(pnl)
        avg_pnls[i] = avg

        # equity via cumprod; total return is last wealth - 1
        wealth = np.cumprod(1.0 + pnl)
        total_returns[i] = wealth[-1] - 1.0

        std = np.std(pnl, ddof=1) if n > 1 else 0.0
        sharpes[i] = (avg / std * np.sqrt(n)) if (std > 0 and n > 1) else np.nan

    def q(x, p):
        return float(np.nanquantile(x, p))

    return {
        "baseline_trials": int(n_trials),
        "baseline_avg_pnl_mean": float(np.nanmean(avg_pnls)),
        "baseline_avg_pnl_p05": q(avg_pnls, 0.05),
        "baseline_avg_pnl_p95": q(avg_pnls, 0.95),
        "baseline_total_return_mean": float(np.nanmean(total_returns)),
        "baseline_total_return_p05": q(total_returns, 0.05),
        "baseline_total_return_p95": q(total_returns, 0.95),
        "baseline_sharpe_mean": float(np.nanmean(sharpes)),
        "baseline_sharpe_p05": q(sharpes, 0.05),
        "baseline_sharpe_p95": q(sharpes, 0.95),
    }


# ----------------------------
# Grid + selection
# ----------------------------
def grid_search(
    tune_df: pd.DataFrame,
    H_list: List[int],
    q_list: List[float],
    cfg: BacktestConfig,
    intensity_low: float,
    intensity_high: float,
) -> pd.DataFrame:
    rows = []
    for H in H_list:
        for q in q_list:
            trades = simulate_non_overlapping_trades(
                tune_df, H=H, q=q, cfg=cfg, intensity_low=intensity_low, intensity_high=intensity_high
            )
            summ = summarize_trades(trades)
            rows.append({"H": H, "q": q, **summ})
    return pd.DataFrame(rows).sort_values(["H", "q"]).reset_index(drop=True)


def pick_best_q_per_H(grid: pd.DataFrame, cfg: BacktestConfig) -> Dict[int, float]:
    """
    Choose q for each H based on tune Sharpe, with minimum trade count constraint.
    """
    best: Dict[int, float] = {}
    for H, g in grid.groupby("H"):
        g2 = g[g["n_trades"] >= cfg.min_trades_tune].copy()
        if g2.empty:
            best[int(H)] = float(g["q"].min())
            continue
        g2 = g2.sort_values("sharpe_trades", ascending=False)
        best[int(H)] = float(g2.iloc[0]["q"])
    return best


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features_1s.parquet")
    ap.add_argument("--out_dir", default="reports/tables")

    # Split control
    ap.add_argument("--split", type=str, default="0.6,0.2,0.2", help="Either 'train,test' or 'train,val,test' fractions.")

    # Grid
    ap.add_argument("--H_list", type=str, default="10,30,60")
    ap.add_argument("--q_list", type=str, default="1.0,1.5,2.0")

    # Costs
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=2.0)

    # Signal + gating
    ap.add_argument("--signal_col", type=str, default="z_svi_60_600_lag1")
    ap.add_argument("--use_intensity_gate", action="store_true")
    ap.add_argument("--intensity_q_low", type=float, default=0.5)
    ap.add_argument("--intensity_q_high", type=float, default=None)

    # Baseline MC
    ap.add_argument("--baseline_trials", type=int, default=500)

    # Misc
    ap.add_argument("--min_trades_tune", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    cfg = BacktestConfig(
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        signal_col=str(args.signal_col),
        use_intensity_gate=bool(args.use_intensity_gate),
        intensity_q_low=float(args.intensity_q_low),
        intensity_q_high=None if args.intensity_q_high is None else float(args.intensity_q_high),
        min_trades_tune=int(args.min_trades_tune),
        seed=int(args.seed),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.features)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("features parquet must preserve DateTimeIndex.")
    if df.index.tz is None:
        raise RuntimeError("features index must be timezone-aware (UTC).")

    fracs = parse_split(args.split)
    splits = split_by_time(df, fracs)

    train_df = splits["train"]
    val_df = splits.get("val")
    test_df = splits["test"]

    # Parse lists
    H_list = [int(x.strip()) for x in args.H_list.split(",") if x.strip()]
    q_list = [float(x.strip()) for x in args.q_list.split(",") if x.strip()]

    # Compute gating bounds ONLY on TRAIN
    intensity_low, intensity_high = pick_intensity_bounds(train_df, cfg)

    # Tune set = VAL if provided, else TRAIN
    tune_df = val_df if val_df is not None else train_df
    tune_name = "val" if val_df is not None else "train"

    # 1) Grid on tune set (val preferred)
    grid = grid_search(tune_df, H_list, q_list, cfg, intensity_low, intensity_high)
    grid_path = os.path.join(args.out_dir, f"grid_tune_{tune_name}.csv")
    grid.to_csv(grid_path, index=False)
    print(f"[OK] Saved tuning grid ({tune_name}) -> {grid_path}")

    # 2) Pick best q per H using tune results
    best_q = pick_best_q_per_H(grid, cfg)

    # 3) Evaluate on TEST once
    oos_rows = []
    for H in H_list:
        q = best_q[int(H)]
        trades_test = simulate_non_overlapping_trades(
            test_df, H=H, q=q, cfg=cfg, intensity_low=intensity_low, intensity_high=intensity_high
        )
        summ = summarize_trades(trades_test)
        base = random_sign_baseline_mc(trades_test, cfg, n_trials=int(args.baseline_trials))

        row = {
            "H": int(H),
            "q_chosen_tune": float(q),
            "tune_set": tune_name,
            "intensity_gate": bool(cfg.use_intensity_gate),
            "intensity_low": float(intensity_low),
            "intensity_high": float(intensity_high) if np.isfinite(intensity_high) else np.inf,
            **summ,
            **base,
        }
        oos_rows.append(row)

        trades_path = os.path.join(args.out_dir, f"trades_test_H{H}.csv")
        trades_test.to_csv(trades_path, index=False)
        print(f"[OK] Saved TEST trades -> {trades_path} (n={len(trades_test)})")

    oos_res = pd.DataFrame(oos_rows).sort_values("H").reset_index(drop=True)
    oos_path = os.path.join(args.out_dir, "best_oos.csv")
    oos_res.to_csv(oos_path, index=False)
    print(f"[OK] Saved TEST summary -> {oos_path}")

    # Save config for reproducibility
    cfg_path = os.path.join(args.out_dir, "config.json")
    payload = {"cfg": asdict(cfg), "split": args.split, "H_list": H_list, "q_list": q_list, "baseline_trials": int(args.baseline_trials)}
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Saved config -> {cfg_path}")


if __name__ == "__main__":
    main()
