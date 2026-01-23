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
    # prices
    exec_price_col: str = "open"      # execution at bar open
    monitor_price_col: str = "close"  # monitor take-profit on bar close
    signal_col: str = "z_svi_60_600_lag1"
    intensity_col: str = "intensity_60"

    # gating options
    use_intensity_gate: bool = False
    intensity_q_low: float = 0.5
    intensity_q_high: Optional[float] = None

    # costs (per side, bps)
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    # tuning hygiene
    min_trades_tune: int = 30

    # randomness
    seed: int = 7


# ----------------------------
# Splits
# ----------------------------
def parse_split(split_str: str) -> List[float]:
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
    n = len(df)
    if n < 10:
        raise ValueError("Not enough rows to split meaningfully.")

    if len(fracs) == 2:
        f_train, f_test = fracs
        cut = int(np.floor(f_train * n))
        return {"train": df.iloc[:cut].copy(), "test": df.iloc[cut:].copy()}

    f_train, f_val, f_test = fracs
    cut1 = int(np.floor(f_train * n))
    cut2 = int(np.floor((f_train + f_val) * n))
    return {"train": df.iloc[:cut1].copy(), "val": df.iloc[cut1:cut2].copy(), "test": df.iloc[cut2:].copy()}


# ----------------------------
# Costs
# ----------------------------
def compute_cost_per_trade(cfg: BacktestConfig) -> float:
    # round-trip cost in return space
    return 2.0 * (cfg.fee_bps + cfg.slippage_bps) / 1e4


# ----------------------------
# Gating thresholds (TRAIN only)
# ----------------------------
def pick_intensity_bounds(train_df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[float, float]:
    if not cfg.use_intensity_gate:
        return -np.inf, np.inf

    x = train_df[cfg.intensity_col].dropna()
    if len(x) == 0:
        return -np.inf, np.inf

    low = float(x.quantile(cfg.intensity_q_low))

    if cfg.intensity_q_high is None:
        high = np.inf
    else:
        high = float(x.quantile(cfg.intensity_q_high))
        if high < low:
            high = low

    return low, high


# ----------------------------
# Trade simulation (with optional take-profit)
# ----------------------------
def simulate_non_overlapping_trades(
    df: pd.DataFrame,
    H: int,
    q: float,
    pt_bps: float,              # take-profit in bps (0 disables)
    cfg: BacktestConfig,
    intensity_low: float = -np.inf,
    intensity_high: float = np.inf,
) -> pd.DataFrame:
    """
    Non-overlapping trades, max holding H seconds, optional take-profit.

    Decision at time t (signal at t).
    Entry at (t+1) open.

    Time exit: at (entry_time + H) open.

    Take-profit exit (discrete + conservative):
      - monitor unrealized return using close at times:
          entry_time, entry_time+1, ..., entry_time+H-1
      - if threshold crossed at close(time u), exit at open(u+1)
      - ensures we do NOT "decide and fill" at the same open price.

    pt_bps = 0 -> disabled (pure time exit).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("df index must be DatetimeIndex.")
    if df.index.tz is None:
        raise RuntimeError("df index must be timezone-aware (UTC).")

    cost = compute_cost_per_trade(cfg)
    pt = float(pt_bps) / 1e4  # convert bps -> return threshold

    exec_px = df[cfg.exec_price_col]       # open
    mon_px = df[cfg.monitor_price_col]     # close
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
            if np.isnan(it) or it < intensity_low or it > intensity_high:
                continue

        if s > q:
            side = 1
        elif s < -q:
            side = -1
        else:
            continue

        entry_time = t + one_sec
        max_exit_time = entry_time + hold

        if entry_time not in df.index or max_exit_time not in df.index:
            continue

        entry_px = float(exec_px.at[entry_time])
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        # default: time exit
        exit_time = max_exit_time
        exit_reason = "time"

        # optional take-profit
        if pt > 0:
            # We monitor at closes from entry_time .. entry_time+H-1
            # If crossed at close(check_time), we exit next second open.
            for k in range(H):
                check_time = entry_time + pd.Timedelta(seconds=k)
                exit_candidate = check_time + one_sec
                if check_time not in df.index or exit_candidate not in df.index:
                    continue

                check_close = float(mon_px.at[check_time])
                if not np.isfinite(check_close):
                    continue

                unreal = side * (check_close / entry_px - 1.0)
                if unreal >= pt:
                    exit_time = exit_candidate
                    exit_reason = "take_profit"
                    break

        exit_px = float(exec_px.at[exit_time])
        if not np.isfinite(exit_px) or exit_px <= 0:
            continue

        gross_ret = side * (exit_px / entry_px - 1.0)
        pnl = gross_ret - cost

        hold_seconds = int((exit_time - entry_time) / one_sec)

        trades.append(
            {
                "decision_time": t,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": side,
                "signal": float(s),
                "q": float(q),
                "H": int(H),
                "pt_bps": float(pt_bps),
                "entry_px": entry_px,
                "exit_px": exit_px,
                "gross_ret": gross_ret,
                "cost": cost,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "hold_seconds": hold_seconds,
            }
        )

        # non-overlapping: next decision after the exit second
        next_allowed_t = exit_time + one_sec

    return pd.DataFrame(trades)


# ----------------------------
# Equity curve + drawdown
# ----------------------------
def trades_to_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["exit_time", "wealth", "drawdown"])

    t = trades.sort_values("exit_time").copy()
    t["wealth"] = (1.0 + t["pnl"]).cumprod()
    t["peak"] = t["wealth"].cummax()
    t["drawdown"] = t["wealth"] / t["peak"] - 1.0
    return t[["exit_time", "wealth", "drawdown"]]


# ----------------------------
# Metrics
# ----------------------------
def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "n_trades": 0,
            "avg_pnl": np.nan,
            "median_pnl": np.nan,
            "win_rate": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_trades": np.nan,
            "avg_hold_seconds": np.nan,
            "tp_exit_rate": np.nan,
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

    avg_hold = float(np.mean(trades["hold_seconds"])) if "hold_seconds" in trades.columns else np.nan
    tp_rate = float(np.mean(trades["exit_reason"] == "take_profit")) if "exit_reason" in trades.columns else np.nan

    return {
        "n_trades": int(n),
        "avg_pnl": avg,
        "median_pnl": med,
        "win_rate": win,
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "sharpe_trades": float(sharpe),
        "avg_hold_seconds": avg_hold,
        "tp_exit_rate": tp_rate,
    }


# ----------------------------
# Baseline Monte Carlo (supports take-profit logic)
# ----------------------------
def random_sign_baseline_mc(
    df_prices: pd.DataFrame,      # needs open+close columns
    trades: pd.DataFrame,
    H: int,
    pt_bps: float,
    cfg: BacktestConfig,
    n_trials: int = 500,
) -> Dict[str, float]:
    """
    Baseline: keep SAME entry times, randomize side (+1/-1), and apply the same exit policy:
      - time exit at entry+H open
      - optional take-profit triggered on close, exit next open

    This is more coherent than keeping the original exit_time when exits are side-dependent.
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
    pt = float(pt_bps) / 1e4

    open_px = df_prices[cfg.exec_price_col]
    close_px = df_prices[cfg.monitor_price_col]

    one_sec = pd.Timedelta(seconds=1)

    entries = pd.to_datetime(trades["entry_time"]).to_list()
    n = len(entries)

    # Precompute trigger returns (close) and fill returns (open at next second)
    # For each trade i, for k=0..H-1:
    #   trigger_ret[i,k] = close(entry+k)/entry_open - 1
    #   fill_ret[i,k]    = open(entry+k+1)/entry_open - 1   (exit at next open)
    trigger_ret = np.empty((n, H), dtype="float64")
    fill_ret = np.empty((n, H), dtype="float64")

    for i, et in enumerate(entries):
        entry_px = float(open_px.at[et])
        for k in range(H):
            check_t = et + pd.Timedelta(seconds=k)
            exit_t = check_t + one_sec
            trigger_ret[i, k] = float(close_px.at[check_t]) / entry_px - 1.0
            fill_ret[i, k] = float(open_px.at[exit_t]) / entry_px - 1.0

    avg_pnls = np.empty(n_trials, dtype="float64")
    total_returns = np.empty(n_trials, dtype="float64")
    sharpes = np.empty(n_trials, dtype="float64")

    for j in range(n_trials):
        side = rng.choice([-1.0, 1.0], size=n)

        # determine exit index per trade
        if pt <= 0:
            idx = np.full(n, H - 1, dtype=int)
        else:
            signed_trigger = side[:, None] * trigger_ret
            hit = signed_trigger >= pt
            any_hit = hit.any(axis=1)
            first = hit.argmax(axis=1)  # returns 0 if all False, so fix with any_hit
            idx = np.where(any_hit, first, H - 1)

        gross = side * fill_ret[np.arange(n), idx]
        pnl = gross - cost

        avg = float(np.mean(pnl))
        avg_pnls[j] = avg

        wealth = np.cumprod(1.0 + pnl)
        total_returns[j] = wealth[-1] - 1.0

        std = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
        sharpes[j] = (avg / std * np.sqrt(n)) if (std > 0 and n > 1) else np.nan

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
    pt_bps_list: List[float],
    cfg: BacktestConfig,
    intensity_low: float,
    intensity_high: float,
) -> pd.DataFrame:
    rows = []
    for H in H_list:
        for q in q_list:
            for pt_bps in pt_bps_list:
                trades = simulate_non_overlapping_trades(
                    tune_df, H=H, q=q, pt_bps=pt_bps, cfg=cfg,
                    intensity_low=intensity_low, intensity_high=intensity_high
                )
                summ = summarize_trades(trades)
                rows.append({"H": H, "q": q, "pt_bps": float(pt_bps), **summ})
    return pd.DataFrame(rows).sort_values(["H", "q", "pt_bps"]).reset_index(drop=True)


def pick_best_params_per_H(grid: pd.DataFrame, cfg: BacktestConfig) -> Dict[int, Tuple[float, float]]:
    """
    Choose (q, pt_bps) per H based on tune Sharpe, with minimum trade constraint.
    Tie-breakers: higher sharpe, higher n_trades, smaller pt_bps (simpler).
    """
    best: Dict[int, Tuple[float, float]] = {}
    for H, g in grid.groupby("H"):
        g2 = g[g["n_trades"] >= cfg.min_trades_tune].copy()
        if g2.empty:
            # fallback: smallest q, pt=0 if available else smallest pt
            g0 = g.sort_values(["q", "pt_bps"]).iloc[0]
            best[int(H)] = (float(g0["q"]), float(g0["pt_bps"]))
            continue

        g2 = g2.sort_values(
            ["sharpe_trades", "n_trades", "pt_bps"],
            ascending=[False, False, True],
        )
        row = g2.iloc[0]
        best[int(H)] = (float(row["q"]), float(row["pt_bps"]))
    return best


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features_1s.parquet")
    ap.add_argument("--out_dir", default="reports/tables")

    # Split control
    ap.add_argument("--split", type=str, default="0.6,0.2,0.2")

    # Grid
    ap.add_argument("--H_list", type=str, default="10,30,60")
    ap.add_argument("--q_list", type=str, default="1.0,1.5,2.0")
    ap.add_argument("--pt_bps_list", type=str, default="0", help="Take-profit candidates in bps, e.g. '0,5,10,20' (0 disables)")

    # Costs
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)

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
    pt_bps_list = [float(x.strip()) for x in args.pt_bps_list.split(",") if x.strip()]

    # Compute gating bounds ONLY on TRAIN
    intensity_low, intensity_high = pick_intensity_bounds(train_df, cfg)

    # Tune set = VAL if provided, else TRAIN
    tune_df = val_df if val_df is not None else train_df
    tune_name = "val" if val_df is not None else "train"

    # 1) Grid on tune set
    grid = grid_search(tune_df, H_list, q_list, pt_bps_list, cfg, intensity_low, intensity_high)
    grid_path = os.path.join(args.out_dir, f"grid_tune_{tune_name}.csv")
    grid.to_csv(grid_path, index=False)
    print(f"[OK] Saved tuning grid ({tune_name}) -> {grid_path}")

    # 2) Pick best (q, pt_bps) per H
    best_params = pick_best_params_per_H(grid, cfg)

    # 3) Evaluate on TEST once
    oos_rows = []
    for H in H_list:
        q, pt_bps = best_params[int(H)]
        trades_test = simulate_non_overlapping_trades(
            test_df, H=H, q=q, pt_bps=pt_bps, cfg=cfg, intensity_low=intensity_low, intensity_high=intensity_high
        )
        summ = summarize_trades(trades_test)

        base = random_sign_baseline_mc(
            df_prices=test_df[[cfg.exec_price_col, cfg.monitor_price_col]],
            trades=trades_test,
            H=H,
            pt_bps=pt_bps,
            cfg=cfg,
            n_trials=int(args.baseline_trials),
        )

        row = {
            "H": int(H),
            "q_chosen_tune": float(q),
            "pt_bps_chosen_tune": float(pt_bps),
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

    # Save config
    cfg_path = os.path.join(args.out_dir, "config.json")
    payload = {
        "cfg": asdict(cfg),
        "split": args.split,
        "H_list": H_list,
        "q_list": q_list,
        "pt_bps_list": pt_bps_list,
        "baseline_trials": int(args.baseline_trials),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Saved config -> {cfg_path}")


if __name__ == "__main__":
    main()
