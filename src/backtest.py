# src/backtest.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    # prices
    exec_price_col: str = "open"      # execution at bar open
    monitor_price_col: str = "close"  # monitor take-profit on bar close

    # signal (supports 1 col or an ensemble of cols)
    signal_cols: List[str] = field(default_factory=lambda: ["z_svi_60_600_lag1"])
    signal_weights: Optional[List[float]] = None  # if None -> equal weights
    side_mode: str = "trend"  # "trend": long if s>q; "contrarian": long if s<-q
    signal_clip: Optional[float] = None  # optional clip of signal values

    # primary gate (usually intensity from klines)
    intensity_col: str = "intensity_60"
    use_intensity_gate: bool = False
    intensity_q_low: float = 0.5
    intensity_q_high: Optional[float] = None

    # optional secondary gate (any column, e.g. intensity_trades_60, max_share_1s, etc.)
    use_aux_gate: bool = False
    aux_gate_col: str = ""
    aux_gate_q_low: float = 0.5
    aux_gate_q_high: Optional[float] = None

    # optional data-quality filters
    skip_imputed_bars: bool = False        # uses 'is_imputed' from bars
    skip_imputed_trades: bool = False      # uses 'is_imputed_trades' from trades_1s

    # costs (per side, bps)
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    # tuning hygiene
    min_trades_tune: int = 30
    select_metric: str = "sharpe_trades"   # metric used to pick best params per H

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
    return 2.0 * (cfg.fee_bps + cfg.slippage_bps) / 1e4


# ----------------------------
# Column checks
# ----------------------------
def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


# ----------------------------
# Signal construction
# ----------------------------
def _build_signal(df: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    cols = cfg.signal_cols
    _require_columns(df, cols, "features dataframe (signal)")

    sig_df = df[cols].astype("float64")

    # strict: require all components present to produce a signal
    ok = sig_df.notna().all(axis=1)

    if cfg.signal_weights is None:
        w = np.ones(len(cols), dtype="float64") / float(len(cols))
    else:
        if len(cfg.signal_weights) != len(cols):
            raise ValueError(f"--signal_weights length must match --signal_col list. Got {len(cfg.signal_weights)} vs {len(cols)}")
        w = np.array(cfg.signal_weights, dtype="float64")
        s = float(np.sum(np.abs(w)))
        if not np.isfinite(s) or s <= 0:
            raise ValueError("--signal_weights must have finite non-zero magnitude.")
        w = w / s  # normalize by L1 for stability

    combined = (sig_df.to_numpy(dtype="float64") * w[None, :]).sum(axis=1)
    s = pd.Series(combined, index=df.index, name="signal")

    if cfg.signal_clip is not None:
        c = float(cfg.signal_clip)
        if c <= 0 or not np.isfinite(c):
            raise ValueError("--signal_clip must be a positive finite number.")
        s = s.clip(lower=-c, upper=c)

    s = s.where(ok, np.nan)
    return s


# ----------------------------
# Gating thresholds (TRAIN only)
# ----------------------------
def _pick_gate_bounds(train_df: pd.DataFrame, col: str, q_low: float, q_high: Optional[float]) -> Tuple[float, float]:
    x = train_df[col].dropna()
    if len(x) == 0:
        return -np.inf, np.inf

    low = float(x.quantile(q_low))
    if q_high is None:
        high = np.inf
    else:
        high = float(x.quantile(q_high))
        if high < low:
            high = low
    return low, high


def pick_intensity_bounds(train_df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[float, float]:
    if not cfg.use_intensity_gate:
        return -np.inf, np.inf
    _require_columns(train_df, [cfg.intensity_col], "train_df (intensity gate)")
    return _pick_gate_bounds(train_df, cfg.intensity_col, cfg.intensity_q_low, cfg.intensity_q_high)


def pick_aux_gate_bounds(train_df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[float, float]:
    if not cfg.use_aux_gate:
        return -np.inf, np.inf
    if not cfg.aux_gate_col:
        raise ValueError("--use_aux_gate requires --aux_gate_col")
    _require_columns(train_df, [cfg.aux_gate_col], "train_df (aux gate)")
    return _pick_gate_bounds(train_df, cfg.aux_gate_col, cfg.aux_gate_q_low, cfg.aux_gate_q_high)


# ----------------------------
# Trade simulation (with optional take-profit)
# ----------------------------
def simulate_non_overlapping_trades(
    df: pd.DataFrame,
    H: int,
    q: float,
    pt_bps: float,
    cfg: BacktestConfig,
    intensity_low: float = -np.inf,
    intensity_high: float = np.inf,
    aux_low: float = -np.inf,
    aux_high: float = np.inf,
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("df index must be DatetimeIndex.")
    if df.index.tz is None:
        raise RuntimeError("df index must be timezone-aware (UTC).")

    req = [cfg.exec_price_col, cfg.monitor_price_col]
    req += cfg.signal_cols

    if cfg.use_intensity_gate:
        req.append(cfg.intensity_col)
    if cfg.use_aux_gate:
        if not cfg.aux_gate_col:
            raise ValueError("--use_aux_gate requires --aux_gate_col")
        req.append(cfg.aux_gate_col)
    if cfg.skip_imputed_bars:
        req.append("is_imputed")
    if cfg.skip_imputed_trades:
        req.append("is_imputed_trades")

    _require_columns(df, req, "features dataframe")

    cost = compute_cost_per_trade(cfg)
    pt = float(pt_bps) / 1e4

    exec_px = df[cfg.exec_price_col]
    mon_px = df[cfg.monitor_price_col]

    signal = _build_signal(df, cfg)

    intensity = df[cfg.intensity_col] if cfg.use_intensity_gate else None
    aux = df[cfg.aux_gate_col] if cfg.use_aux_gate else None

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

        # quality filters at decision time
        if cfg.skip_imputed_bars and bool(df.at[t, "is_imputed"]):
            continue
        if cfg.skip_imputed_trades and bool(df.at[t, "is_imputed_trades"]):
            continue

        # primary gate
        if cfg.use_intensity_gate:
            it = intensity.at[t]
            if np.isnan(it) or it < intensity_low or it > intensity_high:
                continue

        # aux gate
        if cfg.use_aux_gate:
            gx = aux.at[t]
            if np.isnan(gx) or gx < aux_low or gx > aux_high:
                continue

        # side decision
        if cfg.side_mode == "trend":
            if s > q:
                side = 1
            elif s < -q:
                side = -1
            else:
                continue
        elif cfg.side_mode == "contrarian":
            if s > q:
                side = -1
            elif s < -q:
                side = 1
            else:
                continue
        else:
            raise ValueError(f"Unsupported side_mode: {cfg.side_mode}")

        entry_time = t + one_sec
        max_exit_time = entry_time + hold

        if entry_time not in df.index or max_exit_time not in df.index:
            continue

        # quality filters at entry time (conservative)
        if cfg.skip_imputed_bars and bool(df.at[entry_time, "is_imputed"]):
            continue
        if cfg.skip_imputed_trades and bool(df.at[entry_time, "is_imputed_trades"]):
            continue

        entry_px = float(exec_px.at[entry_time])
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        exit_time = max_exit_time
        exit_reason = "time"

        if pt > 0:
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
                "side_mode": cfg.side_mode,
            }
        )

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
    df_prices: pd.DataFrame,
    trades: pd.DataFrame,
    H: int,
    pt_bps: float,
    cfg: BacktestConfig,
    n_trials: int = 500,
) -> Dict[str, float]:
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

    _require_columns(df_prices, [cfg.exec_price_col, cfg.monitor_price_col], "df_prices")

    rng = np.random.default_rng(cfg.seed)
    cost = compute_cost_per_trade(cfg)
    pt = float(pt_bps) / 1e4

    open_px = df_prices[cfg.exec_price_col]
    close_px = df_prices[cfg.monitor_price_col]

    one_sec = pd.Timedelta(seconds=1)

    entries = pd.to_datetime(trades["entry_time"]).to_list()
    n = len(entries)

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

        if pt <= 0:
            idx = np.full(n, H - 1, dtype=int)
        else:
            signed_trigger = side[:, None] * trigger_ret
            hit = signed_trigger >= pt
            any_hit = hit.any(axis=1)
            first = hit.argmax(axis=1)
            idx = np.where(any_hit, first, H - 1)

        gross = side * fill_ret[np.arange(n), idx]
        pnl = gross - cost

        avg = float(np.mean(pnl))
        avg_pnls[j] = avg

        wealth = np.cumprod(1.0 + pnl)
        total_returns[j] = wealth[-1] - 1.0

        std = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
        sharpes[j] = (avg / std * np.sqrt(n)) if (std > 0 and n > 1) else np.nan

    def qnt(x, p):
        return float(np.nanquantile(x, p))

    return {
        "baseline_trials": int(n_trials),
        "baseline_avg_pnl_mean": float(np.nanmean(avg_pnls)),
        "baseline_avg_pnl_p05": qnt(avg_pnls, 0.05),
        "baseline_avg_pnl_p95": qnt(avg_pnls, 0.95),
        "baseline_total_return_mean": float(np.nanmean(total_returns)),
        "baseline_total_return_p05": qnt(total_returns, 0.05),
        "baseline_total_return_p95": qnt(total_returns, 0.95),
        "baseline_sharpe_mean": float(np.nanmean(sharpes)),
        "baseline_sharpe_p05": qnt(sharpes, 0.05),
        "baseline_sharpe_p95": qnt(sharpes, 0.95),
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
    aux_low: float,
    aux_high: float,
) -> pd.DataFrame:
    rows = []
    for H in H_list:
        for q in q_list:
            for pt_bps in pt_bps_list:
                trades = simulate_non_overlapping_trades(
                    tune_df,
                    H=H,
                    q=q,
                    pt_bps=pt_bps,
                    cfg=cfg,
                    intensity_low=intensity_low,
                    intensity_high=intensity_high,
                    aux_low=aux_low,
                    aux_high=aux_high,
                )
                summ = summarize_trades(trades)
                rows.append({"H": H, "q": q, "pt_bps": float(pt_bps), **summ})
    return pd.DataFrame(rows).sort_values(["H", "q", "pt_bps"]).reset_index(drop=True)


def pick_best_params_per_H(grid: pd.DataFrame, cfg: BacktestConfig) -> Dict[int, Tuple[float, float]]:
    metric = cfg.select_metric
    if metric not in grid.columns:
        raise RuntimeError(f"select_metric='{metric}' not found in grid columns: {sorted(grid.columns)}")

    best: Dict[int, Tuple[float, float]] = {}
    for H, g in grid.groupby("H"):
        g2 = g[g["n_trades"] >= cfg.min_trades_tune].copy()
        if g2.empty:
            g0 = g.sort_values(["q", "pt_bps"]).iloc[0]
            best[int(H)] = (float(g0["q"]), float(g0["pt_bps"]))
            continue

        # primary objective: maximize metric; tie-breakers: more trades, smaller pt_bps, smaller q (simpler)
        g2 = g2.sort_values(
            [metric, "n_trades", "pt_bps", "q"],
            ascending=[False, False, True, True],
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
    ap.add_argument("--run_name", default=None, help="Optional: create outputs under --out_dir/<run_name>/")

    ap.add_argument("--split", type=str, default="0.6,0.2,0.2")

    ap.add_argument("--H_list", type=str, default="10,30,60")
    ap.add_argument("--q_list", type=str, default="1.0,1.5,2.0")
    ap.add_argument("--pt_bps_list", type=str, default="0", help="Take-profit candidates in bps, e.g. '0,5,10,20'")

    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)

    # Signal
    ap.add_argument("--signal_col", type=str, default="z_svi_60_600_lag1", help="Single col or comma-separated cols for ensemble")
    ap.add_argument("--signal_weights", type=str, default=None, help="Comma-separated weights matching signal_col list")
    ap.add_argument("--side_mode", type=str, default="trend", choices=["trend", "contrarian"])
    ap.add_argument("--signal_clip", type=float, default=None, help="Optional clip for signal values (e.g. 5.0)")

    # Primary gate (intensity)
    ap.add_argument("--use_intensity_gate", action="store_true")
    ap.add_argument("--intensity_col", type=str, default="intensity_60")
    ap.add_argument("--intensity_q_low", type=float, default=0.5)
    ap.add_argument("--intensity_q_high", type=float, default=None)

    # Secondary gate (any column)
    ap.add_argument("--use_aux_gate", action="store_true")
    ap.add_argument("--aux_gate_col", type=str, default="")
    ap.add_argument("--aux_gate_q_low", type=float, default=0.5)
    ap.add_argument("--aux_gate_q_high", type=float, default=None)

    # Quality filters
    ap.add_argument("--skip_imputed_bars", action="store_true")
    ap.add_argument("--skip_imputed_trades", action="store_true")

    # Selection objective
    ap.add_argument(
        "--select_metric",
        type=str,
        default="sharpe_trades",
        help="Metric to maximize when picking best params per H (e.g. sharpe_trades, avg_pnl, total_return)",
    )

    ap.add_argument("--baseline_trials", type=int, default=500)

    ap.add_argument("--min_trades_tune", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    # resolve output dir
    out_dir = str(args.out_dir)
    if args.run_name is not None and str(args.run_name).strip():
        out_dir = os.path.join(out_dir, str(args.run_name).strip())
    os.makedirs(out_dir, exist_ok=True)

    # parse signal list
    signal_cols = [c.strip() for c in str(args.signal_col).split(",") if c.strip()]
    if not signal_cols:
        raise ValueError("--signal_col must contain at least one column name.")

    signal_weights = None
    if args.signal_weights is not None:
        ws = [float(x.strip()) for x in str(args.signal_weights).split(",") if x.strip()]
        signal_weights = ws

    cfg = BacktestConfig(
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        signal_cols=signal_cols,
        signal_weights=signal_weights,
        side_mode=str(args.side_mode),
        signal_clip=None if args.signal_clip is None else float(args.signal_clip),
        use_intensity_gate=bool(args.use_intensity_gate),
        intensity_col=str(args.intensity_col),
        intensity_q_low=float(args.intensity_q_low),
        intensity_q_high=None if args.intensity_q_high is None else float(args.intensity_q_high),
        use_aux_gate=bool(args.use_aux_gate),
        aux_gate_col=str(args.aux_gate_col),
        aux_gate_q_low=float(args.aux_gate_q_low),
        aux_gate_q_high=None if args.aux_gate_q_high is None else float(args.aux_gate_q_high),
        skip_imputed_bars=bool(args.skip_imputed_bars),
        skip_imputed_trades=bool(args.skip_imputed_trades),
        min_trades_tune=int(args.min_trades_tune),
        seed=int(args.seed),
        select_metric=str(args.select_metric),
    )

    df = pd.read_parquet(args.features)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("features parquet must preserve DateTimeIndex.")
    if df.index.tz is None:
        raise RuntimeError("features index must be timezone-aware (UTC).")

    # base column requirements
    base_req = [cfg.exec_price_col, cfg.monitor_price_col] + cfg.signal_cols
    if cfg.use_intensity_gate:
        base_req.append(cfg.intensity_col)
    if cfg.use_aux_gate:
        if not cfg.aux_gate_col:
            raise ValueError("--use_aux_gate requires --aux_gate_col")
        base_req.append(cfg.aux_gate_col)
    if cfg.skip_imputed_bars:
        base_req.append("is_imputed")
    if cfg.skip_imputed_trades:
        base_req.append("is_imputed_trades")
    _require_columns(df, base_req, "features dataframe")

    fracs = parse_split(args.split)
    splits = split_by_time(df, fracs)

    train_df = splits["train"]
    val_df = splits.get("val")
    test_df = splits["test"]

    H_list = [int(x.strip()) for x in args.H_list.split(",") if x.strip()]
    q_list = [float(x.strip()) for x in args.q_list.split(",") if x.strip()]
    pt_bps_list = [float(x.strip()) for x in args.pt_bps_list.split(",") if x.strip()]

    intensity_low, intensity_high = pick_intensity_bounds(train_df, cfg)
    aux_low, aux_high = pick_aux_gate_bounds(train_df, cfg)

    tune_df = val_df if val_df is not None else train_df
    tune_name = "val" if val_df is not None else "train"

    grid = grid_search(tune_df, H_list, q_list, pt_bps_list, cfg, intensity_low, intensity_high, aux_low, aux_high)
    grid_path = os.path.join(out_dir, f"grid_tune_{tune_name}.csv")
    grid.to_csv(grid_path, index=False)
    print(f"[OK] Saved tuning grid ({tune_name}) -> {grid_path}")

    best_params = pick_best_params_per_H(grid, cfg)

    oos_rows = []
    for H in H_list:
        q, pt_bps = best_params[int(H)]
        trades_test = simulate_non_overlapping_trades(
            test_df,
            H=H,
            q=q,
            pt_bps=pt_bps,
            cfg=cfg,
            intensity_low=intensity_low,
            intensity_high=intensity_high,
            aux_low=aux_low,
            aux_high=aux_high,
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
            "signal_col": ",".join(cfg.signal_cols),
            "signal_weights": "" if cfg.signal_weights is None else ",".join([str(x) for x in cfg.signal_weights]),
            "side_mode": cfg.side_mode,
            "signal_clip": "" if cfg.signal_clip is None else float(cfg.signal_clip),
            "intensity_gate": bool(cfg.use_intensity_gate),
            "intensity_col": cfg.intensity_col if cfg.use_intensity_gate else "",
            "intensity_low": float(intensity_low),
            "intensity_high": float(intensity_high) if np.isfinite(intensity_high) else np.inf,
            "aux_gate": bool(cfg.use_aux_gate),
            "aux_gate_col": cfg.aux_gate_col if cfg.use_aux_gate else "",
            "aux_low": float(aux_low),
            "aux_high": float(aux_high) if np.isfinite(aux_high) else np.inf,
            "skip_imputed_bars": bool(cfg.skip_imputed_bars),
            "skip_imputed_trades": bool(cfg.skip_imputed_trades),
            "select_metric": cfg.select_metric,
            **summ,
            **base,
        }
        oos_rows.append(row)

        trades_path = os.path.join(out_dir, f"trades_test_H{H}.csv")
        trades_test.to_csv(trades_path, index=False)
        print(f"[OK] Saved TEST trades -> {trades_path} (n={len(trades_test)})")

    oos_res = pd.DataFrame(oos_rows).sort_values("H").reset_index(drop=True)
    oos_path = os.path.join(out_dir, "best_oos.csv")
    oos_res.to_csv(oos_path, index=False)
    print(f"[OK] Saved TEST summary -> {oos_path}")

    cfg_path = os.path.join(out_dir, "config.json")
    payload = {
        "cfg": asdict(cfg),
        "split": args.split,
        "H_list": H_list,
        "q_list": q_list,
        "pt_bps_list": pt_bps_list,
        "baseline_trials": int(args.baseline_trials),
        "out_dir": out_dir,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Saved config -> {cfg_path}")


if __name__ == "__main__":
    main()
