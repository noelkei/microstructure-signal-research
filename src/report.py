# src/report.py
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_run_dirs(runs_root: str) -> List[str]:
    pattern = os.path.join(runs_root, "*", "best_oos.csv")
    files = glob.glob(pattern)
    return sorted({os.path.dirname(f) for f in files})


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_times(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c])
    return out


def _choose_cmap_and_norm(values: np.ndarray, value_col: str):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None, None

    vmin = float(np.min(v))
    vmax = float(np.max(v))

    signed_like = any(
        k in value_col
        for k in [
            "pnl", "return", "sharpe", "avg_pnl",
            "net_total_return", "gross_total_return", "total_return",
            "alpha_", "drawdown"
        ]
    )

    if signed_like and vmin < 0.0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap("RdYlGn")
        return cmap, norm

    # If it's signed-like but doesn't cross 0, still use RdYlGn for intuitive "bad->good"
    if signed_like:
        cmap = plt.get_cmap("RdYlGn")
        return cmap, None

    # fallback
    return plt.get_cmap("viridis"), None


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# ----------------------------
# Data loading
# ----------------------------
def load_best_oos(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "best_oos.csv")
    df = pd.read_csv(path)
    df["run"] = os.path.basename(run_dir)
    return df


def load_grid_tune(run_dir: str) -> Optional[pd.DataFrame]:
    p_val = os.path.join(run_dir, "grid_tune_val.csv")
    p_train = os.path.join(run_dir, "grid_tune_train.csv")
    if os.path.exists(p_val):
        return pd.read_csv(p_val)
    if os.path.exists(p_train):
        return pd.read_csv(p_train)
    return None


def load_trades(run_dir: str, H: int) -> pd.DataFrame:
    path = os.path.join(run_dir, f"trades_test_H{H}.csv")
    df = safe_read_csv(path)
    if df.empty:
        return df
    return parse_times(df, ["decision_time", "entry_time", "exit_time"])


# ----------------------------
# Equity / drawdown
# ----------------------------
def equity_from_returns(exit_time: pd.Series, rets: np.ndarray) -> pd.DataFrame:
    if len(rets) == 0:
        return pd.DataFrame(columns=["exit_time", "wealth", "drawdown"])

    wealth = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return pd.DataFrame({"exit_time": exit_time, "wealth": wealth, "drawdown": drawdown})


def compute_trade_diagnostics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "gross_total_return": np.nan,
            "net_total_return": np.nan,
            "cost_drag": np.nan,
            "avg_cost": np.nan,
            "tp_exit_rate": np.nan,
            "avg_hold_seconds": np.nan,
            "pnl_p05": np.nan,
            "pnl_p95": np.nan,
        }

    t = trades.sort_values("exit_time").copy()

    net = t["pnl"].to_numpy(dtype="float64")
    net_total = float(np.cumprod(1.0 + net)[-1] - 1.0)

    if "gross_ret" in t.columns:
        gross = t["gross_ret"].to_numpy(dtype="float64")
        gross_total = float(np.cumprod(1.0 + gross)[-1] - 1.0)
    else:
        gross_total = np.nan

    avg_cost = float(np.mean(t["cost"])) if "cost" in t.columns else np.nan
    cost_drag = float(gross_total - net_total) if np.isfinite(gross_total) else np.nan

    tp_exit_rate = float(np.mean(t["exit_reason"] == "take_profit")) if "exit_reason" in t.columns else np.nan
    avg_hold_seconds = float(np.mean(t["hold_seconds"])) if "hold_seconds" in t.columns else np.nan

    pnl_p05 = float(np.nanquantile(net, 0.05)) if len(net) else np.nan
    pnl_p95 = float(np.nanquantile(net, 0.95)) if len(net) else np.nan

    return {
        "gross_total_return": gross_total,
        "net_total_return": net_total,
        "cost_drag": cost_drag,
        "avg_cost": avg_cost,
        "tp_exit_rate": tp_exit_rate,
        "avg_hold_seconds": avg_hold_seconds,
        "pnl_p05": pnl_p05,
        "pnl_p95": pnl_p95,
    }


# ----------------------------
# Plotting (trades)
# ----------------------------
def plot_equity_net_vs_gross(trades: pd.DataFrame, out_path: str, title: str) -> None:
    if trades.empty:
        return

    t = trades.sort_values("exit_time").copy()
    exit_time = t["exit_time"]

    net = t["pnl"].to_numpy(dtype="float64")
    eq_net = equity_from_returns(exit_time, net)

    gross = t["gross_ret"].to_numpy(dtype="float64") if "gross_ret" in t.columns else None
    eq_gross = equity_from_returns(exit_time, gross) if gross is not None else None

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(eq_net["exit_time"], eq_net["wealth"], label="net wealth")
    if eq_gross is not None:
        ax1.plot(eq_gross["exit_time"], eq_gross["wealth"], label="gross wealth")
    ax1.set_title(title)
    ax1.set_xlabel("time")
    ax1.set_ylabel("wealth")
    ax1.legend(loc="best")

    ax2 = ax1.twinx()
    ax2.plot(eq_net["exit_time"], eq_net["drawdown"])
    ax2.set_ylabel("net drawdown")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pnl_hist(trades: pd.DataFrame, out_path: str, title: str) -> None:
    if trades.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(trades["pnl"].to_numpy(dtype="float64"), bins=40)
    ax.set_title(title)
    ax.set_xlabel("pnl per trade (net)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hold_seconds_hist(trades: pd.DataFrame, out_path: str, title: str) -> None:
    if trades.empty or "hold_seconds" not in trades.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(trades["hold_seconds"].to_numpy(dtype="int64"), bins=30)
    ax.set_title(title)
    ax.set_xlabel("hold_seconds")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_exit_reason_bar(trades: pd.DataFrame, out_path: str, title: str) -> None:
    if trades.empty or "exit_reason" not in trades.columns:
        return
    vc = trades["exit_reason"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(vc.index.astype(str), vc.values.astype(int))
    ax.set_title(title)
    ax.set_xlabel("exit_reason")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_intraday_avg_pnl(trades: pd.DataFrame, out_path: str, title: str) -> None:
    if trades.empty or "decision_time" not in trades.columns or "pnl" not in trades.columns:
        return

    t = trades.copy()
    # Use decision_time hour to avoid mixing execution effects
    t["hour"] = pd.to_datetime(t["decision_time"]).dt.hour
    g = t.groupby("hour", as_index=False)["pnl"].mean()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(g["hour"].to_numpy(), g["pnl"].to_numpy(dtype="float64"), marker="o")
    ax.set_title(title)
    ax.set_xlabel("hour (UTC)")
    ax.set_ylabel("avg pnl per trade")
    ax.set_xticks(list(range(0, 24, 2)))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------
# Plotting (grid)
# ----------------------------
def plot_grid_heatmap_H_q(
    grid: pd.DataFrame,
    best_oos: Optional[pd.DataFrame],
    out_path: str,
    title: str,
    value_col: str,
) -> None:
    """
    Heatmap over (q, H), optionally collapsing pt_bps by max(value_col) per (H,q).
    Overlay chosen (H, q_chosen_tune) if best_oos is provided.
    """
    if grid is None or grid.empty or value_col not in grid.columns:
        return

    g = grid.copy()
    if "pt_bps" in g.columns:
        g = g.groupby(["H", "q"], as_index=False)[value_col].max()

    piv = g.pivot(index="q", columns="H", values=value_col).sort_index().sort_index(axis=1)
    arr = piv.to_numpy(dtype="float64")
    cmap, norm = _choose_cmap_and_norm(arr, value_col)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("H (seconds)")
    ax.set_ylabel("q (threshold)")

    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(int(x)) for x in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(x) for x in piv.index])

    # overlay chosen q per H (from best_oos)
    if best_oos is not None and not best_oos.empty:
        for _, r in best_oos.iterrows():
            H = int(r["H"])
            q_ch = _safe_float(r.get("q_chosen_tune", np.nan))
            if not np.isfinite(q_ch):
                continue
            if H not in piv.columns:
                continue
            # find closest q index (exact usually)
            q_vals = piv.index.to_numpy(dtype="float64")
            qi = int(np.argmin(np.abs(q_vals - q_ch)))
            hi = int(np.where(piv.columns.to_numpy(dtype="int64") == H)[0][0])
            ax.scatter([hi], [qi], marker="x")

    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_grid_heatmap_q_pt_for_H(
    grid: pd.DataFrame,
    best_oos_row: Optional[pd.Series],
    H: int,
    out_path: str,
    title: str,
    value_col: str,
) -> None:
    """
    Heatmap over (q, pt_bps) at fixed H.
    Overlay chosen (q, pt_bps) from best_oos_row if provided.
    """
    if grid is None or grid.empty or value_col not in grid.columns:
        return
    if "pt_bps" not in grid.columns:
        return

    g = grid[grid["H"] == int(H)].copy()
    if g.empty:
        return

    piv = g.pivot(index="q", columns="pt_bps", values=value_col).sort_index().sort_index(axis=1)
    arr = piv.to_numpy(dtype="float64")
    cmap, norm = _choose_cmap_and_norm(arr, value_col)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("pt_bps")
    ax.set_ylabel("q")

    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(int(x)) if float(x).is_integer() else str(x) for x in piv.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(x) for x in piv.index])

    # overlay chosen (q, pt) from best_oos
    if best_oos_row is not None:
        q_ch = _safe_float(best_oos_row.get("q_chosen_tune", np.nan))
        pt_ch = _safe_float(best_oos_row.get("pt_bps_chosen_tune", np.nan))
        if np.isfinite(q_ch) and np.isfinite(pt_ch):
            q_vals = piv.index.to_numpy(dtype="float64")
            pt_vals = piv.columns.to_numpy(dtype="float64")
            qi = int(np.argmin(np.abs(q_vals - q_ch)))
            pi = int(np.argmin(np.abs(pt_vals - pt_ch)))
            ax.scatter([pi], [qi], marker="x")

    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_grid_surface_q_pt_for_H(grid: pd.DataFrame, H: int, out_path: str, title: str, value_col: str) -> None:
    if grid is None or grid.empty or value_col not in grid.columns:
        return
    if "pt_bps" not in grid.columns:
        return

    g = grid[grid["H"] == int(H)].copy()
    if g.empty:
        return

    piv = g.pivot(index="q", columns="pt_bps", values=value_col).sort_index().sort_index(axis=1)
    Z = piv.to_numpy(dtype="float64")
    q_vals = piv.index.to_numpy(dtype="float64")
    pt_vals = piv.columns.to_numpy(dtype="float64")
    PT, Q = np.meshgrid(pt_vals, q_vals)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(PT, Q, Z, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("pt_bps")
    ax.set_ylabel("q")
    ax.set_zlabel(value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------
# Plotting (leaderboard / params)
# ----------------------------
def plot_leaderboard_compare(lb: pd.DataFrame, out_dir: str, metric: str, title_prefix: str) -> None:
    if lb.empty or metric not in lb.columns or "run" not in lb.columns or "H" not in lb.columns:
        return

    for H in sorted(lb["H"].unique()):
        d = lb[lb["H"] == H].copy()
        d = d.sort_values(metric, ascending=False)

        fig, ax = plt.subplots(figsize=(11, 0.35 * max(len(d), 6)))
        ax.barh(d["run"].astype(str), d[metric].to_numpy(dtype="float64"))
        ax.invert_yaxis()
        ax.set_title(f"{title_prefix} (H={int(H)}s): {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("run")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"leaderboard_{metric}_H{int(H)}.png"), dpi=180)
        plt.close(fig)


def plot_param_heatmap_run_by_H(best_all: pd.DataFrame, out_path: str, value_col: str, title: str) -> None:
    """
    Heatmap with rows=runs, cols=H, values=value_col (e.g. q_chosen_tune or pt_bps_chosen_tune).
    """
    if best_all.empty or value_col not in best_all.columns:
        return

    piv = best_all.pivot(index="run", columns="H", values=value_col).sort_index().sort_index(axis=1)
    arr = piv.to_numpy(dtype="float64")
    cmap, norm = _choose_cmap_and_norm(arr, value_col)

    fig, ax = plt.subplots(figsize=(10, 0.35 * max(len(piv.index), 6)))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, origin="upper", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("H")
    ax.set_ylabel("run")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(int(x)) for x in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(x) for x in piv.index])

    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_topk_equity_overlay(
    runs_root: str,
    lb: pd.DataFrame,
    out_path: str,
    H: int,
    metric: str,
    top_k: int = 5,
) -> None:
    """
    Overlay net wealth curves for top-K runs by metric for a given H.
    """
    if lb.empty or metric not in lb.columns:
        return

    d = lb[lb["H"] == int(H)].copy()
    d = d.sort_values(metric, ascending=False).head(int(top_k))
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for _, r in d.iterrows():
        run_name = str(r["run"])
        run_dir = os.path.join(runs_root, run_name)
        trades = load_trades(run_dir, int(H))
        if trades.empty:
            continue
        t = trades.sort_values("exit_time").copy()
        eq = equity_from_returns(t["exit_time"], t["pnl"].to_numpy(dtype="float64"))
        ax.plot(eq["exit_time"], eq["wealth"], label=run_name)

    ax.set_title(f"Top-{top_k} equity overlay (H={int(H)}s) by {metric}")
    ax.set_xlabel("time")
    ax.set_ylabel("net wealth")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------
# Optional: signal correlation (SVI vs OFI narrative)
# ----------------------------
def compute_and_plot_signal_correlation(
    features_path: str,
    out_tables_dir: str,
    out_fig_dir: str,
    cols: List[str],
    sample_n: int = 20000,
    seed: int = 7,
) -> None:
    if not features_path or not os.path.exists(features_path):
        return

    df = pd.read_parquet(features_path)
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return

    sub = df[cols].astype("float64").dropna()
    if sub.empty:
        return

    corr = sub.corr()
    ensure_dir(out_tables_dir)
    corr.to_csv(os.path.join(out_tables_dir, "signal_correlation.csv"), index=True)

    # heatmap
    arr = corr.to_numpy(dtype="float64")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, aspect="auto", cmap=plt.get_cmap("RdYlGn"), norm=TwoSlopeNorm(vmin=float(np.min(arr)), vcenter=0.0, vmax=float(np.max(arr))))
    ax.set_title("Signal correlation (features_1s)")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax, label="corr")
    fig.tight_layout()
    ensure_dir(out_fig_dir)
    fig.savefig(os.path.join(out_fig_dir, "signal_correlation_heatmap.png"), dpi=180)
    plt.close(fig)

    # scatter for first two cols
    rng = np.random.default_rng(seed)
    if len(sub) > sample_n:
        idx = rng.choice(len(sub), size=sample_n, replace=False)
        s = sub.iloc[idx]
    else:
        s = sub

    xcol, ycol = cols[0], cols[1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(s[xcol].to_numpy(), s[ycol].to_numpy(), s=4, alpha=0.3)
    ax.set_title(f"Scatter: {xcol} vs {ycol} (corr={float(corr.loc[xcol, ycol]):.4f})")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig_dir, f"signal_scatter_{xcol}_vs_{ycol}.png"), dpi=180)
    plt.close(fig)


# ----------------------------
# Main pipeline
# ----------------------------
def build_leaderboard(runs_root: str, out_tables_dir: str) -> pd.DataFrame:
    run_dirs = list_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No runs found: expected {runs_root}/<run>/best_oos.csv")

    rows = []
    for rd in run_dirs:
        run_name = os.path.basename(rd)
        best = load_best_oos(rd)

        for _, r in best.iterrows():
            H = int(r["H"])
            trades = load_trades(rd, H)
            diag = compute_trade_diagnostics(trades)

            row = dict(r)
            row["run"] = run_name
            row.update(diag)

            # alpha vs baseline (most ROI-per-line metric)
            b_avg = _safe_float(row.get("baseline_avg_pnl_mean", np.nan))
            b_sh = _safe_float(row.get("baseline_sharpe_mean", np.nan))
            b_tr = _safe_float(row.get("baseline_total_return_mean", np.nan))

            row["alpha_avg_pnl"] = _safe_float(row.get("avg_pnl", np.nan)) - b_avg if np.isfinite(b_avg) else np.nan
            row["alpha_sharpe"] = _safe_float(row.get("sharpe_trades", np.nan)) - b_sh if np.isfinite(b_sh) else np.nan

            net_tr = _safe_float(row.get("net_total_return", np.nan))
            if not np.isfinite(net_tr):
                net_tr = _safe_float(row.get("total_return", np.nan))
            row["alpha_total_return"] = net_tr - b_tr if (np.isfinite(net_tr) and np.isfinite(b_tr)) else np.nan

            rows.append(row)

    lb = pd.DataFrame(rows).sort_values(["H", "run"]).reset_index(drop=True)

    ensure_dir(out_tables_dir)
    out_all = os.path.join(out_tables_dir, "summary_best_oos_all_runs.csv")
    lb.to_csv(out_all, index=False)

    compact_cols = [
        "run", "H",
        "signal_col", "signal_weights", "side_mode",
        "intensity_gate", "aux_gate", "aux_gate_col",
        "q_chosen_tune", "pt_bps_chosen_tune",
        "n_trades", "avg_pnl", "median_pnl", "win_rate",
        "gross_total_return", "net_total_return", "cost_drag",
        "max_drawdown", "sharpe_trades",
        "tp_exit_rate", "avg_hold_seconds",
        "pnl_p05", "pnl_p95",
        "baseline_avg_pnl_mean", "baseline_total_return_mean", "baseline_sharpe_mean",
        "alpha_avg_pnl", "alpha_total_return", "alpha_sharpe",
    ]
    compact_cols = [c for c in compact_cols if c in lb.columns]
    out_compact = os.path.join(out_tables_dir, "leaderboard_compact.csv")
    lb[compact_cols].to_csv(out_compact, index=False)

    print(f"[OK] Saved leaderboard -> {out_all}")
    print(f"[OK] Saved compact leaderboard -> {out_compact}")
    return lb


def generate_figures_for_run(
    run_dir: str,
    out_fig_dir: str,
    out_tables_dir: str,
    value_col: str,
    plot_3d: bool,
) -> None:
    run_name = os.path.basename(run_dir)
    best = load_best_oos(run_dir)

    grid = load_grid_tune(run_dir)
    if grid is not None and not grid.empty:
        ensure_dir(out_tables_dir)
        grid_out = os.path.join(out_tables_dir, f"{run_name}_grid_tune.csv")
        grid.to_csv(grid_out, index=False)

        ensure_dir(out_fig_dir)
        heat_path = os.path.join(out_fig_dir, f"{run_name}_grid_Hxq_{value_col}.png")
        plot_grid_heatmap_H_q(
            grid, best_oos=best, out_path=heat_path,
            title=f"{run_name}: grid (H x q) [{value_col}]",
            value_col=value_col,
        )

        if "pt_bps" in grid.columns:
            for H in sorted(grid["H"].unique()):
                best_row = best[best["H"] == int(H)].iloc[0] if (not best.empty and (best["H"] == int(H)).any()) else None

                out2 = os.path.join(out_fig_dir, f"{run_name}_grid_qxpt_H{int(H)}_{value_col}.png")
                plot_grid_heatmap_q_pt_for_H(
                    grid, best_oos_row=best_row, H=int(H), out_path=out2,
                    title=f"{run_name}: grid (q x pt) H={int(H)} [{value_col}]",
                    value_col=value_col,
                )

                if plot_3d:
                    out3 = os.path.join(out_fig_dir, f"{run_name}_grid_qxpt_3d_H{int(H)}_{value_col}.png")
                    plot_grid_surface_q_pt_for_H(
                        grid, int(H), out3,
                        title=f"{run_name}: surface (q x pt) H={int(H)} [{value_col}]",
                        value_col=value_col,
                    )

    # per-H figures
    for H in sorted(best["H"].unique()):
        trades = load_trades(run_dir, int(H))
        if trades.empty:
            continue

        ensure_dir(out_fig_dir)
        eq_path = os.path.join(out_fig_dir, f"{run_name}_equity_net_vs_gross_H{int(H)}.png")
        plot_equity_net_vs_gross(trades, eq_path, title=f"{run_name}: net vs gross equity (H={int(H)}s)")

        hist_path = os.path.join(out_fig_dir, f"{run_name}_pnl_hist_H{int(H)}.png")
        plot_pnl_hist(trades, hist_path, title=f"{run_name}: pnl distribution (H={int(H)}s)")

        hold_path = os.path.join(out_fig_dir, f"{run_name}_hold_seconds_hist_H{int(H)}.png")
        plot_hold_seconds_hist(trades, hold_path, title=f"{run_name}: holding time (H={int(H)}s)")

        er_path = os.path.join(out_fig_dir, f"{run_name}_exit_reason_H{int(H)}.png")
        plot_exit_reason_bar(trades, er_path, title=f"{run_name}: exit reasons (H={int(H)}s)")

        intraday_path = os.path.join(out_fig_dir, f"{run_name}_intraday_avg_pnl_H{int(H)}.png")
        plot_intraday_avg_pnl(trades, intraday_path, title=f"{run_name}: intraday avg pnl (H={int(H)}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="reports/tables")
    ap.add_argument("--out_fig_dir", default="reports/figures")
    ap.add_argument("--out_tables_dir", default="reports/tables")
    ap.add_argument("--run", default=None)

    ap.add_argument("--value_col", default="alpha_avg_pnl", help="Metric to visualize in grids (e.g. avg_pnl, sharpe_trades, total_return)")
    ap.add_argument("--plot_3d", action="store_true", help="Also render 3D surfaces for q x pt per H (adds, does not replace 2D).")

    # ROI knobs
    ap.add_argument("--top_k_overlay", type=int, default=5, help="Top-K runs to overlay equity curves per H.")
    ap.add_argument("--overlay_metric", type=str, default="alpha_avg_pnl", help="Metric to rank runs for equity overlays.")

    # optional: compute signal correlation directly from features parquet (SVI vs OFI narrative)
    ap.add_argument("--features", default=None, help="Optional features parquet to compute signal correlation plots/tables.")
    ap.add_argument(
        "--corr_cols",
        default="z_svi_60_600_lag1,z_ofi_60_600_lag1,z_cfi_60_600_lag1,z_cfi_ratio_60_600_lag1,z_max_share_60_600_lag1",
        help="Comma-separated columns to include in correlation analysis (if --features is set).",
    )
    args = ap.parse_args()

    ensure_dir(args.out_fig_dir)
    ensure_dir(args.out_tables_dir)

    if args.run is None:
        lb = build_leaderboard(args.runs_root, args.out_tables_dir)

        # Leaderboard comparisons (incl. alpha)
        ensure_dir(args.out_fig_dir)
        for m in ["alpha_avg_pnl", "alpha_total_return", "alpha_sharpe", "net_total_return", "avg_pnl", "sharpe_trades", "max_drawdown"]:
            if m in lb.columns:
                plot_leaderboard_compare(lb, args.out_fig_dir, metric=m, title_prefix="Leaderboard")

        # Parameter heatmaps: what got chosen
        if not lb.empty:
            best_all = lb.copy()
            if "q_chosen_tune" in best_all.columns:
                plot_param_heatmap_run_by_H(
                    best_all,
                    out_path=os.path.join(args.out_fig_dir, "params_heatmap_q_chosen.png"),
                    value_col="q_chosen_tune",
                    title="Chosen q by run and H (TEST selection)",
                )
            if "pt_bps_chosen_tune" in best_all.columns:
                plot_param_heatmap_run_by_H(
                    best_all,
                    out_path=os.path.join(args.out_fig_dir, "params_heatmap_pt_bps_chosen.png"),
                    value_col="pt_bps_chosen_tune",
                    title="Chosen pt_bps by run and H (TEST selection)",
                )

        # Equity overlays for top-K
        if args.top_k_overlay and args.top_k_overlay > 0 and (args.overlay_metric in lb.columns):
            for H in sorted(lb["H"].unique()):
                outp = os.path.join(args.out_fig_dir, f"overlay_top{int(args.top_k_overlay)}_{args.overlay_metric}_H{int(H)}.png")
                plot_topk_equity_overlay(
                    runs_root=args.runs_root,
                    lb=lb,
                    out_path=outp,
                    H=int(H),
                    metric=str(args.overlay_metric),
                    top_k=int(args.top_k_overlay),
                )

        # Per-run figures
        run_dirs = list_run_dirs(args.runs_root)
        for rd in run_dirs:
            generate_figures_for_run(
                rd, args.out_fig_dir, args.out_tables_dir,
                value_col=str(args.value_col),
                plot_3d=bool(args.plot_3d),
            )

        # Optional signal correlation (SVI vs OFI etc.)
        if args.features is not None:
            cols = [c.strip() for c in str(args.corr_cols).split(",") if c.strip()]
            compute_and_plot_signal_correlation(
                features_path=str(args.features),
                out_tables_dir=args.out_tables_dir,
                out_fig_dir=args.out_fig_dir,
                cols=cols,
            )

        print(f"[OK] Generated figures for {len(run_dirs)} runs -> {args.out_fig_dir}")

    else:
        rd = os.path.join(args.runs_root, args.run)
        if not os.path.isdir(rd):
            raise FileNotFoundError(f"Run dir not found: {rd}")

        generate_figures_for_run(rd, args.out_fig_dir, args.out_tables_dir, value_col=str(args.value_col), plot_3d=bool(args.plot_3d))
        print(f"[OK] Generated figures for run={args.run} -> {args.out_fig_dir}")


if __name__ == "__main__":
    main()
