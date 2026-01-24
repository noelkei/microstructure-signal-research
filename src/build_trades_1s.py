# src/build_trades_1s.py
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


CANON_COLS = [
    "agg_trade_id",
    "price",
    "qty",
    "first_trade_id",
    "last_trade_id",
    "trade_time_ms",
    "is_buyer_maker",
]

ALT_MAP = {
    "a": "agg_trade_id",
    "p": "price",
    "q": "qty",
    "f": "first_trade_id",
    "l": "last_trade_id",
    "T": "trade_time_ms",
    "m": "is_buyer_maker",
    "M": "is_best_match",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)

    if set(CANON_COLS).issubset(cols):
        return df[CANON_COLS].copy()

    if {"a", "p", "q", "f", "l", "T", "m"}.issubset(cols):
        out = df.rename(columns=ALT_MAP).copy()
        return out[CANON_COLS].copy()

    raise ValueError(f"Unsupported aggTrades parquet schema. Columns found: {sorted(df.columns)}")


def load_raw_parquets(inputs: List[str]) -> pd.DataFrame:
    dfs = []
    for p in inputs:
        df = pd.read_parquet(p)
        df = _canonicalize_columns(df)

        df["agg_trade_id"] = df["agg_trade_id"].astype("int64")
        df["first_trade_id"] = df["first_trade_id"].astype("int64")
        df["last_trade_id"] = df["last_trade_id"].astype("int64")
        df["trade_time_ms"] = df["trade_time_ms"].astype("int64")

        df["price"] = df["price"].astype("float64")
        df["qty"] = df["qty"].astype("float64")

        df["is_buyer_maker"] = df["is_buyer_maker"].astype("bool")

        dfs.append(df)

    out = pd.concat(dfs, axis=0, ignore_index=True)
    out = out.sort_values(["trade_time_ms", "agg_trade_id"])
    out = out.drop_duplicates(subset=["agg_trade_id"], keep="last").reset_index(drop=True)
    return out


def validate_time_index(idx: pd.DatetimeIndex) -> Tuple[int, float]:
    if not idx.is_monotonic_increasing:
        raise RuntimeError("Index is not monotonic increasing.")
    if idx.has_duplicates:
        raise RuntimeError("Index has duplicates.")

    full = pd.date_range(start=idx[0], end=idx[-1], freq="1s", tz="UTC")
    missing = full.difference(idx)
    miss_n = len(missing)
    miss_frac = miss_n / len(full) if len(full) else 0.0
    return miss_n, miss_frac


def build_trades_1s(raw: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    df = raw.copy()

    ts = pd.to_datetime(df["trade_time_ms"], unit="ms", utc=True).dt.floor("s")
    df["ts"] = ts

    is_buy_taker = ~df["is_buyer_maker"]
    is_sell_taker = df["is_buyer_maker"]

    df["quote"] = df["price"] * df["qty"]
    df["taker_buy_qty"] = np.where(is_buy_taker, df["qty"].to_numpy(dtype="float64"), 0.0)
    df["taker_sell_qty"] = np.where(is_sell_taker, df["qty"].to_numpy(dtype="float64"), 0.0)

    g = df.groupby("ts", sort=True)

    base_out = pd.DataFrame(
        {
            "n_aggtrades_1s": g["agg_trade_id"].count().astype("int64"),
            "base_qty_1s": g["qty"].sum().astype("float64"),
            "quote_qty_1s": g["quote"].sum().astype("float64"),
            "taker_buy_base_1s": g["taker_buy_qty"].sum().astype("float64"),
            "taker_sell_base_1s": g["taker_sell_qty"].sum().astype("float64"),
        }
    )

    buy_count = df.loc[is_buy_taker].groupby("ts").size().rename("buy_count_1s")
    sell_count = df.loc[is_sell_taker].groupby("ts").size().rename("sell_count_1s")
    max_qty = g["qty"].max().rename("max_qty_1s")

    out = pd.concat([base_out, buy_count, sell_count, max_qty], axis=1)

    out["buy_count_1s"] = out["buy_count_1s"].fillna(0).astype("int64")
    out["sell_count_1s"] = out["sell_count_1s"].fillna(0).astype("int64")
    out["max_qty_1s"] = out["max_qty_1s"].fillna(0.0).astype("float64")

    out["ofi_base_1s"] = out["taker_buy_base_1s"] - out["taker_sell_base_1s"]
    out["ofi_ratio_1s"] = out["ofi_base_1s"] / np.maximum(out["base_qty_1s"].to_numpy(dtype="float64"), eps)

    out["ofi_count_1s"] = (out["buy_count_1s"] - out["sell_count_1s"]).astype("int64")
    out["mean_qty_1s"] = out["base_qty_1s"] / np.maximum(out["n_aggtrades_1s"].to_numpy(dtype="float64"), 1.0)
    out["max_share_1s"] = out["max_qty_1s"] / np.maximum(out["base_qty_1s"].to_numpy(dtype="float64"), eps)

    full = pd.date_range(start=out.index[0], end=out.index[-1], freq="1s", tz="UTC")
    out = out.reindex(full)

    float_cols = [
        "base_qty_1s",
        "quote_qty_1s",
        "taker_buy_base_1s",
        "taker_sell_base_1s",
        "ofi_base_1s",
        "ofi_ratio_1s",
        "max_qty_1s",
        "mean_qty_1s",
        "max_share_1s",
    ]
    for c in float_cols:
        out[c] = out[c].fillna(0.0).astype("float64")

    int_cols = ["n_aggtrades_1s", "buy_count_1s", "sell_count_1s", "ofi_count_1s"]
    for c in int_cols:
        out[c] = out[c].fillna(0).astype("int64")

    out["is_imputed_trades"] = out["n_aggtrades_1s"].eq(0)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="List of raw aggtrades parquet files. If omitted, loads all aggtrades_*.parquet from --raw_dir.",
    )
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/trades_1s.parquet")
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    if args.inputs and len(args.inputs) > 0:
        inputs = args.inputs
    else:
        pattern = os.path.join(args.raw_dir, "aggtrades_*.parquet")
        inputs = sorted(glob.glob(pattern))
        if not inputs:
            raise FileNotFoundError(f"No aggTrades parquet files found at {pattern}")

    raw = load_raw_parquets(inputs)
    trades_1s = build_trades_1s(raw, eps=float(args.eps))

    miss_n, miss_frac = validate_time_index(trades_1s.index)
    print(f"[VALIDATION] Missing seconds (relative to full trade grid): {miss_n:,} ({miss_frac:.3%})")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    trades_1s.to_parquet(args.out)
    print(f"[OK] Saved trades_1s -> {args.out} rows={len(trades_1s):,} cols={trades_1s.shape[1]}")


if __name__ == "__main__":
    main()
