# src/build_trades_1s.py
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


RAW_COLS = [
    "agg_trade_id",
    "price",
    "qty",
    "first_trade_id",
    "last_trade_id",
    "trade_time_ms",
    "is_buyer_maker",
    "is_best_match",
]


def load_raw_parquets(inputs: List[str]) -> pd.DataFrame:
    dfs = []
    for p in inputs:
        df = pd.read_parquet(p)
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")
        dfs.append(df[RAW_COLS].copy())

    out = pd.concat(dfs, axis=0, ignore_index=True)

    out = out.sort_values(["trade_time_ms", "agg_trade_id"]).drop_duplicates(subset=["agg_trade_id"], keep="last")
    out = out.reset_index(drop=True)
    return out


def validate_time_index(idx: pd.DatetimeIndex) -> Tuple[int, float]:
    if not idx.is_monotonic_increasing:
        raise RuntimeError("Index is not monotonic increasing.")
    if idx.has_duplicates:
        raise RuntimeError("Index has duplicates after aggregation.")

    full = pd.date_range(start=idx[0], end=idx[-1], freq="1s", tz="UTC")
    missing = full.difference(idx)
    miss_n = len(missing)
    miss_frac = miss_n / len(full) if len(full) else 0.0
    return miss_n, miss_frac


def impute_to_full_grid(trades_1s: pd.DataFrame) -> pd.DataFrame:
    idx = trades_1s.index
    full = pd.date_range(start=idx[0], end=idx[-1], freq="1s", tz="UTC")
    out = trades_1s.reindex(full)

    is_imputed = out["n_aggs_1s"].isna()
    out["is_imputed_trades"] = is_imputed

    fill0 = ["buy_qty_1s", "sell_qty_1s", "ofi_1s", "qty_1s", "n_aggs_1s", "buy_count_1s", "sell_count_1s", "ofi_count_1s"]
    for c in fill0:
        out[c] = out[c].fillna(0.0)

    out["n_aggs_1s"] = out["n_aggs_1s"].astype("int64")
    out["buy_count_1s"] = out["buy_count_1s"].astype("int64")
    out["sell_count_1s"] = out["sell_count_1s"].astype("int64")
    out["ofi_count_1s"] = out["ofi_count_1s"].astype("int64")

    return out


def build_trades_1s(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("No aggTrades rows loaded.")

    ts = pd.to_datetime(raw["trade_time_ms"].astype("int64"), unit="ms", utc=True).dt.floor("1s")
    raw = raw.copy()
    raw["ts_1s"] = ts

    buy_mask = ~raw["is_buyer_maker"].astype("bool")
    sell_mask = raw["is_buyer_maker"].astype("bool")

    g = raw.groupby("ts_1s", sort=True)

    qty = raw["qty"].astype("float64")
    price = raw["price"].astype("float64")

    buy_qty_1s = g.apply(lambda x: float(x.loc[~x["is_buyer_maker"], "qty"].sum()))
    sell_qty_1s = g.apply(lambda x: float(x.loc[x["is_buyer_maker"], "qty"].sum()))

    n_aggs_1s = g.size().astype("int64")
    buy_count_1s = g.apply(lambda x: int((~x["is_buyer_maker"]).sum())).astype("int64")
    sell_count_1s = g.apply(lambda x: int((x["is_buyer_maker"]).sum())).astype("int64")

    qty_1s = g["qty"].sum().astype("float64")

    out = pd.DataFrame(
        {
            "buy_qty_1s": buy_qty_1s,
            "sell_qty_1s": sell_qty_1s,
            "qty_1s": qty_1s,
            "n_aggs_1s": n_aggs_1s,
            "buy_count_1s": buy_count_1s,
            "sell_count_1s": sell_count_1s,
        }
    ).sort_index()

    out["ofi_1s"] = out["buy_qty_1s"] - out["sell_qty_1s"]
    out["ofi_count_1s"] = out["buy_count_1s"] - out["sell_count_1s"]

    out.index.name = "ts"
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
    ap.add_argument("--no_impute", action="store_true", help="Keep gaps (do not reindex/impute to full 1s grid).")
    args = ap.parse_args()

    if args.inputs and len(args.inputs) > 0:
        inputs = args.inputs
    else:
        pattern = os.path.join(args.raw_dir, "aggtrades_*.parquet")
        inputs = sorted(glob.glob(pattern))
        if not inputs:
            raise FileNotFoundError(f"No raw parquet files found at {pattern}")

    raw = load_raw_parquets(inputs)
    trades_1s = build_trades_1s(raw)

    miss_n, miss_frac = validate_time_index(trades_1s.index)
    print(f"[VALIDATION] Missing seconds (trades_1s): {miss_n:,} ({miss_frac:.3%})")

    if not args.no_impute:
        trades_1s = impute_to_full_grid(trades_1s)
        miss_n2, miss_frac2 = validate_time_index(trades_1s.index)
        print(f"[AFTER IMPUTE] Missing seconds (trades_1s): {miss_n2:,} ({miss_frac2:.3%})")
    else:
        trades_1s["is_imputed_trades"] = False

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    trades_1s.to_parquet(args.out)
    print(f"[OK] Saved trades_1s -> {args.out} rows={len(trades_1s):,} cols={trades_1s.shape[1]}")


if __name__ == "__main__":
    main()
