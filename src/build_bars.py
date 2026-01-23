# src/build_bars.py
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import pandas as pd


# Columns we expect from fetch_binance raw parquet
RAW_COLS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote"
]


def load_raw_parquets(inputs: List[str]) -> pd.DataFrame:
    """
    Load one or more raw klines parquet files and concatenate them into one DataFrame.
    Deduplicate by open_time_ms (the kline open timestamp).
    """
    dfs = []
    for p in inputs:
        df = pd.read_parquet(p)

        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")

        # Keep only canonical columns (and in a fixed order)
        dfs.append(df[RAW_COLS].copy())

    out = pd.concat(dfs, axis=0, ignore_index=True)

    # Sort by time and dedupe border overlaps (chunks include the boundary second)
    out = out.sort_values("open_time_ms").drop_duplicates(subset=["open_time_ms"], keep="last")
    out = out.reset_index(drop=True)
    return out


def validate_time_index(idx: pd.DatetimeIndex) -> Tuple[int, float]:
    """
    Validate that the datetime index is:
      - monotonic increasing
      - has no duplicates
    Then compute missing seconds relative to a full 1-second grid.
    Returns (missing_count, missing_fraction).
    """
    if not idx.is_monotonic_increasing:
        raise RuntimeError("Index is not monotonic increasing.")
    if idx.has_duplicates:
        raise RuntimeError("Index has duplicates after de-duplication step.")

    full = pd.date_range(start=idx[0], end=idx[-1], freq="1s", tz="UTC")
    missing = full.difference(idx)
    miss_n = len(missing)
    miss_frac = miss_n / len(full) if len(full) else 0.0
    return miss_n, miss_frac


def impute_to_full_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to a full 1-second grid and impute missing seconds with a documented policy:
      - Price columns (open/high/low/close): forward-fill using last known close.
        For fully missing seconds, set OHLC = ffilled close.
      - Volume-like columns: fill 0.0
      - n_trades: fill 0
      - Add boolean column is_imputed=True where we filled a missing second.

    This produces a clean "canonical 1s bars" dataset that makes rolling windows stable.
    """
    idx = bars.index
    full = pd.date_range(start=idx[0], end=idx[-1], freq="1s", tz="UTC")

    bars2 = bars.reindex(full)

    # Identify which seconds were missing (and thus imputed)
    is_imputed = bars2["close"].isna()
    bars2["is_imputed"] = is_imputed

    # Forward-fill close; for imputed seconds, set OHLC to that filled close
    bars2["close"] = bars2["close"].ffill()

    for c in ["open", "high", "low"]:
        # For imputed rows, set to close. Then ffill any remaining NaNs.
        bars2[c] = bars2[c].where(~is_imputed, bars2["close"])
        bars2[c] = bars2[c].ffill()

    # For volumes, missing means "no trades in that second" under this policy
    for c in ["volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        bars2[c] = bars2[c].fillna(0.0)

    # Trades count to int
    bars2["n_trades"] = bars2["n_trades"].fillna(0).astype("int64")

    return bars2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="List of raw parquet files. If omitted, loads all klines_*.parquet from --raw_dir."
    )
    ap.add_argument(
        "--raw_dir",
        default="data/raw",
        help="If --inputs not provided, load all klines_*.parquet from this directory."
    )
    ap.add_argument(
        "--out",
        default="data/processed/bars_1s.parquet",
        help="Output parquet path for canonical 1-second bars."
    )
    ap.add_argument(
        "--no_impute",
        action="store_true",
        help="Keep gaps (do not reindex/impute to a full 1-second grid)."
    )
    args = ap.parse_args()

    # Resolve input files
    if args.inputs and len(args.inputs) > 0:
        inputs = args.inputs
    else:
        pattern = os.path.join(args.raw_dir, "klines_*.parquet")
        inputs = sorted(glob.glob(pattern))
        if not inputs:
            raise FileNotFoundError(f"No raw parquet files found at {pattern}")

    raw = load_raw_parquets(inputs)

    # Build canonical bars DataFrame
    bars = raw.copy()
    bars["ts"] = pd.to_datetime(bars["open_time_ms"], unit="ms", utc=True)
    bars = bars.set_index("ts").drop(columns=["open_time_ms", "close_time_ms"])

    # Enforce numeric dtypes
    float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for c in float_cols:
        bars[c] = bars[c].astype("float64")
    bars["n_trades"] = bars["n_trades"].astype("int64")

    # Validate original index and compute missing seconds rate
    miss_n, miss_frac = validate_time_index(bars.index)
    print(f"[VALIDATION] Missing seconds: {miss_n:,} ({miss_frac:.3%})")

    # Optional: impute to full 1-second grid (recommended for stable rolling windows)
    if not args.no_impute:
        bars = impute_to_full_grid(bars)
        miss_n2, miss_frac2 = validate_time_index(bars.index)
        print(f"[AFTER IMPUTE] Missing seconds: {miss_n2:,} ({miss_frac2:.3%})")
    else:
        bars["is_imputed"] = False

    # Save output
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bars.to_parquet(args.out)
    print(f"[OK] Saved bars -> {args.out} rows={len(bars):,} cols={bars.shape[1]}")


if __name__ == "__main__":
    main()
