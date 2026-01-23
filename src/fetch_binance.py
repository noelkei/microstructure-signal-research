# src/fetch_binance.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"


def iso_to_dt_utc(iso_str: str) -> datetime:
    """
    Parse ISO8601 into aware UTC datetime.
    Accepts 'Z' suffix.
    """
    s = iso_str.strip() #remove spaces before and after
    if s.endswith("Z"): #ending with "Z" in iso means "+00:00"
        s = s[:-1] + "+00:00" #remove "Z" and add "+00:00" for utc
    dt = datetime.fromisoformat(s) #transform into format dt for pandas
    if dt.tzinfo is None: #no "Z"
        # assume UTC if naive
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc) #transform to utc


def dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:  # no "Z"
        # assume UTC if naive
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) #transform to milliseconds


@dataclass
class FetchMeta:
    endpoint: str
    base_url: str
    symbol: str
    interval: str
    start_iso: str
    end_iso: str
    start_ms: int
    end_ms: int
    limit: int
    sleep_ms: int
    fetched_at_utc: str
    n_rows: int
    n_requests: int
    notes: str


class BinanceHTTPError(RuntimeError):
    pass


def robust_get(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    timeout_s: float = 10.0,
    max_retries: int = 8,
    base_backoff_s: float = 0.5,
) -> requests.Response:
    """
    Robust GET:
      - retries for 5xx
      - backoff on 429 (uses Retry-After if present)
      - stop on 418
    """
    backoff = base_backoff_s
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout_s)

            if resp.status_code == 200:
                return resp

            if resp.status_code == 418:
                raise BinanceHTTPError(
                    f"418 IP banned risk. Stop immediately. Response: {resp.text[:300]}"
                )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after is not None else backoff
                time.sleep(max(0.0, sleep_s))
                backoff = min(backoff * 2.0, 30.0)
                continue

            if resp.status_code in (500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
                continue

            # other errors: surface
            raise BinanceHTTPError(
                f"HTTP {resp.status_code}. params={params}. body={resp.text[:500]}"
            )

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    raise BinanceHTTPError(f"Failed after {max_retries} attempts. last_err={last_err}")


def fetch_klines_1s(
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = "1s",
    limit: int = 1000,
    sleep_ms: int = 250,
    session: Optional[requests.Session] = None,
    show_progress: bool = True,
) -> Tuple[List[List[Any]], int]:
    """
    Fetch klines [start_ms, end_ms] inclusive-ish using pagination.
    Pagination step: next startTime = last_open_time + 1000ms for 1s interval.

    Returns:
      rows: raw kline rows (list of lists)
      n_requests: number of HTTP requests
    """
    if interval != "1s":
        raise ValueError("This project starter supports interval=1s only (by design).")

    sess = session or requests.Session()

    url = BASE_URL + KLINES_ENDPOINT
    cur = start_ms
    rows: List[List[Any]] = []
    n_requests = 0

    # Rough expected rows for progress bar (not exact due to gaps)
    expected = max(1, int((end_ms - start_ms) / 1000) + 1)
    pbar = tqdm(total=expected, disable=not show_progress, desc="Fetching klines 1s")

    last_open: Optional[int] = None

    while cur <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = robust_get(sess, url, params=params)
        n_requests += 1

        batch = resp.json()
        if not batch:
            break

        # Validate chronological within batch
        open_times = [int(x[0]) for x in batch]
        if open_times != sorted(open_times):
            raise RuntimeError("Non-monotonic batch returned (unexpected).")

        # Prevent infinite loop (if API returns same last open time)
        if last_open is not None and open_times[-1] <= last_open:
            raise RuntimeError("Pagination stalled (last_open not increasing).")

        rows.extend(batch)
        last_open = open_times[-1]

        # Update progress (approx)
        pbar.update(len(batch))

        # Next page starts after the last open time (1s = 1000ms)
        cur = open_times[-1] + 1000

        time.sleep(max(0.0, sleep_ms) / 1000.0)

        # Safety: if we got less than limit, likely done
        if len(batch) < limit:
            break

    pbar.close()
    return rows, n_requests


def raw_rows_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    """
    Convert Binance kline rows to typed DataFrame.
    Schema:
      [ open_time, open, high, low, close, volume,
        close_time, quote_volume, n_trades,
        taker_buy_base, taker_buy_quote, ignore ]
    """
    cols = [
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Cast types
    df["open_time_ms"] = df["open_time_ms"].astype("int64")
    df["close_time_ms"] = df["close_time_ms"].astype("int64")
    df["n_trades"] = df["n_trades"].astype("int64")

    float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for c in float_cols:
        df[c] = df[c].astype("float64")

    return df.drop(columns=["ignore"], errors="ignore")


def make_output_paths(out_dir: str, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    start_tag = start_dt.strftime("%Y%m%dT%H%M%SZ")
    end_tag = end_dt.strftime("%Y%m%dT%H%M%SZ")
    base = f"klines_{symbol}_{interval}_{start_tag}_{end_tag}"
    parquet_path = os.path.join(out_dir, base + ".parquet")
    meta_path = os.path.join(out_dir, base + ".meta.json")
    return parquet_path, meta_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1s")
    ap.add_argument("--start", type=str, default=None, help="ISO8601 UTC e.g. 2026-01-22T00:00:00Z")
    ap.add_argument("--hours", type=float, default=None, help="Hours from start")
    ap.add_argument("--minutes", type=int, default=None, help="Minutes from start")
    ap.add_argument("--out_dir", type=str, default="data/raw")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sleep_ms", type=int, default=250)
    ap.add_argument("--force", action="store_true", help="Overwrite if exists")
    ap.add_argument("--smoke", action="store_true", help="Fetch 10 minutes from --start (or now-10m if start not provided)")
    args = ap.parse_args()

    if args.smoke:
        if args.start is None:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(minutes=10)
        else:
            start_dt = iso_to_dt_utc(args.start)
            end_dt = start_dt + timedelta(minutes=10)
    else:
        if args.start is None:
            raise ValueError("--start is required unless --smoke is used.")
        start_dt = iso_to_dt_utc(args.start)
        if args.hours is None and args.minutes is None:
            raise ValueError("Provide either --hours or --minutes (unless --smoke).")
        dur = timedelta(hours=float(args.hours or 0.0), minutes=int(args.minutes or 0))
        end_dt = start_dt + dur

    start_ms = dt_to_ms(start_dt)
    end_ms = dt_to_ms(end_dt)

    parquet_path, meta_path = make_output_paths(args.out_dir, args.symbol, args.interval, start_dt, end_dt)

    if (os.path.exists(parquet_path) or os.path.exists(meta_path)) and not args.force:
        print(f"[SKIP] Output exists: {parquet_path}")
        print("Use --force to overwrite.")
        return

    with requests.Session() as sess:
        rows, n_requests = fetch_klines_1s(
            symbol=args.symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            interval=args.interval,
            limit=args.limit,
            sleep_ms=args.sleep_ms,
            session=sess,
            show_progress=True,
        )

    df = raw_rows_to_df(rows)

    # De-dupe safety
    df = df.sort_values("open_time_ms").drop_duplicates(subset=["open_time_ms"], keep="last").reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    meta = FetchMeta(
        endpoint=KLINES_ENDPOINT,
        base_url=BASE_URL,
        symbol=args.symbol,
        interval=args.interval,
        start_iso=start_dt.isoformat().replace("+00:00", "Z"),
        end_iso=end_dt.isoformat().replace("+00:00", "Z"),
        start_ms=start_ms,
        end_ms=end_ms,
        limit=args.limit,
        sleep_ms=args.sleep_ms,
        fetched_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        n_rows=int(df.shape[0]),
        n_requests=int(n_requests),
        notes="Public Binance Spot REST klines. No L2 historical replay.",
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"[OK] Saved {df.shape[0]:,} rows -> {parquet_path}")
    print(f"[OK] Meta -> {meta_path}")


if __name__ == "__main__":
    main()
