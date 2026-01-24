# src/fetch_aggtrades.py
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
AGGTRADES_ENDPOINT = "/api/v3/aggTrades"


def iso_to_dt_utc(iso_str: str) -> datetime:
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass
class FetchMeta:
    endpoint: str
    base_url: str
    symbol: str
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
    backoff = base_backoff_s
    last_err = None
    for _attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout_s)

            if resp.status_code == 200:
                return resp

            if resp.status_code == 418:
                raise BinanceHTTPError(f"418 ban risk. Stop. body={resp.text[:300]}")

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

            raise BinanceHTTPError(f"HTTP {resp.status_code}. params={params}. body={resp.text[:500]}")

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    raise BinanceHTTPError(f"Failed after retries. last_err={last_err}")


def fetch_aggtrades(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    sleep_ms: int = 250,
    session: Optional[requests.Session] = None,
    show_progress: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch aggregate trades filtered by execution time [start_ms, end_ms] (inclusive-ish),
    paginating by fromId to avoid losing trades when multiple share the same timestamp.
    """
    sess = session or requests.Session()
    url = BASE_URL + AGGTRADES_ENDPOINT

    rows: List[Dict[str, Any]] = []
    n_requests = 0

    cur_from_id: Optional[int] = None

    pbar = tqdm(disable=not show_progress, desc="Fetching aggTrades", unit="trades")

    while True:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        if cur_from_id is not None:
            params["fromId"] = cur_from_id

        resp = robust_get(sess, url, params=params)
        n_requests += 1
        batch = resp.json()

        if not batch:
            break

        # Validate monotonic IDs within batch
        ids = [int(x["a"]) for x in batch]
        if ids != sorted(ids):
            raise RuntimeError("Non-monotonic aggTradeId batch returned (unexpected).")

        rows.extend(batch)
        pbar.update(len(batch))

        last_id = int(batch[-1]["a"])
        next_from = last_id + 1

        if cur_from_id is not None and next_from <= cur_from_id:
            raise RuntimeError("Pagination stalled: fromId not increasing.")

        cur_from_id = next_from

        time.sleep(max(0.0, sleep_ms) / 1000.0)

        if len(batch) < limit:
            break

    pbar.close()
    return rows, n_requests


def raw_rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Binance aggTrades response fields:
      a: aggTradeId (int)
      p: price (str)
      q: quantity (str)
      f: firstTradeId (int)
      l: lastTradeId (int)
      T: timestamp (ms, int)
      m: isBuyerMaker (bool)
      M: isBestMatch (bool)
    """
    if not rows:
        return pd.DataFrame(
            columns=[
                "agg_trade_id",
                "price",
                "qty",
                "first_trade_id",
                "last_trade_id",
                "trade_time_ms",
                "is_buyer_maker",
                "is_best_match",
            ]
        )

    df = pd.DataFrame(rows)

    df = df.rename(
        columns={
            "a": "agg_trade_id",
            "p": "price",
            "q": "qty",
            "f": "first_trade_id",
            "l": "last_trade_id",
            "T": "trade_time_ms",
            "m": "is_buyer_maker",
            "M": "is_best_match",
        }
    )

    df["agg_trade_id"] = df["agg_trade_id"].astype("int64")
    df["first_trade_id"] = df["first_trade_id"].astype("int64")
    df["last_trade_id"] = df["last_trade_id"].astype("int64")
    df["trade_time_ms"] = df["trade_time_ms"].astype("int64")

    df["price"] = df["price"].astype("float64")
    df["qty"] = df["qty"].astype("float64")

    df["is_buyer_maker"] = df["is_buyer_maker"].astype("bool")
    df["is_best_match"] = df["is_best_match"].astype("bool")

    return df


def make_output_paths(out_dir: str, symbol: str, start_dt: datetime, end_dt: datetime) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    start_tag = start_dt.strftime("%Y%m%dT%H%M%SZ")
    end_tag = end_dt.strftime("%Y%m%dT%H%M%SZ")
    base = f"aggtrades_{symbol}_{start_tag}_{end_tag}"
    parquet_path = os.path.join(out_dir, base + ".parquet")
    meta_path = os.path.join(out_dir, base + ".meta.json")
    return parquet_path, meta_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", type=str, default=None, help="ISO8601 UTC e.g. 2026-01-22T00:00:00Z")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--minutes", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="data/raw")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sleep_ms", type=int, default=250)
    ap.add_argument("--force", action="store_true")
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

    parquet_path, meta_path = make_output_paths(args.out_dir, args.symbol, start_dt, end_dt)

    if (os.path.exists(parquet_path) or os.path.exists(meta_path)) and not args.force:
        print(f"[SKIP] Output exists: {parquet_path}")
        print("Use --force to overwrite.")
        return

    with requests.Session() as sess:
        rows, n_requests = fetch_aggtrades(
            symbol=args.symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=args.limit,
            sleep_ms=args.sleep_ms,
            session=sess,
            show_progress=True,
        )

    df = raw_rows_to_df(rows)

    # De-dupe and sort
    if not df.empty:
        df = df.sort_values(["trade_time_ms", "agg_trade_id"]).drop_duplicates(subset=["agg_trade_id"], keep="last").reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    meta = FetchMeta(
        endpoint=AGGTRADES_ENDPOINT,
        base_url=BASE_URL,
        symbol=args.symbol,
        start_iso=start_dt.isoformat().replace("+00:00", "Z"),
        end_iso=end_dt.isoformat().replace("+00:00", "Z"),
        start_ms=start_ms,
        end_ms=end_ms,
        limit=args.limit,
        sleep_ms=args.sleep_ms,
        fetched_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        n_rows=int(df.shape[0]),
        n_requests=int(n_requests),
        notes="Public Binance Spot REST aggTrades (time-filtered by T). No L2 historical replay.",
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"[OK] Saved {df.shape[0]:,} rows -> {parquet_path}")
    print(f"[OK] Meta -> {meta_path}")


if __name__ == "__main__":
    main()
