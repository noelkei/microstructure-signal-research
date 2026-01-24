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

            raise BinanceHTTPError(
                f"HTTP {resp.status_code}. params={params}. body={resp.text[:500]}"
            )

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    raise BinanceHTTPError(f"Failed after {max_retries} attempts. last_err={last_err}")


def bootstrap_start_id(
    sess: requests.Session,
    symbol: str,
    start_ms: int,
    limit: int = 1,
) -> Optional[int]:
    """
    Get the first aggregate trade id (a) at/after start_ms.
    IMPORTANT: We do NOT use fromId together with startTime/endTime (invalid combination).
    """
    url = BASE_URL + AGGTRADES_ENDPOINT
    params = {"symbol": symbol, "startTime": start_ms, "limit": limit}
    resp = robust_get(sess, url, params=params)
    batch = resp.json()
    if not batch:
        return None
    return int(batch[0]["a"])


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
    Fetch aggTrades in [start_ms, end_ms] using a robust pattern:

    1) Bootstrap with startTime ONLY to get the first trade id (a).
    2) Page forward with fromId ONLY (no startTime/endTime) and stop when trade time T > end_ms.
       Filter locally to keep only trades within the window.

    This avoids the invalid parameter combination: fromId + startTime/endTime.
    """
    if end_ms < start_ms:
        raise ValueError("end_ms must be >= start_ms")

    sess = session or requests.Session()
    url = BASE_URL + AGGTRADES_ENDPOINT

    start_id = bootstrap_start_id(sess, symbol=symbol, start_ms=start_ms, limit=1)
    if start_id is None:
        return [], 1  # 1 request for bootstrap

    rows: List[Dict[str, Any]] = []
    n_requests = 1  # bootstrap counted
    cur_from = start_id

    pbar = tqdm(disable=not show_progress, desc="Fetching aggTrades", unit="trades")

    last_id: Optional[int] = None

    while True:
        params = {"symbol": symbol, "fromId": cur_from, "limit": limit}
        resp = robust_get(sess, url, params=params)
        n_requests += 1
        batch = resp.json()
        if not batch:
            break

        # Validate monotonic ids within batch
        ids = [int(x["a"]) for x in batch]
        if ids != sorted(ids):
            raise RuntimeError("Non-monotonic aggTrade ids in batch (unexpected).")

        if last_id is not None and ids[0] <= last_id:
            raise RuntimeError("Pagination stalled (aggTrade id not increasing).")

        stop = False
        kept = 0

        for x in batch:
            tid = int(x["a"])
            tms = int(x["T"])
            last_id = tid

            if tms < start_ms:
                # should be rare after bootstrap, but keep it safe
                continue
            if tms > end_ms:
                stop = True
                break

            rows.append(x)
            kept += 1

        pbar.update(kept)

        # Advance pagination
        cur_from = int(batch[-1]["a"]) + 1

        if stop:
            break

        time.sleep(max(0.0, sleep_ms) / 1000.0)

        # If we received less than limit, we might be near the end of available trades.
        # Keep looping until empty batch, but this prevents overly tight loops.
        if len(batch) < limit:
            # try one more iteration; next call likely returns empty
            pass

    pbar.close()
    return rows, n_requests


def rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    aggTrades schema fields:
      a: aggTradeId (int)
      p: price (string)
      q: qty (string)
      f: first tradeId (int)
      l: last tradeId (int)
      T: timestamp (ms) (int)
      m: is buyer the maker (bool)
      M: ignore (bool)
    """
    if not rows:
        return pd.DataFrame(columns=["a", "p", "q", "f", "l", "T", "m"])

    df = pd.DataFrame(rows)

    # keep canonical columns
    keep = ["a", "p", "q", "f", "l", "T", "m"]
    df = df[keep].copy()

    df["a"] = df["a"].astype("int64")
    df["f"] = df["f"].astype("int64")
    df["l"] = df["l"].astype("int64")
    df["T"] = df["T"].astype("int64")
    df["m"] = df["m"].astype("bool")

    df["p"] = df["p"].astype("float64")
    df["q"] = df["q"].astype("float64")

    # sort + dedupe (safety)
    df = df.sort_values(["T", "a"]).drop_duplicates(subset=["a"], keep="last").reset_index(drop=True)
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
    ap.add_argument("--start", type=str, required=False, help="ISO8601 UTC e.g. 2026-01-22T00:00:00Z")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--minutes", type=int, default=None)
    ap.add_argument("--out_dir", type=str, default="data/raw")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sleep_ms", type=int, default=250)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Fetch ~2 minutes from --start (or now-2m if start not provided)")
    args = ap.parse_args()

    if args.smoke:
        if args.start is None:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(minutes=2)
        else:
            start_dt = iso_to_dt_utc(args.start)
            end_dt = start_dt + timedelta(minutes=2)
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

    df = rows_to_df(rows)

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
        notes="Public Binance Spot REST aggTrades. fromId is paginated without startTime/endTime; window filtered locally.",
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"[OK] Saved {df.shape[0]:,} rows -> {parquet_path}")
    print(f"[OK] Meta -> {meta_path}")


if __name__ == "__main__":
    main()
