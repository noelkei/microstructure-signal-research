# src/features.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    name: str
    definition: str
    inputs: list
    params: Dict[str, Any]
    causal: bool


def feature_provenance(W: int, L: int, eps: float) -> Dict[str, FeatureSpec]:
    """
    Returns a dict describing each feature (what it means and how it is computed).
    Useful for the notebook/report to show you know what you're doing.
    """
    return {
        "r1": FeatureSpec(
            name="r1",
            definition="1-second log return: r1(t) = log(close_t) - log(close_{t-1})",
            inputs=["close"],
            params={},
            causal=True,
        ),
        f"rv_{W}": FeatureSpec(
            name=f"rv_{W}",
            definition=f"Realized vol proxy over last {W}s: std(r1 over window W)",
            inputs=["r1"],
            params={"W": W},
            causal=True,
        ),
        f"intensity_{W}": FeatureSpec(
            name=f"intensity_{W}",
            definition=f"Trade intensity over last {W}s: sum(n_trades over window W)",
            inputs=["n_trades"],
            params={"W": W},
            causal=True,
        ),
        "tbr": FeatureSpec(
            name="tbr",
            definition=f"Taker-buy ratio: tbr(t) = taker_buy_base_t / max(volume_t, eps), eps={eps}",
            inputs=["taker_buy_base", "volume"],
            params={"eps": eps},
            causal=True,
        ),
        "sv": FeatureSpec(
            name="sv",
            definition="Signed volume proxy: sv(t) = (2*tbr(t) - 1) * volume_t",
            inputs=["tbr", "volume"],
            params={},
            causal=True,
        ),
        f"svi_{W}": FeatureSpec(
            name=f"svi_{W}",
            definition=f"Rolling signed volume imbalance: svi_W(t) = sum(sv over last {W}s)",
            inputs=["sv"],
            params={"W": W},
            causal=True,
        ),
        f"z_svi_{W}_{L}": FeatureSpec(
            name=f"z_svi_{W}_{L}",
            definition=f"Z-scored svi_W using last {L}s: (svi_W - mean_L) / std_L",
            inputs=[f"svi_{W}"],
            params={"W": W, "L": L},
            causal=True,
        ),
        f"z_svi_{W}_{L}_lag1": FeatureSpec(
            name=f"z_svi_{W}_{L}_lag1",
            definition="Lagged z_svi by 1 second to avoid same-second execution assumptions.",
            inputs=[f"z_svi_{W}_{L}"],
            params={"lag": 1},
            causal=True,
        ),
    }


def compute_features(bars: pd.DataFrame, W: int = 60, L: int = 600, eps: float = 1e-12) -> pd.DataFrame:
    """
    Compute microstructure proxy features from canonical 1-second bars.

    Inputs expected in bars:
      open, high, low, close, volume, quote_volume, n_trades, taker_buy_base, taker_buy_quote, is_imputed

    Output:
      A new DataFrame with the original columns + feature columns.
    """
    df = bars.copy()

    # --- 1) 1s log return ---
    # log(close_t) - log(close_{t-1})
    log_close = np.log(df["close"].astype("float64"))
    df["r1"] = log_close.diff()

    # --- 2) realized vol proxy ---
    # rolling std of r1 over last W seconds
    df[f"rv_{W}"] = df["r1"].rolling(window=W, min_periods=W).std()

    # --- 3) trade intensity ---
    # rolling sum of n_trades over last W seconds
    df[f"intensity_{W}"] = df["n_trades"].rolling(window=W, min_periods=W).sum()

    # --- 4) taker-buy ratio ---
    # taker_buy_base / max(volume, eps)
    denom = np.maximum(df["volume"].to_numpy(dtype="float64"), eps)
    df["tbr"] = df["taker_buy_base"].to_numpy(dtype="float64") / denom

    # --- 5) signed volume proxy ---
    # (2*tbr - 1) * volume
    df["sv"] = (2.0 * df["tbr"] - 1.0) * df["volume"]

    # --- 6) rolling signed volume imbalance ---
    df[f"svi_{W}"] = df["sv"].rolling(window=W, min_periods=W).sum()

    # --- 7) z-score svi using window L (L > W) ---
    svi_col = f"svi_{W}"
    mean_L = df[svi_col].rolling(window=L, min_periods=L).mean()
    std_L = df[svi_col].rolling(window=L, min_periods=L).std()

    z_col = f"z_svi_{W}_{L}"
    df[z_col] = (df[svi_col] - mean_L) / std_L

    # --- 8) 1-second lag version for safer execution assumptions ---
    df[f"{z_col}_lag1"] = df[z_col].shift(1)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default="data/processed/bars_1s.parquet", help="Input canonical 1s bars parquet")
    ap.add_argument("--out", default="data/processed/features_1s.parquet", help="Output features parquet")
    ap.add_argument("--W", type=int, default=60, help="Rolling window W in seconds")
    ap.add_argument("--L", type=int, default=600, help="Normalization window L in seconds (L > W)")
    ap.add_argument("--eps", type=float, default=1e-12, help="Small epsilon to avoid division by zero")
    ap.add_argument("--save_provenance", action="store_true", help="Also save a JSON describing the features.")
    ap.add_argument("--prov_out", default="data/processed/feature_provenance.json", help="Where to save provenance JSON")
    args = ap.parse_args()

    if args.L <= args.W:
        raise ValueError("Require L > W for stable z-scoring (e.g., W=60, L=600).")

    bars = pd.read_parquet(args.bars)

    # basic sanity: require DateTimeIndex
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise RuntimeError("bars must have a DateTimeIndex. Did you run build_bars correctly?")
    if bars.index.tz is None:
        raise RuntimeError("bars index must be timezone-aware (UTC).")

    feats = compute_features(bars, W=args.W, L=args.L, eps=args.eps)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    feats.to_parquet(args.out)
    print(f"[OK] Saved features -> {args.out} rows={len(feats):,} cols={feats.shape[1]}")

    if args.save_provenance:
        prov = feature_provenance(W=args.W, L=args.L, eps=args.eps)
        os.makedirs(os.path.dirname(args.prov_out), exist_ok=True)
        # Convert dataclasses to dicts for JSON
        prov_dict = {k: asdict(v) for k, v in prov.items()}
        with open(args.prov_out, "w", encoding="utf-8") as f:
            json.dump(prov_dict, f, indent=2)
        print(f"[OK] Saved provenance -> {args.prov_out}")


if __name__ == "__main__":
    main()
