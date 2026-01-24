# src/features.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional

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
            definition=f"Trade intensity over last {W}s using kline n_trades: sum(n_trades over window W)",
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
        "ofi_1s": FeatureSpec(
            name="ofi_1s",
            definition="Order-flow imbalance proxy from aggTrades: ofi_1s(t) = buy_qty_1s(t) - sell_qty_1s(t)",
            inputs=["buy_qty_1s", "sell_qty_1s"],
            params={},
            causal=True,
        ),
        f"ofi_{W}": FeatureSpec(
            name=f"ofi_{W}",
            definition=f"Rolling OFI over last {W}s: sum(ofi_1s over window W)",
            inputs=["ofi_1s"],
            params={"W": W},
            causal=True,
        ),
        f"z_ofi_{W}_{L}": FeatureSpec(
            name=f"z_ofi_{W}_{L}",
            definition=f"Z-scored ofi_W using last {L}s: (ofi_W - mean_L) / std_L",
            inputs=[f"ofi_{W}"],
            params={"W": W, "L": L},
            causal=True,
        ),
        f"z_ofi_{W}_{L}_lag1": FeatureSpec(
            name=f"z_ofi_{W}_{L}_lag1",
            definition="Lagged z_ofi by 1 second to avoid same-second execution assumptions.",
            inputs=[f"z_ofi_{W}_{L}"],
            params={"lag": 1},
            causal=True,
        ),
        f"intensity_agg_{W}": FeatureSpec(
            name=f"intensity_agg_{W}",
            definition=f"AggTrades intensity over last {W}s: sum(n_aggs_1s over window W)",
            inputs=["n_aggs_1s"],
            params={"W": W},
            causal=True,
        ),
    }


def compute_features(
    bars: pd.DataFrame,
    trades_1s: Optional[pd.DataFrame] = None,
    W: int = 60,
    L: int = 600,
    eps: float = 1e-12,
) -> pd.DataFrame:
    df = bars.copy()

    if trades_1s is not None:
        t = trades_1s.copy()
        if not isinstance(t.index, pd.DatetimeIndex):
            raise RuntimeError("trades_1s must have a DateTimeIndex.")
        if t.index.tz is None:
            raise RuntimeError("trades_1s index must be timezone-aware (UTC).")

        df = df.join(t, how="left")

        trade_cols_float = ["buy_qty_1s", "sell_qty_1s", "qty_1s", "ofi_1s"]
        for c in trade_cols_float:
            if c in df.columns:
                df[c] = df[c].fillna(0.0).astype("float64")

        trade_cols_int = ["n_aggs_1s", "buy_count_1s", "sell_count_1s", "ofi_count_1s"]
        for c in trade_cols_int:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype("int64")

        if "is_imputed_trades" in df.columns:
            df["is_imputed_trades"] = df["is_imputed_trades"].fillna(True).astype("bool")

        if "ofi_1s" not in df.columns and ("buy_qty_1s" in df.columns and "sell_qty_1s" in df.columns):
            df["ofi_1s"] = df["buy_qty_1s"] - df["sell_qty_1s"]

    log_close = np.log(df["close"].astype("float64"))
    df["r1"] = log_close.diff()

    df[f"rv_{W}"] = df["r1"].rolling(window=W, min_periods=W).std()

    df[f"intensity_{W}"] = df["n_trades"].rolling(window=W, min_periods=W).sum()

    denom = np.maximum(df["volume"].to_numpy(dtype="float64"), eps)
    df["tbr"] = df["taker_buy_base"].to_numpy(dtype="float64") / denom

    df["sv"] = (2.0 * df["tbr"] - 1.0) * df["volume"]

    df[f"svi_{W}"] = df["sv"].rolling(window=W, min_periods=W).sum()

    svi_col = f"svi_{W}"
    mean_L = df[svi_col].rolling(window=L, min_periods=L).mean()
    std_L = df[svi_col].rolling(window=L, min_periods=L).std()

    z_svi = f"z_svi_{W}_{L}"
    df[z_svi] = (df[svi_col] - mean_L) / std_L
    df[f"{z_svi}_lag1"] = df[z_svi].shift(1)

    if trades_1s is not None and "ofi_1s" in df.columns:
        ofiW = f"ofi_{W}"
        df[ofiW] = df["ofi_1s"].rolling(window=W, min_periods=W).sum()

        mean_L_ofi = df[ofiW].rolling(window=L, min_periods=L).mean()
        std_L_ofi = df[ofiW].rolling(window=L, min_periods=L).std()

        z_ofi = f"z_ofi_{W}_{L}"
        df[z_ofi] = (df[ofiW] - mean_L_ofi) / std_L_ofi
        df[f"{z_ofi}_lag1"] = df[z_ofi].shift(1)

        if "n_aggs_1s" in df.columns:
            df[f"intensity_agg_{W}"] = df["n_aggs_1s"].rolling(window=W, min_periods=W).sum()

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default="data/processed/bars_1s.parquet", help="Input canonical 1s bars parquet")
    ap.add_argument("--trades_1s", default=None, help="Optional: trades_1s parquet built from aggTrades")
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

    if not isinstance(bars.index, pd.DatetimeIndex):
        raise RuntimeError("bars must have a DateTimeIndex.")
    if bars.index.tz is None:
        raise RuntimeError("bars index must be timezone-aware (UTC).")

    trades = None
    if args.trades_1s is not None:
        trades = pd.read_parquet(args.trades_1s)
        if not isinstance(trades.index, pd.DatetimeIndex):
            raise RuntimeError("trades_1s must have a DateTimeIndex.")
        if trades.index.tz is None:
            raise RuntimeError("trades_1s index must be timezone-aware (UTC).")

    feats = compute_features(bars, trades_1s=trades, W=args.W, L=args.L, eps=args.eps)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    feats.to_parquet(args.out)
    print(f"[OK] Saved features -> {args.out} rows={len(feats):,} cols={feats.shape[1]}")

    if args.save_provenance:
        prov = feature_provenance(W=args.W, L=args.L, eps=args.eps)
        os.makedirs(os.path.dirname(args.prov_out), exist_ok=True)
        prov_dict = {k: asdict(v) for k, v in prov.items()}
        with open(args.prov_out, "w", encoding="utf-8") as f:
            json.dump(prov_dict, f, indent=2)
        print(f"[OK] Saved provenance -> {args.prov_out}")


if __name__ == "__main__":
    main()
