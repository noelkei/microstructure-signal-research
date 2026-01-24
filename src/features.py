# src/features.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional, List

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
            definition=f"Realized volatility proxy: rv_W(t) = std(r1 over last {W}s)",
            inputs=["r1"],
            params={"W": W},
            causal=True,
        ),
        f"intensity_{W}": FeatureSpec(
            name=f"intensity_{W}",
            definition=f"Trade intensity from klines: sum(n_trades over last {W}s)",
            inputs=["n_trades"],
            params={"W": W},
            causal=True,
        ),
        "tbr": FeatureSpec(
            name="tbr",
            definition=f"Taker-buy ratio (klines): taker_buy_base / max(volume, eps), eps={eps}",
            inputs=["taker_buy_base", "volume"],
            params={"eps": eps},
            causal=True,
        ),
        "sv": FeatureSpec(
            name="sv",
            definition="Signed volume proxy (klines): sv = (2*tbr - 1) * volume",
            inputs=["tbr", "volume"],
            params={},
            causal=True,
        ),
        f"svi_{W}": FeatureSpec(
            name=f"svi_{W}",
            definition=f"Rolling signed volume imbalance (klines): sum(sv over last {W}s)",
            inputs=["sv"],
            params={"W": W},
            causal=True,
        ),
        f"z_svi_{W}_{L}": FeatureSpec(
            name=f"z_svi_{W}_{L}",
            definition=f"Z-score of svi_{W} using last {L}s: (svi_W - mean_L)/std_L",
            inputs=[f"svi_{W}"],
            params={"W": W, "L": L},
            causal=True,
        ),
        f"z_svi_{W}_{L}_lag1": FeatureSpec(
            name=f"z_svi_{W}_{L}_lag1",
            definition="Lagged z_svi by 1 second for conservative execution assumptions.",
            inputs=[f"z_svi_{W}_{L}"],
            params={"lag": 1},
            causal=True,
        ),
        "ofi_base_1s": FeatureSpec(
            name="ofi_base_1s",
            definition="OFI proxy from aggTrades in base units: ofi_base_1s = taker_buy_base_1s - taker_sell_base_1s",
            inputs=["taker_buy_base_1s", "taker_sell_base_1s"],
            params={},
            causal=True,
        ),
        "ofi_ratio_1s": FeatureSpec(
            name="ofi_ratio_1s",
            definition=f"Normalized OFI proxy: ofi_base_1s / max(base_qty_1s, eps), eps={eps}",
            inputs=["ofi_base_1s", "base_qty_1s"],
            params={"eps": eps},
            causal=True,
        ),
        f"ofi_{W}": FeatureSpec(
            name=f"ofi_{W}",
            definition=f"Rolling OFI proxy over last {W}s: sum(ofi_base_1s over window W)",
            inputs=["ofi_base_1s"],
            params={"W": W},
            causal=True,
        ),
        f"z_ofi_{W}_{L}": FeatureSpec(
            name=f"z_ofi_{W}_{L}",
            definition=f"Z-score of ofi_{W} using last {L}s: (ofi_W - mean_L)/std_L",
            inputs=[f"ofi_{W}"],
            params={"W": W, "L": L},
            causal=True,
        ),
        f"z_ofi_{W}_{L}_lag1": FeatureSpec(
            name=f"z_ofi_{W}_{L}_lag1",
            definition="Lagged z_ofi by 1 second for conservative execution assumptions.",
            inputs=[f"z_ofi_{W}_{L}"],
            params={"lag": 1},
            causal=True,
        ),
        f"intensity_trades_{W}": FeatureSpec(
            name=f"intensity_trades_{W}",
            definition=f"Trade intensity from aggTrades: sum(n_aggtrades_1s over last {W}s)",
            inputs=["n_aggtrades_1s"],
            params={"W": W},
            causal=True,
        ),
        "cfi_1s": FeatureSpec(
            name="cfi_1s",
            definition="Count flow imbalance from aggTrades: cfi_1s = buy_count_1s - sell_count_1s",
            inputs=["buy_count_1s", "sell_count_1s"],
            params={},
            causal=True,
        ),
        "cfi_ratio_1s": FeatureSpec(
            name="cfi_ratio_1s",
            definition=f"Normalized count flow: cfi_ratio_1s = cfi_1s / max(n_aggtrades_1s, eps), eps={eps}",
            inputs=["cfi_1s", "n_aggtrades_1s"],
            params={"eps": eps},
            causal=True,
        ),
        f"cfi_{W}": FeatureSpec(
            name=f"cfi_{W}",
            definition=f"Rolling count flow imbalance over last {W}s: sum(cfi_1s over window W)",
            inputs=["cfi_1s"],
            params={"W": W},
            causal=True,
        ),
        f"z_cfi_{W}_{L}": FeatureSpec(
            name=f"z_cfi_{W}_{L}",
            definition=f"Z-score of cfi_{W} using last {L}s: (cfi_W - mean_L)/std_L",
            inputs=[f"cfi_{W}"],
            params={"W": W, "L": L},
            causal=True,
        ),
        f"z_cfi_{W}_{L}_lag1": FeatureSpec(
            name=f"z_cfi_{W}_{L}_lag1",
            definition="Lagged z_cfi by 1 second for conservative execution assumptions.",
            inputs=[f"z_cfi_{W}_{L}"],
            params={"lag": 1},
            causal=True,
        ),
        f"cfi_ratio_{W}": FeatureSpec(
            name=f"cfi_ratio_{W}",
            definition=f"Rolling normalized count flow over last {W}s: sum(cfi_ratio_1s over window W)",
            inputs=["cfi_ratio_1s"],
            params={"W": W},
            causal=True,
        ),
        f"z_cfi_ratio_{W}_{L}": FeatureSpec(
            name=f"z_cfi_ratio_{W}_{L}",
            definition=f"Z-score of cfi_ratio_{W} using last {L}s: (cfi_ratio_W - mean_L)/std_L",
            inputs=[f"cfi_ratio_{W}"],
            params={"W": W, "L": L},
            causal=True,
        ),
        f"z_cfi_ratio_{W}_{L}_lag1": FeatureSpec(
            name=f"z_cfi_ratio_{W}_{L}_lag1",
            definition="Lagged z_cfi_ratio by 1 second for conservative execution assumptions.",
            inputs=[f"z_cfi_ratio_{W}_{L}"],
            params={"lag": 1},
            causal=True,
        ),
        "max_share_1s": FeatureSpec(
            name="max_share_1s",
            definition="Trade size concentration from aggTrades: max_qty_1s / max(base_qty_1s, eps)",
            inputs=["max_qty_1s", "base_qty_1s"],
            params={"eps": eps},
            causal=True,
        ),
        f"z_max_share_{W}_{L}": FeatureSpec(
            name=f"z_max_share_{W}_{L}",
            definition=f"Z-score of rolling mean(max_share_1s, W) using last {L}s: (ms_roll - mean_L)/std_L",
            inputs=["max_share_1s"],
            params={"W": W, "L": L},
            causal=True,
        ),
        f"z_max_share_{W}_{L}_lag1": FeatureSpec(
            name=f"z_max_share_{W}_{L}_lag1",
            definition="Lagged z_max_share by 1 second for conservative execution assumptions.",
            inputs=[f"z_max_share_{W}_{L}"],
            params={"lag": 1},
            causal=True,
        ),
    }


def _assert_has_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def compute_features(
    bars: pd.DataFrame,
    trades_1s: Optional[pd.DataFrame] = None,
    W: int = 60,
    L: int = 600,
    eps: float = 1e-12,
) -> pd.DataFrame:
    df = bars.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("bars must have a DateTimeIndex.")
    if df.index.tz is None:
        raise RuntimeError("bars index must be timezone-aware (UTC).")

    if trades_1s is not None:
        t = trades_1s.copy()
        if not isinstance(t.index, pd.DatetimeIndex):
            raise RuntimeError("trades_1s must have a DateTimeIndex.")
        if t.index.tz is None:
            raise RuntimeError("trades_1s index must be timezone-aware (UTC).")

        required_trade_cols = [
            "n_aggtrades_1s",
            "base_qty_1s",
            "quote_qty_1s",
            "taker_buy_base_1s",
            "taker_sell_base_1s",
            "ofi_base_1s",
            "ofi_ratio_1s",
        ]
        _assert_has_columns(t, required_trade_cols, "trades_1s")

        df = df.join(t, how="left")

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
            if c in df.columns:
                df[c] = df[c].fillna(0.0).astype("float64")

        int_cols = [
            "n_aggtrades_1s",
            "buy_count_1s",
            "sell_count_1s",
            "ofi_count_1s",
        ]
        for c in int_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype("int64")

        if "is_imputed_trades" in df.columns:
            df["is_imputed_trades"] = df["is_imputed_trades"].fillna(True).astype("bool")

        if "ofi_base_1s" not in df.columns and ("taker_buy_base_1s" in df.columns and "taker_sell_base_1s" in df.columns):
            df["ofi_base_1s"] = df["taker_buy_base_1s"] - df["taker_sell_base_1s"]
            df["ofi_base_1s"] = df["ofi_base_1s"].fillna(0.0).astype("float64")

        if "ofi_ratio_1s" not in df.columns and "base_qty_1s" in df.columns and "ofi_base_1s" in df.columns:
            denom_trade = np.maximum(df["base_qty_1s"].to_numpy(dtype="float64"), eps)
            df["ofi_ratio_1s"] = df["ofi_base_1s"].to_numpy(dtype="float64") / denom_trade
            df["ofi_ratio_1s"] = df["ofi_ratio_1s"].astype("float64")

    log_close = np.log(df["close"].astype("float64"))
    df["r1"] = log_close.diff()

    df[f"rv_{W}"] = df["r1"].rolling(window=W, min_periods=W).std()

    df[f"intensity_{W}"] = df["n_trades"].rolling(window=W, min_periods=W).sum()

    denom = np.maximum(df["volume"].to_numpy(dtype="float64"), eps)
    df["tbr"] = df["taker_buy_base"].to_numpy(dtype="float64") / denom

    df["sv"] = (2.0 * df["tbr"] - 1.0) * df["volume"].astype("float64")

    df[f"svi_{W}"] = df["sv"].rolling(window=W, min_periods=W).sum()

    svi_col = f"svi_{W}"
    mean_L = df[svi_col].rolling(window=L, min_periods=L).mean()
    std_L = df[svi_col].rolling(window=L, min_periods=L).std()

    z_svi = f"z_svi_{W}_{L}"
    df[z_svi] = (df[svi_col] - mean_L) / std_L
    df[f"{z_svi}_lag1"] = df[z_svi].shift(1)

    if trades_1s is not None:
        ofiW = f"ofi_{W}"
        df[ofiW] = df["ofi_base_1s"].rolling(window=W, min_periods=W).sum()

        mean_L_ofi = df[ofiW].rolling(window=L, min_periods=L).mean()
        std_L_ofi = df[ofiW].rolling(window=L, min_periods=L).std()

        z_ofi = f"z_ofi_{W}_{L}"
        df[z_ofi] = (df[ofiW] - mean_L_ofi) / std_L_ofi
        df[f"{z_ofi}_lag1"] = df[z_ofi].shift(1)

        df[f"intensity_trades_{W}"] = df["n_aggtrades_1s"].rolling(window=W, min_periods=W).sum()

        if "buy_count_1s" in df.columns and "sell_count_1s" in df.columns:
            df["cfi_1s"] = df["buy_count_1s"].astype("float64") - df["sell_count_1s"].astype("float64")
            df["cfi_ratio_1s"] = df["cfi_1s"] / np.maximum(df["n_aggtrades_1s"].to_numpy(dtype="float64"), eps)

            cfiW = f"cfi_{W}"
            df[cfiW] = df["cfi_1s"].rolling(window=W, min_periods=W).sum()

            mean_L_cfi = df[cfiW].rolling(window=L, min_periods=L).mean()
            std_L_cfi = df[cfiW].rolling(window=L, min_periods=L).std()

            z_cfi = f"z_cfi_{W}_{L}"
            df[z_cfi] = (df[cfiW] - mean_L_cfi) / std_L_cfi
            df[f"{z_cfi}_lag1"] = df[z_cfi].shift(1)

            cfirW = f"cfi_ratio_{W}"
            df[cfirW] = df["cfi_ratio_1s"].rolling(window=W, min_periods=W).sum()

            mean_L_cfir = df[cfirW].rolling(window=L, min_periods=L).mean()
            std_L_cfir = df[cfirW].rolling(window=L, min_periods=L).std()

            z_cfir = f"z_cfi_ratio_{W}_{L}"
            df[z_cfir] = (df[cfirW] - mean_L_cfir) / std_L_cfir
            df[f"{z_cfir}_lag1"] = df[z_cfir].shift(1)

        if "max_share_1s" in df.columns:
            ms_roll = df["max_share_1s"].rolling(window=W, min_periods=W).mean()
            mean_L_ms = ms_roll.rolling(window=L, min_periods=L).mean()
            std_L_ms = ms_roll.rolling(window=L, min_periods=L).std()

            z_ms = f"z_max_share_{W}_{L}"
            df[z_ms] = (ms_roll - mean_L_ms) / std_L_ms
            df[f"{z_ms}_lag1"] = df[z_ms].shift(1)

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
