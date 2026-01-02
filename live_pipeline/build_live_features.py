# live_pipeline/build_live_features.py
from __future__ import annotations

import os
import pandas as pd

from .utils_io import ensure_dir, atomic_write_csv, utc_now, floor_to_hour_utc

LIVE_DIR = os.path.join("data", "live")

MARKET_COLS = ["ts_utc", "crypto", "price_usd"]
FEATURE_COLS = [
    "ts_utc", "crypto",
    "price_usd",
    "return_1h",
    "return_24h",
    "ma_3", "ma_6", "ma_24",
    "vol_6", "vol_24",
]

def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_COLS)

def main():
    ensure_dir(LIVE_DIR)

    run_ts = floor_to_hour_utc(utc_now()).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    history_path = os.path.join(LIVE_DIR, "market_history.csv")
    latest_path = os.path.join(LIVE_DIR, "market_latest.csv")
    out_path = os.path.join(LIVE_DIR, "features_latest.csv")

    # Always ensure latest exists; if not, create empty features and exit safely
    if not os.path.exists(latest_path):
        atomic_write_csv(_empty_features(), out_path)
        return

    latest = pd.read_csv(latest_path)

    # Enforce required columns
    for c in MARKET_COLS:
        if c not in latest.columns:
            latest[c] = pd.NA

    latest = latest[MARKET_COLS].copy()

    # Load history if available; otherwise initialize with latest
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
    else:
        hist = latest.copy()

    for c in MARKET_COLS:
        if c not in hist.columns:
            hist[c] = pd.NA
    hist = hist[MARKET_COLS].copy()

    # Parse timestamps
    hist["ts_utc"] = pd.to_datetime(hist["ts_utc"], errors="coerce", utc=True)
    latest["ts_utc"] = pd.to_datetime(latest["ts_utc"], errors="coerce", utc=True)

    # Drop rows with missing keys
    hist = hist.dropna(subset=["ts_utc", "crypto", "price_usd"])
    latest = latest.dropna(subset=["ts_utc", "crypto", "price_usd"])

    if latest.empty:
        atomic_write_csv(_empty_features(), out_path)
        return

    # Compute rolling features per crypto
    feats = []
    for crypto, g in hist.sort_values("ts_utc").groupby("crypto"):
        g = g.copy()
        g["price_usd"] = pd.to_numeric(g["price_usd"], errors="coerce")

        g["return_1h"] = g["price_usd"].pct_change(1)
        g["return_24h"] = g["price_usd"].pct_change(24)

        g["ma_3"] = g["price_usd"].rolling(3).mean()
        g["ma_6"] = g["price_usd"].rolling(6).mean()
        g["ma_24"] = g["price_usd"].rolling(24).mean()

        g["vol_6"] = g["return_1h"].rolling(6).std()
        g["vol_24"] = g["return_1h"].rolling(24).std()

        # Take the row matching this run hour (latest timestamp for this crypto)
        target_ts = latest.loc[latest["crypto"] == crypto, "ts_utc"].max()
        if pd.isna(target_ts):
            continue

        row = g.loc[g["ts_utc"] == target_ts, ["ts_utc", "crypto", "price_usd",
                                              "return_1h", "return_24h",
                                              "ma_3", "ma_6", "ma_24",
                                              "vol_6", "vol_24"]]
        if row.empty:
            # If history didn't contain that exact hour, take last available row <= target_ts
            row = g.loc[g["ts_utc"] <= target_ts].tail(1)[["ts_utc", "crypto", "price_usd",
                                                          "return_1h", "return_24h",
                                                          "ma_3", "ma_6", "ma_24",
                                                          "vol_6", "vol_24"]]
        if not row.empty:
            feats.append(row)

    if feats:
        out = pd.concat(feats, ignore_inde_
