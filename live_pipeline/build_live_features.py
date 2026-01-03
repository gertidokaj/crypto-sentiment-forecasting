# live_pipeline/build_live_features.py
from __future__ import annotations

import os
import pandas as pd

from .utils_io import ensure_dir, atomic_write_csv

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

    history_path = os.path.join(LIVE_DIR, "market_history.csv")
    latest_path = os.path.join(LIVE_DIR, "market_latest.csv")
    out_path = os.path.join(LIVE_DIR, "features_latest.csv")

    if not os.path.exists(latest_path):
        atomic_write_csv(_empty_features(), out_path)
        return

    latest = pd.read_csv(latest_path)

    # Ensure required columns exist
    for c in MARKET_COLS:
        if c not in latest.columns:
            latest[c] = pd.NA
    latest = latest[MARKET_COLS].copy()

    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
    else:
        hist = latest.copy()

    for c in MARKET_COLS:
        if c not in hist.columns:
            hist[c] = pd.NA
    hist = hist[MARKET_COLS].copy()

    # Parse / coerce types
    hist["ts_utc"] = pd.to_datetime(hist["ts_utc"], errors="coerce", utc=True)
    latest["ts_utc"] = pd.to_datetime(latest["ts_utc"], errors="coerce", utc=True)

    hist["price_usd"] = pd.to_numeric(hist["price_usd"], errors="coerce")
    latest["price_usd"] = pd.to_numeric(latest["price_usd"], errors="coerce")

    hist = hist.dropna(subset=["ts_utc", "crypto", "price_usd"])
    latest = latest.dropna(subset=["ts_utc", "crypto", "price_usd"])

    if latest.empty:
        atomic_write_csv(_empty_features(), out_path)
        return

    feats = []
    for crypto, g in hist.sort_values("ts_utc").groupby("crypto"):
        g = g.copy()
        g = g.set_index("ts_utc").sort_index()

        # Resample to hourly (Pandas now prefers lowercase "h")
        hourly = g.resample("1h").last().dropna(subset=["price_usd"]).copy()
        if hourly.empty:
            continue

        hourly["crypto"] = crypto

        # Returns
        hourly["return_1h"] = hourly["price_usd"].pct_change(1)
        hourly["return_24h"] = hourly["price_usd"].pct_change(24)

        # Moving averages
        hourly["ma_3"] = hourly["price_usd"].rolling(3).mean()
        hourly["ma_6"] = hourly["price_usd"].rolling(6).mean()
        hourly["ma_24"] = hourly["price_usd"].rolling(24).mean()

        # Volatility of 1h returns
        hourly["vol_6"] = hourly["return_1h"].rolling(6).std()
        hourly["vol_24"] = hourly["return_1h"].rolling(24).std()

        # Target timestamp = latest timestamp for this crypto, floored to hour (lowercase "h")
        target_ts = latest.loc[latest["crypto"] == crypto, "ts_utc"].max()
        if pd.isna(target_ts):
            continue
        target_ts = target_ts.floor("h")

        # Take the last available hourly row <= target_ts
        hourly = hourly.reset_index()
        row = hourly.loc[hourly["ts_utc"] <= target_ts].tail(1)[
            ["ts_utc", "crypto", "price_usd", "return_1h", "return_24h", "ma_3", "ma_6", "ma_24", "vol_6", "vol_24"]
        ]
        if not row.empty:
            feats.append(row)

    if feats:
        out = pd.concat(feats, ignore_index=True)
        out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        for c in FEATURE_COLS:
            if c not in out.columns:
                out[c] = pd.NA
        out = out[FEATURE_COLS]
    else:
        out = _empty_features()

    atomic_write_csv(out, out_path)


if __name__ == "__main__":
    main()
