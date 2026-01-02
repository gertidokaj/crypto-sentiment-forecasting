# live_pipeline/fetch_market.py
from __future__ import annotations

import os
import json
from typing import Dict, List

import pandas as pd

from .utils_io import ensure_dir, atomic_write_csv, utc_now, floor_to_hour_utc

LIVE_DIR = os.path.join("data", "live")

REQUIRED_COLS = [
    "ts_utc", "crypto", "price_usd",
    "mktcap_usd", "volume_24h_usd",
    "pct_change_1h", "pct_change_24h",
    "source", "fetched_at_utc",
]

CRYPTO_LIST = ["BTC", "ETH", "BNB"]
SOURCE_NAME = "coingecko"


def fetch_prices(cryptos: List[str]) -> Dict[str, Dict]:
    """
    Implement using your chosen API.
    For now we intentionally raise so the pipeline writes empty-but-valid outputs.
    """
    raise NotImplementedError("Implement fetch_prices() with your market data provider.")


def build_market_latest(run_ts: str) -> pd.DataFrame:
    fetched_at = utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")

    try:
        raw = fetch_prices(CRYPTO_LIST)
    except Exception:
        # Always return schema-valid df
        return pd.DataFrame(columns=REQUIRED_COLS)

    rows = []
    for c in CRYPTO_LIST:
        rec = raw.get(c, {}) or {}
        rows.append({
            "ts_utc": run_ts,
            "crypto": c,
            "price_usd": rec.get("price_usd"),
            "mktcap_usd": rec.get("mktcap_usd"),
            "volume_24h_usd": rec.get("volume_24h_usd"),
            "pct_change_1h": rec.get("pct_change_1h"),
            "pct_change_24h": rec.get("pct_change_24h"),
            "source": SOURCE_NAME,
            "fetched_at_utc": fetched_at,
        })

    df = pd.DataFrame(rows)

    # Enforce schema + order
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[REQUIRED_COLS]


def append_market_history(latest_df: pd.DataFrame, history_path: str) -> pd.DataFrame:
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
    else:
        hist = pd.DataFrame(columns=REQUIRED_COLS)

    for col in REQUIRED_COLS:
        if col not in hist.columns:
            hist[col] = pd.NA
        if col not in latest_df.columns:
            latest_df[col] = pd.NA

    latest_df = latest_df[REQUIRED_COLS]
    hist = hist[REQUIRED_COLS]

    out = pd.concat([hist, latest_df], ignore_index=True)
    out = out.drop_duplicates(subset=["ts_utc", "crypto"], keep="last")
    return out


def main():
    ensure_dir(LIVE_DIR)

    run_dt = floor_to_hour_utc(utc_now())
    run_ts = run_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    latest = build_market_latest(run_ts)

    latest_path = os.path.join(LIVE_DIR, "market_latest.csv")
    history_path = os.path.join(LIVE_DIR, "market_history.csv")

    atomic_write_csv(latest, latest_path)

    history = append_market_history(latest, history_path)
    atomic_write_csv(history, history_path)

    # Status (must be inside main)
    status_path = os.path.join(LIVE_DIR, "status.json")
    status = {
        "ts_utc": run_ts,
        "market_latest_rows": int(len(latest)),
        "market_history_rows": int(len(history)),
        "updated_at_utc": utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()
