# scripts/backfill_coingecko_hourly.py
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import requests

LIVE_DIR = os.path.join("data", "live")
OUT_HISTORY = os.path.join(LIVE_DIR, "market_history.csv")
OUT_LATEST = os.path.join(LIVE_DIR, "market_latest.csv")

# CoinGecko IDs (important: these are NOT tickers)
COINS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
}

REQUIRED_COLS = [
    "ts_utc",
    "crypto",
    "price_usd",
    "mktcap_usd",
    "volume_24h_usd",
    "pct_change_1h",
    "pct_change_24h",
    "source",
    "fetched_at_utc",
]

SOURCE_NAME = "coingecko"

def to_iso_z(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def floor_hour(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)

def fetch_market_chart_usd(coin_id: str, days: int) -> Dict:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days)}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def build_hourly_df(sym: str, coin_id: str, days: int) -> pd.DataFrame:
    data = fetch_market_chart_usd(coin_id, days=days)

    # arrays of [ms, value]
    prices = pd.DataFrame(data.get("prices", []), columns=["ms", "price_usd"])
    mcap = pd.DataFrame(data.get("market_caps", []), columns=["ms", "mktcap_usd"])
    vols = pd.DataFrame(data.get("total_volumes", []), columns=["ms", "volume_24h_usd"])

    if prices.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    df = prices.merge(mcap, on="ms", how="left").merge(vols, on="ms", how="left")

    # convert ms -> hourly ts_utc
    df["ts_utc"] = pd.to_datetime(df["ms"], unit="ms", utc=True).dt.floor("H")
    df = df.drop(columns=["ms"])

    df["crypto"] = sym
    df["source"] = SOURCE_NAME
    df["fetched_at_utc"] = to_iso_z(datetime.now(timezone.utc))

    # remove duplicates created by flooring
    df = df.drop_duplicates(subset=["ts_utc", "crypto"], keep="last")

    # sort
    df = df.sort_values("ts_utc")

    # pct change features
    df["pct_change_1h"] = df["price_usd"].pct_change(1) * 100.0
    df["pct_change_24h"] = df["price_usd"].pct_change(24) * 100.0

    # format ts_utc as ISO Z for consistency with your pipeline
    df["ts_utc"] = df["ts_utc"].apply(lambda x: to_iso_z(x.to_pydatetime()))

    # enforce schema order
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[REQUIRED_COLS]

    return df

def main():
    os.makedirs(LIVE_DIR, exist_ok=True)

    # Choose backfill window
    days = 7  # change to 14 or 30 if you want more hours

    all_rows = []
    for sym, coin_id in COINS.items():
        print(f"Fetching {sym} ({coin_id}) for {days} days...")
        df = build_hourly_df(sym, coin_id, days=days)
        all_rows.append(df)
        time.sleep(1)  # be polite to API

    hist = pd.concat(all_rows, ignore_index=True)

    # drop missing prices
    hist["price_usd"] = pd.to_numeric(hist["price_usd"], errors="coerce")
    hist = hist.dropna(subset=["ts_utc", "crypto", "price_usd"])

    # save history
    hist.to_csv(OUT_HISTORY, index=False)

    # latest = latest hour per crypto
    hist_ts = pd.to_datetime(hist["ts_utc"], utc=True, errors="coerce")
    hist2 = hist.copy()
    hist2["_ts"] = hist_ts
    latest = (
        hist2.sort_values("_ts")
             .groupby("crypto", as_index=False)
             .tail(1)
             .drop(columns=["_ts"])
    )
    latest.to_csv(OUT_LATEST, index=False)

    print("Saved:", OUT_HISTORY)
    print("Saved:", OUT_LATEST)
    print("Rows:", len(hist), "Latest rows:", len(latest))

if __name__ == "__main__":
    main()
