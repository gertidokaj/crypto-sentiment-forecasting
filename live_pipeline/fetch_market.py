# live_pipeline/fetch_market.py
from __future__ import annotations

import os
import json
import traceback
from typing import Dict, List
from datetime import datetime, timezone

import pandas as pd

from .utils_io import ensure_dir, atomic_write_csv, utc_now, floor_to_hour_utc

LIVE_DIR = os.path.join("data", "live")

REQUIRED_COLS = [
    "ts_utc", "crypto", "price_usd",
    "mktcap_usd", "volume_24h_usd",
    "pct_change_1h", "pct_change_24h",
    "source", "fetched_at_utc",
]

CRYPTO_LIST = ["BTC", "ETH", "BNB"]     # keep consistent everywhere
SOURCE_NAME = "coingecko"


def fetch_prices(cryptos: List[str]) -> Dict[str, Dict]:
    """
    Fetch prices from CoinGecko (public endpoint, no key).
    Returns dict keyed by symbols: BTC/ETH/BNB.
    """
    import time
    import requests

    id_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
    }

    ids = ",".join(id_map[c] for c in cryptos if c in id_map)

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ids,
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
    }

    headers = {"User-Agent": "crypto-sentiment-forecasting/1.0"}

    last_exc = None
    data = None

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)

            # Some GitHub runners get rate-limited; retry a few times
            if r.status_code in (403, 429):
                time.sleep(2 * (attempt + 1))
                continue

            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_exc = e
            time.sleep(2 * (attempt + 1))

    if data is None:
        raise RuntimeError(f"CoinGecko request failed after retries: {last_exc}")

    out: Dict[str, Dict] = {}
    for sym in cryptos:
        cg_id = id_map.get(sym)
        rec = data.get(cg_id, {}) if cg_id else {}

        out[sym] = {
            "price_usd": rec.get("usd"),
            "mktcap_usd": rec.get("usd_market_cap"),
            "volume_24h_usd": rec.get("usd_24h_vol"),
            "pct_change_24h": rec.get("usd_24h_change"),
            # Simple endpoint usually doesn't provide 1h change reliably
            "pct_change_1h": None,
        }

    return out


def build_market_latest(run_ts: str) -> pd.DataFrame:
    fetched_at = utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")

    try:
        raw = fetch_prices(CRYPTO_LIST)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        print("ERROR in fetch_prices():", err)

        ensure_dir(LIVE_DIR)
        with open(os.path.join(LIVE_DIR, "market_error.txt"), "w", encoding="utf-8") as f:
            f.write(err)

        # Return empty-but-schema-correct dataframe
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

    # Enforce schema and order
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[REQUIRED_COLS]

    return df


def append_market_history(latest_df: pd.DataFrame, history_path: str) -> pd.DataFrame:
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
    else:
        hist = pd.DataFrame(columns=REQUIRED_COLS)

    # Align columns
    for col in REQUIRED_COLS:
        if col not in hist.columns:
            hist[col] = pd.NA
        if col not in latest_df.columns:
            latest_df[col] = pd.NA

    latest_df = latest_df[REQUIRED_COLS]
    hist = hist[REQUIRED_COLS]

    out = pd.concat([hist, latest_df], ignore_index=True)

    # Optional: drop duplicates (same ts_utc+crypto)
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

    # ALWAYS write a debug file so we can see what happened in CI
    debug_path = os.path.join(LIVE_DIR, "market_debug.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(f"run_ts={run_ts}\n")
        f.write(f"latest_rows={len(latest)}\n")
        f.write(f"history_rows={len(history)}\n")
        f.write(f"latest_cols={list(latest.columns)}\n")

    # Status file
    status_path = os.path.join(LIVE_DIR, "status.json")
    status = {
        "ts_utc": run_ts,
        "market_latest_rows": int(len(latest)),
        "market_history_rows": int(len(history)),
        "updated_at_utc": utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    atomic_write_json(status, status_path)

    print("Saved market_latest.csv, market_history.csv, market_debug.txt, status.json")


def atomic_write_json(obj: dict, path: str) -> None:
    # Small helper to avoid partial writes
    import tempfile
    d = os.path.dirname(path)
    ensure_dir(d)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=d)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
