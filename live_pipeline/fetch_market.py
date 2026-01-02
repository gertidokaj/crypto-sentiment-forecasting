import os
import requests
import pandas as pd
from live_pipeline.utils_io import write_csv, append_csv, utc_now_iso

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

# coin_id -> symbol
COINS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binancecoin": "BNB",
}

def fetch_prices():
    params = {
        "ids": ",".join(COINS.keys()),
        "vs_currencies": "usd",
        "include_24hr_change": "true"
    }
    r = requests.get(COINGECKO_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    ts = utc_now_iso()
    rows = []
    for coin_id, sym in COINS.items():
        price = data[coin_id]["usd"]
        chg24 = data[coin_id].get("usd_24h_change", None)
        rows.append({
            "ts_utc": ts,
            "crypto": sym,
            "price_usd": float(price),
            "pct_change_24h": None if chg24 is None else float(chg24),
        })
    return pd.DataFrame(rows)

def main():
    base = os.path.join("data", "live")
    df = fetch_prices()

    write_csv(df, os.path.join(base, "market_latest.csv"))
    append_csv(df, os.path.join(base, "market_history.csv"), dedupe_cols=["ts_utc", "crypto"])
    print("Saved market_latest.csv and appended market_history.csv")

if __name__ == "__main__":
    main()

