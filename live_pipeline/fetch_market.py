import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

OUT_DIR = Path("data/live")
OUT_DIR.mkdir(parents=True, exist_ok=True)

now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

df = pd.DataFrame([
    {"ts_utc": now.isoformat(), "crypto": "BTC", "close": 1.0, "volume": 1.0},
    {"ts_utc": now.isoformat(), "crypto": "ETH", "close": 1.0, "volume": 1.0},
    {"ts_utc": now.isoformat(), "crypto": "BNB", "close": 1.0, "volume": 1.0},
])

df.to_csv(OUT_DIR / "market_latest.csv", index=False)
print("Wrote data/live/market_latest.csv")
print(df.head())
