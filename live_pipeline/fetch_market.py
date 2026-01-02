import pandas as pd
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("data/live")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Temporary test row (replace later with real market data)
df = pd.DataFrame([{
    "timestamp": datetime.utcnow().isoformat(),
    "crypto": "BTC",
    "close": 1.0,
    "volume": 1.0
}])

# IMPORTANT: write the file that build_live_features expects
df.to_csv(OUT_DIR / "market_latest.csv", index=False)

print("Wrote data/live/market_latest.csv")
