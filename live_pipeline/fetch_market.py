import pandas as pd
from pathlib import Path
from datetime import datetime

Path("data/live").mkdir(parents=True, exist_ok=True)

df = pd.DataFrame([{
    "timestamp": datetime.utcnow().isoformat(),
    "source": "test",
    "value": 1
}])

df.to_csv("data/live/test_output.csv", index=False)

print("Wrote data/live/test_output.csv")
