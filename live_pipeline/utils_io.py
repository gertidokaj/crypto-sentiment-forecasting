import os
import pandas as pd
from datetime import datetime, timezone

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def append_csv(df: pd.DataFrame, path: str, dedupe_cols=None) -> None:
    ensure_dir(os.path.dirname(path))
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
        if dedupe_cols:
            out = out.drop_duplicates(subset=dedupe_cols, keep="last")
    else:
        out = df.copy()
    out.to_csv(path, index=False)

