# live_pipeline/utils_io.py
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def floor_to_hour_utc(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    d = os.path.dirname(path)
    ensure_dir(d)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=d)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# Backwards-compatible helpers used by fetch_news_rss.py
def write_csv(df: pd.DataFrame, path: str) -> None:
    atomic_write_csv(df, path)


def append_csv(df: pd.DataFrame, path: str, dedupe_cols: Optional[List[str]] = None) -> None:
    if os.path.exists(path):
        existing = pd.read_csv(path)
        out = pd.concat([existing, df], ignore_index=True)
    else:
        out = df.copy()

    if dedupe_cols:
        for c in dedupe_cols:
            if c not in out.columns:
                out[c] = pd.NA
        out = out.drop_duplicates(subset=dedupe_cols, keep="last")

    atomic_write_csv(out, path)
