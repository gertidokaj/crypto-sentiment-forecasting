import os
import re
import csv
import time
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    # Lightweight sentiment (recommended)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


# -----------------------------
# Config
# -----------------------------
DEFAULT_SOURCES = [
    # You can add/remove sources here safely
    ("CoinDesk", "https://feeds.feedburner.com/CoinDesk"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("CryptoSlate", "https://cryptoslate.com/feed/"),
]

CRYPTO_KEYWORDS = {
    "BTC": [r"\bbitcoin\b", r"\bbtc\b"],
    "ETH": [r"\bethereum\b", r"\beth\b"],
    "BNB": [r"\bbnb\b", r"\bbinance\b", r"\bbinance coin\b", r"\bbnb chain\b"],
}

TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(TS_FMT)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tag_crypto(text: str) -> str:
    t = (text or "").lower()
    for coin, pats in CRYPTO_KEYWORDS.items():
        for p in pats:
            if re.search(p, t, flags=re.IGNORECASE):
                return coin
    return "ALL"

def stable_id(link: str, title: str) -> str:
    key = (link or "") + "||" + (title or "")
    return hashlib.md5(key.encode("utf-8")).hexdigest()

def compute_sentiment(analyzer, text: str) -> float:
    """
    Returns compound score in [-1, 1].
    """
    if analyzer is None:
        return float("nan")
    text = normalize_text(text)
    if not text:
        return float("nan")
    return float(analyzer.polarity_scores(text)["compound"])

def read_existing_links(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        if "link" in df.columns:
            return set(df["link"].dropna().astype(str).tolist())
    except Exception:
        pass
    return set()

def append_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# -----------------------------
# Main
# -----------------------------
def fetch_news_to_csv(
    out_latest_path: str,
    out_history_path: str,
    max_items_per_source: int = 50,
    dedupe_by_link: bool = True,
) -> dict:
    """
    Writes:
      - news_latest.csv (overwrite each run)
      - news_history.csv (append)
    Returns a status dict for logging.
    """
    started = time.time()
    ts_utc = utc_now_iso()

    if feedparser is None:
        raise RuntimeError("feedparser is not installed. Add it to requirements.")

    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None

    existing_links = read_existing_links(out_history_path) if dedupe_by_link else set()

    collected = []
    total_seen = 0
    total_kept = 0

    for source_name, url in DEFAULT_SOURCES:
        feed = feedparser.parse(url)
        entries = getattr(feed, "entries", []) or []
        entries = entries[:max_items_per_source]

        for e in entries:
            total_seen += 1

            title = normalize_text(getattr(e, "title", "") or "")
            link = normalize_text(getattr(e, "link", "") or "")
            summary = normalize_text(getattr(e, "summary", "") or "")

            if dedupe_by_link and link and link in existing_links:
                continue

            text_for_sent = f"{title}. {summary}".strip()
            sent = compute_sentiment(analyzer, text_for_sent)

            coin = tag_crypto(f"{title} {summary}")

            row = {
                "ts_utc": ts_utc,
                "source": source_name,
                "crypto": coin,                 # BTC/ETH/BNB/ALL
                "title": title,
                "link": link,
                "sentiment": sent,              # [-1, 1]
                "id": stable_id(link, title),   # stable hash for debugging
            }

            collected.append(row)
            total_kept += 1
            if dedupe_by_link and link:
                existing_links.add(link)

    # Write latest (overwrite)
    df_latest = pd.DataFrame(collected)
    ensure_dir(os.path.dirname(out_latest_path))
    df_latest.to_csv(out_latest_path, index=False)

    # Append to history
    ensure_dir(os.path.dirname(out_history_path))
    fields = ["ts_utc", "source", "crypto", "title", "link", "sentiment", "id"]
    if collected:
        append_csv(out_history_path, collected, fields)

    return {
        "ts_utc": ts_utc,
        "news_latest_rows": int(len(df_latest)),
        "news_seen_raw": int(total_seen),
        "news_kept": int(total_kept),
        "sentiment_enabled": bool(analyzer is not None),
        "elapsed_sec": round(time.time() - started, 2),
    }


if __name__ == "__main__":
    # Paths compatible with your repo structure
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    live_dir = os.path.join(base_dir, "data", "live")

    latest = os.path.join(live_dir, "news_latest.csv")
    history = os.path.join(live_dir, "news_history.csv")

    status = fetch_news_to_csv(latest, history, max_items_per_source=50, dedupe_by_link=True)
    print(json.dumps(status, indent=2))
