import os
import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from live_pipeline.utils_io import write_csv, append_csv, utc_now_iso

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://news.google.com/rss/search?q=bitcoin+OR+ethereum+OR+BNB+crypto&hl=en-GB&gl=GB&ceid=GB:en",
]

KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "BNB": ["bnb", "binance"],
}

def tag_crypto(text: str):
    t = (text or "").lower()
    hits = []
    for sym, kws in KEYWORDS.items():
        if any(k in t for k in kws):
            hits.append(sym)
    return hits

def main():
    base = os.path.join("data", "live")
    analyzer = SentimentIntensityAnalyzer()
    ts = utc_now_iso()

    rows = []
    for url in RSS_FEEDS:
        d = feedparser.parse(url)
        for e in d.entries[:50]:
            title = getattr(e, "title", "")
            summary = getattr(e, "summary", "")
            link = getattr(e, "link", "")
            text = f"{title} {summary}".strip()
            cryptos = tag_crypto(text)

            if not cryptos:
                continue

            score = analyzer.polarity_scores(text)["compound"]
            for c in cryptos:
                rows.append({
                    "ts_utc": ts,
                    "crypto": c,
                    "source": url,
                    "title": title,
                    "link": link,
                    "sentiment_compound": float(score),
                })

    df = pd.DataFrame(rows)
    out_latest = os.path.join(base, "news_latest.csv")
    out_hist = os.path.join(base, "news_history.csv")

    if df.empty:
        # still write a valid empty file so dashboard doesn't crash
        df = pd.DataFrame(columns=["ts_utc","crypto","source","title","link","sentiment_compound"])

    write_csv(df, out_latest)
    append_csv(df, out_hist, dedupe_cols=["ts_utc","crypto","link"])
    print("Saved news_latest.csv and appended news_history.csv")

if __name__ == "__main__":
    main()

