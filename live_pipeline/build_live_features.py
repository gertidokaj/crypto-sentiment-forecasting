
import os
import pandas as pd

def main():
    base = os.path.join("data", "live")

    market_latest = os.path.join(base, "market_latest.csv")
    news_latest = os.path.join(base, "news_latest.csv")

    m = pd.read_csv(market_latest)
    n = pd.read_csv(news_latest)

    # Aggregate news sentiment per crypto for the current run timestamp
    if not n.empty:
        sent = (n.groupby(["ts_utc","crypto"], as_index=False)
                  .agg(news_count=("link","count"),
                       news_sent_mean=("sentiment_compound","mean"),
                       news_sent_std=("sentiment_compound","std")))
    else:
        sent = pd.DataFrame(columns=["ts_utc","crypto","news_count","news_sent_mean","news_sent_std"])

    feat = pd.merge(m, sent, on=["ts_utc","crypto"], how="left")

    # fill missing news values
    feat["news_count"] = feat["news_count"].fillna(0).astype(int)
    feat["news_sent_mean"] = feat["news_sent_mean"].fillna(0.0)
    feat["news_sent_std"] = feat["news_sent_std"].fillna(0.0)

    out = os.path.join(base, "live_features.csv")
    feat.to_csv(out, index=False)
    print("Saved live_features.csv")

if __name__ == "__main__":
    main()
