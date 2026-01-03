import os
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Crypto Live Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIVE_DIR = os.path.join(BASE_DIR, "data", "live")

FILES = {
    "market_history": os.path.join(LIVE_DIR, "market_history.csv"),
    "market_latest": os.path.join(LIVE_DIR, "market_latest.csv"),
    "features_latest": os.path.join(LIVE_DIR, "features_latest.csv"),
    "predictions_latest": os.path.join(LIVE_DIR, "predictions_latest.csv"),
    "predictions_history": os.path.join(LIVE_DIR, "predictions_history.csv"),
    "news_latest": os.path.join(LIVE_DIR, "news_latest.csv"),
    "status": os.path.join(LIVE_DIR, "status.json"),
}

DEFAULT_COINS = ["BTC", "ETH", "BNB"]

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      [data-testid="stSidebar"] { padding-top: 1rem; }

      .kpi-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px 16px 12px 16px;
        background: rgba(255,255,255,0.03);
        box-shadow: 0 6px 18px rgba(0,0,0,0.20);
      }
      .kpi-title { font-size: 0.85rem; opacity: 0.75; margin-bottom: 6px; }
      .kpi-value { font-size: 1.55rem; font-weight: 750; margin-bottom: 2px; }
      .kpi-sub { font-size: 0.85rem; opacity: 0.7; }

      .pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        font-size: 0.82rem; border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
      }
      .ok { color: #7CFC98; }
      .warn { color: #FFD166; }
      .bad { color: #FF6B6B; }
      .muted { opacity: 0.78; }

      .callout {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.025);
      }
      .callout-title { font-weight: 700; font-size: 1.02rem; margin-bottom: 4px; }
      .callout-body { opacity: 0.84; }
      .big-arrow { font-size: 1.25rem; font-weight: 900; }

      .small-note { font-size: 0.88rem; opacity: 0.78; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        # Handle empty files cleanly
        if os.path.getsize(path) == 0:
            return pd.DataFrame()

        # Try standard CSV
        try:
            return pd.read_csv(path)
        except Exception:
            # Sometimes news feeds get written with ; delimiter
            return pd.read_csv(path, sep=";", engine="python")
    except Exception:
        return pd.DataFrame()

def read_json_safe(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)

def fmt_pct(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:.{decimals}f}%"

def fmt_num(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def kpi(container, title: str, value: str, sub: str):
    container.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def status_badge(status: dict) -> tuple[str, str, str]:
    if not status:
        return ("UPDATED", "warn", "—")

    ok_val = None
    for key in ["ok", "success", "is_ok", "status_ok"]:
        if key in status:
            ok_val = status.get(key)
            break

    last_run = None
    for key in ["last_run_utc", "updated_at_utc", "ts_utc", "timestamp_utc"]:
        if key in status:
            last_run = status.get(key)
            break

    last_run_str = last_run if isinstance(last_run, str) and last_run else "—"

    if ok_val is True:
        return ("OK", "ok", last_run_str)
    if ok_val is False:
        return ("FAILED", "bad", last_run_str)
    return ("UPDATED", "warn", last_run_str)

def direction_label(yhat: float) -> tuple[str, str]:
    if pd.isna(yhat):
        return ("—", "warn")
    if yhat > 0:
        return ("UP", "ok")
    if yhat < 0:
        return ("DOWN", "bad")
    return ("FLAT", "warn")

def signal_strength(yhat: float, recent_vol: float) -> str:
    if pd.isna(yhat) or pd.isna(recent_vol) or recent_vol <= 0:
        return "Low"
    ratio = abs(yhat) / recent_vol
    if ratio < 0.25:
        return "Low"
    if ratio < 0.75:
        return "Medium"
    return "High"

def sentiment_bucket(score: float) -> tuple[str, str]:
    if pd.isna(score):
        return ("—", "warn")
    if score >= 0.20:
        return ("Positive", "ok")
    if score <= -0.20:
        return ("Negative", "bad")
    if score >= 0.05:
        return ("Slightly positive", "ok")
    if score <= -0.05:
        return ("Slightly negative", "bad")
    return ("Neutral", "warn")

def sentiment_index(score: float) -> float:
    if pd.isna(score):
        return float("nan")
    x = max(-1.0, min(1.0, float(score)))
    return (x + 1.0) * 50.0

def normalize_crypto(df: pd.DataFrame) -> pd.DataFrame:
    # IMPORTANT: do NOT drop rows aggressively; only normalize if crypto exists.
    if df.empty or "crypto" not in df.columns:
        return df
    out = df.copy()
    out["crypto"] = out["crypto"].astype(str).str.strip().str.upper()
    return out

def standardize_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the News tab resilient to different schemas.
    Accepts any of: ts_utc / timestamp / published / published_at, etc.
    Adds crypto='ALL' if missing.
    """
    if df.empty:
        return df

    out = df.copy()

    # Timestamp column normalization
    if "ts_utc" not in out.columns:
        for alt in ["timestamp_utc", "timestamp", "published", "published_at", "time", "date"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "ts_utc"})
                break

    # Ensure crypto exists
    if "crypto" not in out.columns:
        out["crypto"] = "ALL"

    # Title/source/link normalization (optional but helps UI)
    if "source" not in out.columns:
        for alt in ["publisher", "site", "domain"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "source"})
                break

    if "title" not in out.columns:
        for alt in ["headline", "name"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "title"})
                break
        if "title" not in out.columns:
            out["title"] = ""

    if "link" not in out.columns:
        for alt in ["url", "href"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "link"})
                break
        if "link" not in out.columns:
            out["link"] = ""

    # Parse timestamp if present
    if "ts_utc" in out.columns:
        out["ts_utc"] = parse_ts(out["ts_utc"])

    # Sentiment numeric if present
    if "sentiment" in out.columns:
        out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce")

    # Normalize crypto formatting
    out = normalize_crypto(out)
    return out

def compute_actual_1h_ahead(market_hist: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Returns columns: ts_utc, crypto, actual_1h_ahead
    actual_1h_ahead(t) = price(t+1)/price(t) - 1
    """
    if market_hist.empty or "ts_utc" not in market_hist.columns or "crypto" not in market_hist.columns:
        return pd.DataFrame()
    if not price_col or price_col not in market_hist.columns:
        return pd.DataFrame()

    df = market_hist[["ts_utc", "crypto", price_col]].copy()
    df = df.dropna(subset=["ts_utc", "crypto", price_col])

    # CRITICAL: convert to numeric to avoid 'str/str' errors
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    df = df.sort_values(["crypto", "ts_utc"])

    denom = df[price_col].replace(0, pd.NA)
    df["actual_1h_ahead"] = df.groupby("crypto")[price_col].shift(-1) / denom - 1.0

    return df[["ts_utc", "crypto", "actual_1h_ahead"]].copy()

def safe_row_count(path: str) -> int | None:
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        return 0
    df = read_csv_safe(path)
    return int(len(df)) if not df.empty else 0

def approx_hours_covered(df: pd.DataFrame) -> float | None:
    if df.empty or "ts_utc" not in df.columns:
        return None
    tmin = df["ts_utc"].min()
    tmax = df["ts_utc"].max()
    if pd.isna(tmin) or pd.isna(tmax):
        return None
    return (tmax - tmin).total_seconds() / 3600.0

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("### Controls")

auto_refresh = st.sidebar.toggle("Auto-refresh", value=False)
refresh_sec = st.sidebar.slider("Refresh interval (seconds)", 10, 300, 60, step=10, disabled=not auto_refresh)
if auto_refresh:
    st.markdown(f"<meta http-equiv='refresh' content='{refresh_sec}'>", unsafe_allow_html=True)

history_hours = st.sidebar.slider("History window (hours)", 24, 336, 72, step=12)

# -----------------------------
# Load data
# -----------------------------
market_hist = normalize_crypto(read_csv_safe(FILES["market_history"]))
market_latest = normalize_crypto(read_csv_safe(FILES["market_latest"]))
features_latest = normalize_crypto(read_csv_safe(FILES["features_latest"]))
pred_latest = normalize_crypto(read_csv_safe(FILES["predictions_latest"]))
pred_hist = normalize_crypto(read_csv_safe(FILES["predictions_history"]))
news_latest = standardize_news(read_csv_safe(FILES["news_latest"]))
status = read_json_safe(FILES["status"])

for df in [market_hist, market_latest, features_latest, pred_latest, pred_hist]:
    if not df.empty and "ts_utc" in df.columns:
        df["ts_utc"] = parse_ts(df["ts_utc"])

# Coins list
coins = []
if not market_hist.empty and "crypto" in market_hist.columns:
    coins = sorted([c for c in market_hist["crypto"].dropna().unique().tolist() if c and c != "ALL"])
if not coins:
    coins = DEFAULT_COINS.copy()

PRICE_COL = pick_first_col(market_hist, ["close", "price_usd", "price", "close_usd"])

# -----------------------------
# Header
# -----------------------------
topL, topM, topR = st.columns([0.58, 0.20, 0.22], vertical_alignment="center")

with topL:
    st.markdown("## Crypto Live Forecast Dashboard")
    st.caption("Live outputs from `data/live/` (market, features, predictions, news).")

with topM:
    selected_coin = st.selectbox("Asset", coins, index=0)

with topR:
    label, cls, last_run_str = status_badge(status)
    st.markdown(
        f"""
        <div style="text-align:right;">
          <div class="pill {cls}">Pipeline: <b>{label}</b></div><br/>
          <span class="muted" style="font-size:0.85rem;">Last run: {last_run_str}</span><br/>
          <span class="muted" style="font-size:0.80rem;">Now: {now_utc_str()}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# -----------------------------
# Market history window
# -----------------------------
coin_hist_full = pd.DataFrame()
if not market_hist.empty:
    coin_hist_full = market_hist[market_hist["crypto"] == selected_coin].copy() if "crypto" in market_hist.columns else market_hist.copy()

if not coin_hist_full.empty and PRICE_COL and PRICE_COL in coin_hist_full.columns:
    coin_hist_full[PRICE_COL] = pd.to_numeric(coin_hist_full[PRICE_COL], errors="coerce")

available_hours = approx_hours_covered(coin_hist_full)
available_rows = int(len(coin_hist_full)) if not coin_hist_full.empty else 0

coin_hist = coin_hist_full.copy()
effective_hours = history_hours

if not coin_hist.empty and "ts_utc" in coin_hist.columns:
    coin_hist = coin_hist.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    latest_ts = coin_hist["ts_utc"].max()
    cutoff = latest_ts - pd.Timedelta(hours=history_hours)
    coin_hist = coin_hist[coin_hist["ts_utc"] >= cutoff].copy()
    if available_hours is not None:
        effective_hours = min(history_hours, int(round(available_hours)))

# Latest rows
feat_row = features_latest[features_latest["crypto"] == selected_coin].copy() if (not features_latest.empty and "crypto" in features_latest.columns) else pd.DataFrame()
pred_row = pred_latest[pred_latest["crypto"] == selected_coin].copy() if (not pred_latest.empty and "crypto" in pred_latest.columns) else pd.DataFrame()

# KPIs from market
latest_price = float("nan")
latest_ts_str = "—"
ret_1h = float("nan")
recent_vol = float("nan")

if not coin_hist.empty and PRICE_COL and PRICE_COL in coin_hist.columns:
    latest_price = coin_hist[PRICE_COL].iloc[-1]
    if pd.notna(coin_hist["ts_utc"].iloc[-1]):
        latest_ts_str = coin_hist["ts_utc"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
    if len(coin_hist) >= 2:
        prev_price = coin_hist[PRICE_COL].iloc[-2]
        if pd.notna(prev_price) and prev_price != 0:
            ret_1h = (latest_price / prev_price) - 1.0
    tmp_ret = coin_hist[PRICE_COL].pct_change()
    if tmp_ret.notna().sum() >= 6:
        recent_vol = tmp_ret.tail(12).std()

# Prediction row
yhat = float("nan")
yhat_type = "—"
model_name = "—"
pred_ts_str = "—"

if not pred_row.empty:
    if "yhat" in pred_row.columns:
        yhat = float(pd.to_numeric(pred_row["yhat"].iloc[0], errors="coerce"))
    if "yhat_type" in pred_row.columns:
        yhat_type = str(pred_row["yhat_type"].iloc[0])
    if "model_name" in pred_row.columns:
        model_name = str(pred_row["model_name"].iloc[0])
    if "ts_utc" in pred_row.columns and pd.notna(pred_row["ts_utc"].iloc[0]):
        pred_ts_str = pred_row["ts_utc"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")

dir_label, dir_cls = direction_label(yhat)
strength = signal_strength(yhat, recent_vol)

# Feature snapshot
ma_6 = float("nan")
vol_6 = float("nan")
if not feat_row.empty:
    if "ma_6" in feat_row.columns:
        ma_6 = pd.to_numeric(feat_row["ma_6"].iloc[0], errors="coerce")
    if "vol_6" in feat_row.columns:
        vol_6 = pd.to_numeric(feat_row["vol_6"].iloc[0], errors="coerce")

# -----------------------------
# Sentiment from news
# -----------------------------
news_sent_score = float("nan")
news_sent_idx = float("nan")
news_sent_bucket, news_sent_cls = ("—", "warn")
news_count = 0
pos_pct = neu_pct = neg_pct = float("nan")

df_news = news_latest.copy() if not news_latest.empty else pd.DataFrame()
if not df_news.empty and "sentiment" in df_news.columns:
    # Keep selected + ALL
    if "crypto" in df_news.columns:
        df_coin = df_news[df_news["crypto"].isin([selected_coin, "ALL"])].copy()
    else:
        df_coin = df_news.copy()

    df_coin["sentiment"] = pd.to_numeric(df_coin["sentiment"], errors="coerce")
    df_coin = df_coin[df_coin["sentiment"].notna()].copy()

    news_count = int(len(df_coin))
    if news_count > 0:
        news_sent_score = float(df_coin["sentiment"].mean())
        news_sent_idx = sentiment_index(news_sent_score)
        news_sent_bucket, news_sent_cls = sentiment_bucket(news_sent_score)

        s = df_coin["sentiment"]
        pos_pct = float((s > 0.10).mean()) * 100.0
        neu_pct = float(((s >= -0.10) & (s <= 0.10)).mean()) * 100.0
        neg_pct = float((s < -0.10).mean()) * 100.0

# -----------------------------
# Prediction track record
# -----------------------------
pred_track = pred_hist.copy() if not pred_hist.empty else pred_latest.copy()

if not pred_track.empty:
    if "prediction" in pred_track.columns and "yhat" not in pred_track.columns:
        pred_track = pred_track.rename(columns={"prediction": "yhat"})
    if "target" in pred_track.columns and "yhat_type" not in pred_track.columns:
        pred_track = pred_track.rename(columns={"target": "yhat_type"})

actuals = compute_actual_1h_ahead(market_hist, PRICE_COL) if (PRICE_COL and not market_hist.empty) else pd.DataFrame()

pred_track_coin = pd.DataFrame()
df_eval = pd.DataFrame()

if (not pred_track.empty and not actuals.empty and "ts_utc" in pred_track.columns and "crypto" in pred_track.columns):
    merged = pred_track.merge(actuals, on=["ts_utc", "crypto"], how="left")
    merged_coin = merged[merged["crypto"] == selected_coin].copy()
    if not merged_coin.empty and "ts_utc" in merged_coin.columns:
        merged_coin = merged_coin.dropna(subset=["ts_utc"]).sort_values("ts_utc")
        tmax = merged_coin["ts_utc"].max()
        cutoff = tmax - pd.Timedelta(hours=history_hours)
        merged_coin = merged_coin[merged_coin["ts_utc"] >= cutoff].copy()

    pred_track_coin = merged_coin

    dfp = pred_track_coin.copy()
    if "yhat" in dfp.columns:
        dfp["yhat"] = pd.to_numeric(dfp["yhat"], errors="coerce")
    if "actual_1h_ahead" in dfp.columns:
        dfp["actual_1h_ahead"] = pd.to_numeric(dfp["actual_1h_ahead"], errors="coerce")

    df_eval = dfp.dropna(subset=["yhat", "actual_1h_ahead"]).copy()
    if not df_eval.empty:
        df_eval = df_eval.sort_values("ts_utc")
        df_eval["error"] = df_eval["yhat"] - df_eval["actual_1h_ahead"]
        df_eval["direction_hit"] = ((df_eval["yhat"] > 0) & (df_eval["actual_1h_ahead"] > 0)) | (
            (df_eval["yhat"] < 0) & (df_eval["actual_1h_ahead"] < 0)
        )

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_news, tab_system = st.tabs(["Overview", "News", "System"])

with tab_overview:
    k1, k2, k3, k4, k5 = st.columns(5)

    kpi(k1, f"{selected_coin} Price", f"${fmt_num(latest_price, 2)}", f"Latest: {latest_ts_str}")
    kpi(k2, "Last 1h Return", fmt_pct(ret_1h), "Observed from prices")
    kpi(k3, "Predicted Next 1h Return", fmt_pct(yhat), f"{yhat_type} • {model_name}")
    kpi(k4, "Feature Snapshot", f"MA(6): {fmt_num(ma_6, 2)}", f"Vol(6): {fmt_num(vol_6, 6)}")

    idx_val = "—" if pd.isna(news_sent_idx) else f"{news_sent_idx:.0f}/100"
    score_val = fmt_num(news_sent_score, 3)
    dist_sub = "—"
    if not (pd.isna(pos_pct) or pd.isna(neu_pct) or pd.isna(neg_pct)):
        dist_sub = f"Pos/Neu/Neg: {pos_pct:.1f}% / {neu_pct:.1f}% / {neg_pct:.1f}%"
    kpi(
        k5,
        "Market Sentiment (latest batch)",
        idx_val,
        f"{news_sent_bucket} • Score: {score_val} • Items: {news_count} • {dist_sub}",
    )

    st.markdown("")
    arrow = "▲" if dir_label == "UP" else ("▼" if dir_label == "DOWN" else "•")

    st.markdown(
        f"""
        <div class="callout">
          <div class="callout-title">
            <span class="big-arrow {dir_cls}">{arrow}</span>
            Next-hour direction: <span class="{dir_cls}"><b>{dir_label}</b></span>
            <span class="pill" style="margin-left:10px;">Signal: <b>{strength}</b></span>
            <span class="pill" style="margin-left:10px;">Sentiment index: <b class="{news_sent_cls}">{idx_val}</b></span>
          </div>
          <div class="callout-body">
            Sentiment score: <b>{fmt_num(news_sent_score, 3)}</b> (scale −1 to +1). Prediction timestamp: {pred_ts_str}.
          </div>
          <div class="callout-body" style="margin-top:6px;">
            Returns are decimals (e.g., 0.001 = 0.1%).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    c1, c2 = st.columns([0.62, 0.38])
    with c1:
        st.markdown("### Price history")

        if available_hours is not None:
            st.caption(f"Window requested: {history_hours}h • Data available: {available_hours:.1f}h • Rows plotted: {len(coin_hist)}")
        else:
            st.caption(f"Window requested: {history_hours}h • Rows plotted: {len(coin_hist)}")

        if coin_hist.empty or "ts_utc" not in coin_hist.columns or not PRICE_COL or PRICE_COL not in coin_hist.columns:
            st.info("Price chart not available yet.")
        else:
            fig = px.line(coin_hist, x="ts_utc", y=PRICE_COL, title=f"{selected_coin} price (showing ~{effective_hours}h)")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Observed returns")
        if coin_hist.empty or "ts_utc" not in coin_hist.columns or not PRICE_COL or PRICE_COL not in coin_hist.columns:
            st.info("Returns chart not available yet.")
        else:
            tmp = coin_hist[["ts_utc", PRICE_COL]].copy()
            tmp["return_1h"] = tmp[PRICE_COL].pct_change()
            fig2 = px.line(tmp, x="ts_utc", y="return_1h", title="1h returns (observed)")
            fig2.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Prediction track record")

    pred_hist_rows = safe_row_count(FILES["predictions_history"]) or 0
    pred_latest_rows = safe_row_count(FILES["predictions_latest"]) or 0
    points_in_window = int(len(pred_track_coin)) if not pred_track_coin.empty else 0

    st.caption(
        f"Predictions logged: {pred_hist_rows} rows • Latest batch: {pred_latest_rows} rows • Points in window: {points_in_window} • Evaluated: {len(df_eval) if not df_eval.empty else 0}"
    )

    if df_eval.empty:
        st.info("Evaluation will appear after the next hourly close is recorded (requires one additional hour of market history).")
    else:
        hit_rate = df_eval["direction_hit"].mean() if len(df_eval) > 0 else float("nan")
        mae = df_eval["error"].abs().mean() if len(df_eval) > 0 else float("nan")

        m1, m2, m3 = st.columns(3)
        kpi(m1, "Direction hit rate", f"{fmt_num(hit_rate*100, 1)}%", "Same sign (prediction vs realized)")
        kpi(m2, "Mean absolute error", fmt_num(mae, 6), "MAE on returns")
        kpi(m3, "Evaluated points", f"{len(df_eval)}", "Aligned rows")

        c3, c4 = st.columns([0.62, 0.38])
        with c3:
            figp = px.line(df_eval, x="ts_utc", y=["yhat", "actual_1h_ahead"], title="Predicted vs realized next-hour return")
            figp.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), legend_title_text="")
            st.plotly_chart(figp, use_container_width=True)

        with c4:
            fige = px.line(df_eval, x="ts_utc", y="error", title="Prediction error (prediction - realized)")
            fige.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fige, use_container_width=True)

        show_cols = [c for c in ["ts_utc", "yhat", "actual_1h_ahead", "error", "model_name", "yhat_type"] if c in df_eval.columns]
        st.dataframe(df_eval.sort_values("ts_utc", ascending=False)[show_cols].head(20), use_container_width=True, hide_index=True)

with tab_news:
    st.markdown("### Latest news")

    if news_latest.empty:
        st.info("No news found yet. (This usually means data/live/news_latest.csv is empty, missing, or written with a different schema.)")
    else:
        df = news_latest.copy()

        f1, f2, f3 = st.columns([0.25, 0.25, 0.50])
        with f1:
            asset_filter = st.selectbox("Asset filter", ["All", "Selected asset only"], index=1)
        with f2:
            sentiment_filter = st.selectbox("Sentiment filter", ["All", "Non-zero only"], index=0)
        with f3:
            text_search = st.text_input("Search title/source", value="", placeholder="Type keywords…")

        if "link" in df.columns and df["link"].astype(str).str.len().sum() > 0:
            df = df.drop_duplicates(subset=["link"], keep="first")

        if "ts_utc" in df.columns:
            df = df.sort_values("ts_utc", ascending=False)

        if asset_filter == "Selected asset only" and "crypto" in df.columns:
            df = df[df["crypto"].isin([selected_coin, "ALL"])].copy()

        if sentiment_filter == "Non-zero only" and "sentiment" in df.columns:
            df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
            df = df[df["sentiment"].fillna(0) != 0].copy()

        if text_search.strip():
            q = text_search.strip().lower()
            mask = False
            for c in [c for c in ["title", "source"] if c in df.columns]:
                mask = mask | df[c].fillna("").astype(str).str.lower().str.contains(q)
            df = df[mask].copy() if isinstance(mask, pd.Series) else df

        sent_score = float("nan")
        sent_idx = float("nan")
        sent_lbl, sent_cls = ("—", "warn")
        pos2 = neu2 = neg2 = float("nan")

        if "sentiment" in df.columns:
            s = pd.to_numeric(df["sentiment"], errors="coerce")
            if s.notna().sum() > 0:
                sent_score = float(s.mean())
                sent_idx = sentiment_index(sent_score)
                sent_lbl, sent_cls = sentiment_bucket(sent_score)
                pos2 = float((s > 0.10).mean()) * 100.0
                neu2 = float(((s >= -0.10) & (s <= 0.10)).mean()) * 100.0
                neg2 = float((s < -0.10).mean()) * 100.0

        s1, s2, s3 = st.columns(3)
        kpi(s1, "Items shown", f"{len(df)}", "After filters")
        kpi(
            s2,
            "Sentiment (filtered)",
            ("—" if pd.isna(sent_idx) else f"{sent_idx:.0f}/100"),
            f"{sent_lbl} • Score: {fmt_num(sent_score, 3)} • Scale [-1, +1]",
        )
        if not (pd.isna(pos2) or pd.isna(neu2) or pd.isna(neg2)):
            kpi(s3, "Pos / Neu / Neg", f"{pos2:.1f}% / {neu2:.1f}% / {neg2:.1f}%", "Distribution")
        else:
            kpi(s3, "Pos / Neu / Neg", "—", "Sentiment not available")

        st.markdown("---")

        df_show = df.copy()
        df_show["open"] = df_show["link"] if "link" in df_show.columns else ""

        cols = [c for c in ["ts_utc", "crypto", "title", "source", "sentiment", "open"] if c in df_show.columns]

        column_config = {}
        if "open" in cols:
            column_config["open"] = st.column_config.LinkColumn("open", help="Open article", display_text="open")
        if "sentiment" in cols:
            column_config["sentiment"] = st.column_config.NumberColumn("sentiment", help="Sentiment score (scale -1 to +1)", format="%.4f")
        if "ts_utc" in cols:
            column_config["ts_utc"] = st.column_config.DatetimeColumn("ts_utc")

        st.data_editor(
            df_show[cols].head(120),
            use_container_width=True,
            hide_index=True,
            disabled=True,
            column_config=column_config,
        )

        if "sentiment" in df_show.columns and df_show["sentiment"].notna().sum() > 0:
            st.markdown("---")
            st.markdown("### Sentiment distribution (latest batch)")
            fig3 = px.histogram(df_show.dropna(subset=["sentiment"]), x="sentiment", nbins=20, title="Sentiment histogram")
            fig3.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig3, use_container_width=True)

with tab_system:
    st.markdown("### System")
    st.caption("Operational summary based on the latest pipeline outputs.")

    if available_hours is not None:
        st.markdown(
            f"History available for **{selected_coin}**: **{available_hours:.1f}h** • Total rows: **{available_rows}** • Rows plotted: **{len(coin_hist)}**"
        )
    else:
        st.markdown(f"Rows plotted: **{len(coin_hist)}**")

    ts_val = status.get("ts_utc") or status.get("updated_at_utc") or "—"

    mh = status.get("market_history_rows", None) or safe_row_count(FILES["market_history"])
    ml = status.get("market_latest_rows", None) or safe_row_count(FILES["market_latest"])
    pr_latest = status.get("predictions_latest_rows", None) or safe_row_count(FILES["predictions_latest"])
    pr_hist = status.get("predictions_history_rows", None) or safe_row_count(FILES["predictions_history"])
    nl = status.get("news_latest_rows", None) or safe_row_count(FILES["news_latest"])

    st.markdown(
        f"""
        <div class="callout">
          <div class="callout-title">Latest update</div>
          <div class="callout-body">Timestamp: <b>{ts_val}</b></div>
          <div class="callout-body">Market latest rows: <b>{ml if ml is not None else "—"}</b></div>
          <div class="callout-body">Market history rows: <b>{mh if mh is not None else "—"}</b></div>
          <div class="callout-body">Predictions latest rows: <b>{pr_latest if pr_latest is not None else "—"}</b></div>
          <div class="callout-body">Predictions history rows: <b>{pr_hist if pr_hist is not None else "—"}</b></div>
          <div class="callout-body">News latest rows: <b>{nl if nl is not None else "—"}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Data sources", expanded=False):
        st.markdown("The dashboard reads the following live outputs:")
        for k, p in FILES.items():
            rel = os.path.relpath(p, BASE_DIR)
            st.code(f"{k}: {rel}")

st.markdown(
    "<div class='small-note'>For research demonstration only. Not financial advice.</div>",
    unsafe_allow_html=True,
)
