# scripts/train_hourly_model.py
from __future__ import annotations

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

LIVE_DIR = os.path.join("data", "live")
MODELS_DIR = "models"

HISTORY_PATH = os.path.join(LIVE_DIR, "market_history.csv")
OUT_PKL = os.path.join(MODELS_DIR, "final_regression_model.pkl")

# Cold-start features: only need a few hours of history
FEATURE_COLS = [
    "price_usd",
    "return_1h",
    "ma_3",
    "ma_6",
    "vol_6",
]

TARGET = "return_1h_ahead"

def main():
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"Missing {HISTORY_PATH}. Run live pipeline first to build history.")

    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(HISTORY_PATH)

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df = df.dropna(subset=["ts_utc", "crypto", "price_usd"]).copy()
    df = df.sort_values(["crypto", "ts_utc"])

    feats = []
    for crypto, g in df.groupby("crypto"):
        g = g.copy()

        g["return_1h"] = g["price_usd"].pct_change(1)

        g["ma_3"] = g["price_usd"].rolling(3).mean()
        g["ma_6"] = g["price_usd"].rolling(6).mean()

        g["vol_6"] = g["return_1h"].rolling(6).std()

        g["return_1h_ahead"] = g["return_1h"].shift(-1)

        feats.append(g)

    data = pd.concat(feats, ignore_index=True)

    keep_cols = ["ts_utc", "crypto"] + FEATURE_COLS + [TARGET]
    data = data[keep_cols].dropna(subset=FEATURE_COLS + [TARGET]).copy()

    # Need some minimum rows; 60+ is enough to start; more is better
    if len(data) < 60:
        raise RuntimeError(
            f"Not enough training rows ({len(data)}). "
            "Let the hourly workflow run longer (aim 8–12 hours+) then try again."
        )

    X = data[FEATURE_COLS].astype(float).values
    y = data[TARGET].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    bundle = {
        "model": model,
        "scaler": scaler,
        "features": FEATURE_COLS,
        "target": TARGET,
        "model_name": "hourly_ridge_coldstart_v1",
        "test_metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    joblib.dump(bundle, OUT_PKL)
    print("Saved:", OUT_PKL)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
