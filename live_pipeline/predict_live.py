# live_pipeline/predict_live.py
from __future__ import annotations

import os
import joblib
import pandas as pd

from .utils_io import atomic_write_csv, utc_now

LIVE_DIR = os.path.join("data", "live")
MODEL_PATH = os.path.join("models", "final_regression_model.pkl")

FEATURES_FALLBACK = ["price_usd", "return_1h", "ma_6", "vol_24"]  # safe defaults


def main():
    feat_path = os.path.join(LIVE_DIR, "features_latest.csv")
    out_path = os.path.join(LIVE_DIR, "predictions_latest.csv")

    if not os.path.exists(feat_path):
        # schema-safe fallback
        out = pd.DataFrame(columns=["ts_utc", "crypto", "yhat", "yhat_type", "model_name", "generated_at_utc", "note"])
        atomic_write_csv(out, out_path)
        print("features_latest.csv not found; wrote empty predictions_latest.csv")
        return

    df = pd.read_csv(feat_path)

    # Ensure keys exist
    for c in ["ts_utc", "crypto"]:
        if c not in df.columns:
            df[c] = pd.NA

    if df.empty:
        out = pd.DataFrame(columns=["ts_utc", "crypto", "yhat", "yhat_type", "model_name", "generated_at_utc", "note"])
        atomic_write_csv(out, out_path)
        print("features_latest.csv empty; wrote empty predictions_latest.csv")
        return

    generated_at = utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")

    if not os.path.exists(MODEL_PATH):
        out = df[["ts_utc", "crypto"]].copy()
        out["yhat"] = pd.NA
        out["yhat_type"] = "return_1h"
        out["model_name"] = "none"
        out["generated_at_utc"] = generated_at
        out["note"] = "Model not found in models/final_regression_model.pkl"
        atomic_write_csv(out, out_path)
        print("Model not found; wrote predictions_latest.csv (no model)")
        return

    bundle = joblib.load(MODEL_PATH)
    model = bundle.get("model", bundle)
    scaler = bundle.get("scaler", None)
    feature_cols = bundle.get("features", FEATURES_FALLBACK)
    model_name = bundle.get("model_name", "final_regression_model")

    X = df.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols]

    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict(X)

    out = df[["ts_utc", "crypto"]].copy()
    out["yhat"] = preds
    out["yhat_type"] = "return_1h"
    out["model_name"] = model_name
    out["generated_at_utc"] = generated_at
    out["note"] = "ok"
    atomic_write_csv(out, out_path)
    print("Saved predictions_latest.csv")


if __name__ == "__main__":
    main()
