# live_pipeline/predict_live.py
from __future__ import annotations

import os
import time
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join("models", "final_regression_model.pkl")

def main():
    base = os.path.join("data", "live")
    feat_path = os.path.join(base, "features_latest.csv")
    out_path = os.path.join(base, "predictions_latest.csv")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing {feat_path}. Run build_live_features first.")

    df = pd.read_csv(feat_path)

    # Ensure minimal id columns exist
    for c in ["ts_utc", "crypto"]:
        if c not in df.columns:
            df[c] = ""

    if not os.path.exists(MODEL_PATH):
        out = df[["ts_utc", "crypto"]].copy()
        out["yhat"] = np.nan
        out["yhat_type"] = "return_1h"
        out["model_name"] = "none"
        out["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out["note"] = f"Model not found in {MODEL_PATH}"
        out.to_csv(out_path, index=False)
        print("Saved predictions_latest.csv (no model)")
        return

    bundle = joblib.load(MODEL_PATH)

    # Support either raw model or bundle dict
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle)
        scaler = bundle.get("scaler", None)
        feature_cols = bundle.get("features", [])
        target = bundle.get("target", "return_1h_ahead")
        model_name = bundle.get("model_name", "best_model")
    else:
        model = bundle
        scaler = None
        feature_cols = []
        target = "unknown"
        model_name = "raw_model"

    if not feature_cols:
        raise RuntimeError(
            "Your model PKL does not include 'features'. "
            "Re-export model bundle with a feature list, or commit the JSON and hardcode feature order."
        )

    X = df.copy()

    # Ensure all required feature columns exist; fill missing with 0
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0

    # Convert to numeric & fill NaNs
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[feature_cols] = X[feature_cols].fillna(0.0)

    X_mat = X[feature_cols].values

    if scaler is not None:
        X_mat = scaler.transform(X_mat)

    preds = model.predict(X_mat)

    out = df[["ts_utc", "crypto"]].copy()
    out["yhat"] = preds
    out["yhat_type"] = target
    out["model_name"] = model_name
    out["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["note"] = "ok"
    out.to_csv(out_path, index=False)
    print("Saved predictions_latest.csv")

if __name__ == "__main__":
    main()
