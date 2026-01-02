
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "final_regression_model.pkl")

# Minimal feature set that we know exists from live_features.csv
FEATURES = ["price_usd", "pct_change_24h", "news_count", "news_sent_mean", "news_sent_std"]

def main():
    base = os.path.join("data", "live")
    feat_path = os.path.join(base, "live_features.csv")
    out_path = os.path.join(base, "live_predictions.csv")

    df = pd.read_csv(feat_path)

    if not os.path.exists(MODEL_PATH):
        # fallback: no model yet in repo
        df["y_pred_return_1d"] = None
        df["note"] = "Model not found in repo (models/final_regression_model.pkl)"
        df.to_csv(out_path, index=False)
        print("Saved live_predictions.csv (no model)")
        return

    bundle = joblib.load(MODEL_PATH)

    model = bundle.get("model", bundle)  # support either raw model or bundle
    scaler = bundle.get("scaler", None)
    feature_cols = bundle.get("features", FEATURES)

    X = df.copy()
    # ensure all feature columns exist
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0

    X = X[feature_cols]

    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict(X)
    df["y_pred_return_1d"] = preds
    df["note"] = "ok"
    df.to_csv(out_path, index=False)
    print("Saved live_predictions.csv (with model)")

if __name__ == "__main__":
    main()
