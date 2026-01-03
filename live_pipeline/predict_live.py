import os
import json
from datetime import datetime, timezone

import joblib
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_model_bundle(model_path: str):
    obj = joblib.load(model_path)
    # supports: bundle dict OR raw sklearn model
    if isinstance(obj, dict):
        model = obj.get("model", None)
        features = obj.get("features", None)
        model_name = obj.get("model_name", "model")
        target = obj.get("target", "return_1h_ahead")
        scaler = obj.get("scaler", None)
        return model, scaler, features, model_name, target
    return obj, None, None, "model", "return_1h_ahead"


def apply_scaler_if_needed(X: pd.DataFrame, scaler):
    if scaler is None:
        return X
    # scaler may be sklearn transformer expecting numpy array
    try:
        Xs = scaler.transform(X.values)
        return pd.DataFrame(Xs, columns=X.columns, index=X.index)
    except Exception:
        return X


def append_history(history_path: str, df_new: pd.DataFrame) -> pd.DataFrame:
    if df_new.empty:
        return pd.DataFrame()

    df_new = df_new.copy()
    if "ts_utc" in df_new.columns:
        df_new["ts_utc"] = pd.to_datetime(df_new["ts_utc"], errors="coerce", utc=True)

    df_old = read_csv_safe(history_path)
    if not df_old.empty and "ts_utc" in df_old.columns:
        df_old["ts_utc"] = pd.to_datetime(df_old["ts_utc"], errors="coerce", utc=True)

    if df_old.empty:
        df_all = df_new
    else:
        df_all = pd.concat([df_old, df_new], ignore_index=True)

    # drop duplicates (keep latest)
    keys = [c for c in ["ts_utc", "crypto", "model_name", "yhat_type"] if c in df_all.columns]
    if keys:
        df_all = df_all.drop_duplicates(subset=keys, keep="last")

    df_all = df_all.sort_values(["crypto", "ts_utc"]) if ("crypto" in df_all.columns and "ts_utc" in df_all.columns) else df_all
    df_all.to_csv(history_path, index=False)
    return df_all


def main():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    LIVE_DIR = os.path.join(ROOT, "data", "live")
    MODELS_DIR = os.path.join(ROOT, "models")

    ensure_dir(LIVE_DIR)

    features_path = os.path.join(LIVE_DIR, "features_latest.csv")
    pred_latest_path = os.path.join(LIVE_DIR, "predictions_latest.csv")
    pred_hist_path = os.path.join(LIVE_DIR, "predictions_history.csv")
    status_path = os.path.join(LIVE_DIR, "status.json")

    model_path = os.path.join(MODELS_DIR, "final_regression_model.pkl")

    features_df = read_csv_safe(features_path)
    if features_df.empty:
        # still write a status so the dashboard can show it
        status = {
            "ok": False,
            "stage": "predict",
            "message": "features_latest.csv not found or empty",
            "updated_at_utc": utc_now_iso(),
        }
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        return

    # timestamp parsing
    if "ts_utc" in features_df.columns:
        features_df["ts_utc"] = pd.to_datetime(features_df["ts_utc"], errors="coerce", utc=True)

    model, scaler, feature_cols, model_name, target = load_model_bundle(model_path)

    # Determine input columns
    protected = {"ts_utc", "crypto"}
    if feature_cols and isinstance(feature_cols, list) and len(feature_cols) > 0:
        X = features_df.copy()
        missing = [c for c in feature_cols if c not in X.columns]
        if missing:
            # If missing schema, fallback to numeric cols (safe mode)
            num_cols = [c for c in X.columns if c not in protected and pd.api.types.is_numeric_dtype(X[c])]
            X = X[num_cols].copy()
        else:
            X = X[feature_cols].copy()
    else:
        # fallback: all numeric columns excluding identifiers
        num_cols = [c for c in features_df.columns if c not in protected and pd.api.types.is_numeric_dtype(features_df[c])]
        X = features_df[num_cols].copy()

    # fill NAs defensively
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(0.0)

    Xs = apply_scaler_if_needed(X, scaler)

    # Predict
    try:
        yhat = model.predict(Xs.values)
    except Exception:
        # some models expect df not numpy
        yhat = model.predict(Xs)

    out = pd.DataFrame(
        {
            "ts_utc": features_df["ts_utc"] if "ts_utc" in features_df.columns else pd.NaT,
            "crypto": features_df["crypto"] if "crypto" in features_df.columns else "UNKNOWN",
            "yhat": yhat,
            "yhat_type": str(target),
            "model_name": str(model_name),
        }
    )

    # Ensure correct types
    out["yhat"] = pd.to_numeric(out["yhat"], errors="coerce")
    if "ts_utc" in out.columns:
        out["ts_utc"] = pd.to_datetime(out["ts_utc"], errors="coerce", utc=True)

    # Save latest
    out.to_csv(pred_latest_path, index=False)

    # Append history
    hist = append_history(pred_hist_path, out)

    # Status
    status = {
        "ok": True,
        "stage": "predict",
        "updated_at_utc": utc_now_iso(),
        "predictions_latest_rows": int(len(out)),
        "predictions_history_rows": int(len(hist)) if isinstance(hist, pd.DataFrame) else None,
        "model_name": str(model_name),
        "target": str(target),
    }
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()
