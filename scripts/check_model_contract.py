# scripts/check_model_contract.py
import os
import joblib
import pandas as pd

MODEL_PKL = os.path.join("models", "final_regression_model.pkl")
FEATURES_CSV = os.path.join("data", "live", "features_latest.csv")

def main():
    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(f"Missing {MODEL_PKL}")
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Missing {FEATURES_CSV}. Run build_live_features first.")

    bundle = joblib.load(MODEL_PKL)
    print("Loaded PKL type:", type(bundle))

    if isinstance(bundle, dict):
        print("PKL dict keys:", sorted(bundle.keys()))
        feature_cols = bundle.get("features") or bundle.get("feature_cols")
        target = bundle.get("target")
        model_name = bundle.get("model_name")
        print("model_name:", model_name)
        print("target:", target)
        print("features exists:", feature_cols is not None)
        if feature_cols is not None:
            print("feature count:", len(feature_cols))
            print("first 20 features:", feature_cols[:20])
    else:
        print("PKL is a raw model (no dict). No feature names stored.")
        feature_cols = None

    df = pd.read_csv(FEATURES_CSV)
    print("\nLive columns:", list(df.columns))
    print("\nSample live rows:")
    print(df.head(3).to_string(index=False))

    if not feature_cols:
        print("\nCannot compare columns because PKL does not include feature names.")
        print("Fix: commit final_regression_model.json OR re-save the model bundle with 'features' included.")
        return

    have = set(df.columns)
    missing = [c for c in feature_cols if c not in have]
    extra = [c for c in df.columns if c not in feature_cols]

    print("\nMissing vs model:", len(missing))
    if missing:
        print("Missing (first 30):")
        for c in missing[:30]:
            print(c)

    print("\nExtra vs model:", len(extra))
    if extra:
        print("Extra (first 30):")
        for c in extra[:30]:
            print(c)

if __name__ == "__main__":
    main()
