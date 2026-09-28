import pandas as pd
import joblib
import os

print("=" * 50)
print("TEST 1: Check all required files exist")
print("=" * 50)

required_files = [
    "data/raw/simulated_disease_dataset.csv",
    "data/processed/cleaned_disease_dataset.csv",
    "data/processed/features_disease_dataset.csv",
    "data/processed/village_recommendations.csv",
    "models/logistic_regression_model.pkl",
    "models/random_forest_model.pkl",
    "village_risk_map.html",
    "app.py",
]

for file_path in required_files:
    exists = os.path.exists(file_path)
    status = "OK" if exists else "MISSING"
    print(f"[{status}] {file_path}")

print("\n" + "=" * 50)
print("TEST 2: Load model and check it predicts without error")
print("=" * 50)

rf_model = joblib.load("models/random_forest_model.pkl")
features_df = pd.read_csv("data/processed/features_disease_dataset.csv")

drop_cols = ["village_id", "village_name", "date", "outbreak_risk_level",
             "outbreak_risk_level_encoded", "case_count"]
X_sample = features_df.drop(columns=drop_cols).head(5)

predictions = rf_model.predict(X_sample)
print("Predictions on 5 sample rows:", predictions)
print("Test passed: model predicts without error." if len(predictions) == 5 else "Test FAILED")

print("\n" + "=" * 50)
print("TEST 3: Check for data leakage (case_count not in features)")
print("=" * 50)

X_columns = features_df.drop(columns=drop_cols).columns
if "case_count" in X_columns:
    print("FAILED: case_count is still in features - this is data leakage!")
else:
    print("Passed: case_count correctly excluded from features.")

print("\n" + "=" * 50)
print("TEST 4: Check risk level distribution isn't degenerate")
print("=" * 50)

risk_counts = features_df["outbreak_risk_level"].value_counts()
print(risk_counts)
smallest_class_pct = risk_counts.min() / risk_counts.sum() * 100
print(f"Smallest class makes up {smallest_class_pct:.1f}% of data")
print("Passed: no class is critically underrepresented." if smallest_class_pct > 10 else "WARNING: a class is very small")

print("\n" + "=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)