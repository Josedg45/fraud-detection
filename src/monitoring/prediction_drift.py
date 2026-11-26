import pandas as pd
import mlflow
import mlflow.pyfunc
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
import os

from src.config import FEATURES, MODEL_URI, TRACKING_URI

REFERENCE_PATH = "data/processed/features.csv"
NEW_PATH = "data/processed/features.csv"
OUTPUT_HTML = "monitoring/prediction_drift.html"

def run_prediction_drift():
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)

    reference_df = pd.read_csv(REFERENCE_PATH)
    new_df = pd.read_csv(NEW_PATH)

    features = FEATURES

    ref_input = reference_df[features].astype("float64")
    new_input = new_df[features].astype("float64")

    if len(ref_input) > 5000:
        ref_input = ref_input.sample(5000, random_state=42)
        new_input = new_input.sample(5000, random_state=42)

    print(f"Features para predicción ({len(features)}): {features}")
    
    # Sincronizamos los dataframes originales con la muestra para el reporte
    reference_df = ref_input.copy()
    new_df = new_input.copy()
    
    reference_df["prediction"] = model.predict(ref_input)
    new_df["prediction"] = model.predict(new_input)

    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference_df, current_data=new_df)
    report.save_html(OUTPUT_HTML)

    print("Analysis complete.")

if __name__ == "__main__":
    run_prediction_drift()