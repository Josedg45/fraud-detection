import pandas as pd
import mlflow.pyfunc
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
import os

REFERENCE_PATH = "data/processed/features.csv"
NEW_PATH = "data/processed/features.csv"
OUTPUT_HTML = "monitoring/prediction_drift.html"

MODEL_NAME = "fraud_lightgbm"
ALIAS = "staging"

def run_prediction_drift():
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)

    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    reference_df = pd.read_csv(REFERENCE_PATH)
    new_df = pd.read_csv(NEW_PATH)

    all_numeric = reference_df.select_dtypes(include=["number"]).columns.tolist()
    
    # Solo quitamos 'label'. Mantendremos 'transaction_id' para llegar a 8.
    # Si sigue fallando con 9, entonces el que sobra es 'user_id' o 'device_id'.
    to_exclude = ["label"] 
    features = [col for col in all_numeric if col not in to_exclude]

    ref_input = reference_df[features].copy()
    new_input = new_df[features].copy()

    if len(ref_input) > 5000:
        ref_input = ref_input.sample(5000, random_state=42)
        new_input = new_df[features].sample(5000, random_state=42)

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