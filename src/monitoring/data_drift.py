import pandas as pd
from evidently.metrics import DataDriftPreset
from evidently.report import Report

REFERENCE_PATH = "data/processed/features.csv"
NEW_PATH = "data/processed/features.csv" 

reference_df = pd.read_csv(REFERENCE_PATH)
new_df = pd.read_csv(NEW_PATH)


reference_df = reference_df.select_dtypes(include=["number"])
new_df = new_df.select_dtypes(include=["number"])


report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=new_df)

report.save_html("reports/data_drift.html")
print("✅ Data drift report saved at reports/data_drift.html")
