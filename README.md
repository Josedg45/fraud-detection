# Fraud Detection Pipeline

[![tests](https://github.com/Josedg45/fraud-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Josedg45/fraud-detection/actions/workflows/tests.yml)

End-to-end machine learning system for detecting fraudulent transactions: data validation contracts, feature engineering, experiment tracking, a model registry, a scoring API and drift monitoring.

## Architecture

```mermaid
graph TD
    A[Raw Data: Kaggle CSV] --> B[Ingestion & Validation: Pandera]
    B --> C[Exploratory Data Analysis]
    C --> D[Feature Engineering]
    D --> E[Training & Tracking: MLflow]
    E --> F[Model Registry: fraud_lightgbm@staging]
    F --> G[Scoring API: FastAPI]
    F --> H[Drift Monitoring: Evidently]
```

## Key features

- **Data validation:** schema enforcement with Pandera catches data issues before training.
- **Experiment tracking:** parameters, metrics (ROC-AUC, PR-AUC) and models logged to MLflow, with an input signature.
- **Model registry:** each training run registers a new version and points the `staging` alias at it.
- **Scoring API:** FastAPI endpoint that returns a fraud score and an `ALLOW` / `CHALLENGE` / `BLOCK` decision.
- **Drift monitoring:** HTML reports for data drift and prediction drift with Evidently.
- **Tests and CI:** pytest suite that runs on every push and pull request.

## Project structure

```
src/
  config.py        # tracking URI, model name/alias, feature order
  ingestion/       # load the Kaggle CSV, map it to a transaction schema, validate it
  features/        # temporal and per-user rolling features
  models/          # baseline, LightGBM and stacking trainers, metrics
  serving/         # FastAPI app, decision rule, local prediction script
  monitoring/      # data and prediction drift reports
notebooks/         # EDA and feature design
tests/
```

## Quick start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### 2. Data

Download `creditcard.csv` from the Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset and put it in `data/raw/`. The data is not versioned in this repository.

### 3. Run the pipeline

Run every command from the repository root.

```bash
# Build features -> data/processed/features.csv
python -m src.features.run_pipeline

# Train, log to MLflow and register the model as fraud_lightgbm@staging
python -m src.models.train_lgbm

# Start the scoring API
uvicorn src.serving.app:app --reload

# Drift report -> monitoring/prediction_drift.html
python -m src.monitoring.prediction_drift
```

MLflow stores runs in a local `mlflow.db` by default. Set `MLFLOW_TRACKING_URI` to use a remote tracking server.

### 4. Tests

```bash
pytest -q
```

## API usage

`POST /predict`

```json
{
  "amount": 8000.0,
  "hour": 14,
  "day_of_week": 3,
  "txn_count_1h": 5,
  "txn_sum_24h": 12,
  "user_id": 102,
  "device_id": 5,
  "transaction_id": 999
}
```

Response:

```json
{ "fraud_score": 0.07, "decision": "ALLOW" }
```

## Results

| Model    | ROC-AUC | PR-AUC |
|----------|---------|--------|
| LightGBM | 0.7607  | 0.0478 |

The Kaggle dataset is highly imbalanced (about 0.17% fraud), so PR-AUC is the more informative metric.
