from pathlib import Path
from src.ingestion.load_kaggle import load_data
from src.features.pipeline import build_features

RAW_PATH = "data/raw/creditcard.csv"
OUT_PATH = "data/processed/features.csv"

def run():
    df = load_data(RAW_PATH)
    df_features = build_features(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_features.to_csv(OUT_PATH, index=False)

    print(f"Feature pipeline executed successfully → {OUT_PATH}")


if __name__ == "__main__":
    run()
