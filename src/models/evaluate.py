import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.train_baseline import train
from src.models.metrics import evaluate_classification

def run_evaluation():
    model =  train()

    df = pd.read_csv('data/processed/features.csv')
    X = df.drop(columns=['Class'])
    y = df['Class']

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    y_score = model.predict_proba(X_test)[:,1]
    y_pred = (y_score > 0.5).astype(int)

    evaluate_classification(y_test, y_pred, y_score)


if __name__ == '__main__':
    run_evaluation()