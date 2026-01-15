from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

def evaluate_classification(y_true, y_pred, y_score):
    print("ROC-AUC:", roc_auc_score(y_true, y_score))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))