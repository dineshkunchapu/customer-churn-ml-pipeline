import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.utils.logger import get_logger

logger = get_logger("EVALUATE")

def evaluate_model():
    logger.info("Starting model evaluation...")

    # Load model and data
    model = joblib.load("artifacts/model.pkl")
    X = pd.read_csv("data/processed/features.csv")
    y = pd.read_csv("data/processed/target.csv").squeeze()

    preds = model.predict(X)

    # Compute metrics
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)

    report = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4)
    }

    # Save metrics to JSON for DVC tracking
    with open("artifacts/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Evaluation complete: {report}")
    return report


if __name__ == "__main__":
    evaluate_model()
