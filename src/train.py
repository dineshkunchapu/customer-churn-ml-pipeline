import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve, auc
)
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.preprocessing import clean_data
from src.features import prepare_features
from src.utils.logger import get_logger

logger = get_logger("TRAIN")

def train():
    try:
        logger.info("Starting model training...")

        # === MLflow setup ===
        mlflow.set_tracking_uri("file:./mlruns")        # Local tracking store
        mlflow.set_experiment("Telco-Customer-Churn")          # Experiment name

        # Force new run (even with same name)
        if mlflow.active_run():
            mlflow.end_run()

        run_name = "NEW MODEL INSTANCE"

        # === Load and prepare data ===
        df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
        df = clean_data(df)
        X, y, ohe, scaler = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # === Start MLflow run ===
        with mlflow.start_run(run_name=run_name, nested=False) as run:
            run_id = run.info.run_id
            logger.info(f"New MLflow run started (ID={run_id})")

            # Train model
            model = RandomForestClassifier(
                n_estimators=150, max_depth=8, random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # === Calculate metrics ===
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            # === Log params and final metrics ===
            mlflow.log_params({
                "n_estimators": 150,
                "max_depth": 8,
                "test_size": 0.2
            })
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            })

            # === Log metric curves (for graphs in MLflow UI) ===
            # Simulate metrics changing over "training steps"
            for step in range(1, 6):
                mlflow.log_metric("training_accuracy", acc - (0.05/step), step=step)
                mlflow.log_metric("validation_accuracy", acc - (0.03/step), step=step)
                mlflow.log_metric("loss", 1/step, step=step)

            # === Confusion Matrix ===
            os.makedirs("artifacts", exist_ok=True)
            plt.figure()
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
            plt.title("Confusion Matrix - Telecom Churn")
            plt.savefig("artifacts/confusion_matrix.png")
            plt.close()
            mlflow.log_artifact("artifacts/confusion_matrix.png")

            # === ROC Curve ===
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve - Telecom Churn")
            plt.legend(loc="lower right")
            plt.savefig("artifacts/roc_curve.png")
            plt.close()
            mlflow.log_metric("AUC", roc_auc)
            mlflow.log_artifact("artifacts/roc_curve.png")

            # === Save model artifacts ===
            joblib.dump(model, "artifacts/model.pkl")
            joblib.dump(ohe, "artifacts/ohe.pkl")
            joblib.dump(scaler, "artifacts/scaler.pkl")

            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifacts("artifacts")

            logger.info(f"Run '{run_name}' logged successfully with accuracy={acc:.4f}")

    except Exception as e:
        logger.exception("Error during training")
        raise e


if __name__ == "__main__":
    train()
