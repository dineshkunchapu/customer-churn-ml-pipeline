from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from src.utils.logger import get_logger
import sklearn

logger = get_logger("FEATURES")

def prepare_features(df):
    try:
        logger.info("Starting feature preparation...")
        df = df.copy()
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        categorical = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]
        numerical = ["tenure", "MonthlyCharges", "TotalCharges"]

        # ✅ Handle sklearn version differences automatically
        skl_major, skl_minor = map(int, sklearn.__version__.split(".")[:2])
        if skl_major >= 1 and skl_minor >= 2:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        else:
            ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

        encoded = ohe.fit_transform(df[categorical])
        enc_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical))

        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(df[numerical])
        num_df = pd.DataFrame(num_scaled, columns=numerical)

        X = pd.concat([num_df, enc_df], axis=1)
        y = df["Churn"]

        logger.info(f"Feature matrix prepared. X shape={X.shape}, y shape={y.shape}")
        return X, y, ohe, scaler

    except Exception as e:
        logger.exception("Error during feature preparation")
        raise e
