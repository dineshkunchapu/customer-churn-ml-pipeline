import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("PREPROCESSING")

def clean_data(df):
    try:
        logger.info(" Starting data cleaning")
        df.columns = [c.strip() for c in df.columns]
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        df.drop(columns=["customerID"], inplace=True, errors="ignore")
        logger.info(f" Cleaning complete. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.exception(" Error in preprocessing")
        raise e
import os

if __name__ == "__main__":
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv("data/processed/cleaned.csv", index=False)

