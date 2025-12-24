import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger("INGESTION")

def ingest_data(source_url, dest_path):
    try:
        logger.info(" Starting data ingestion")
        df = pd.read_csv(source_url)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        df.to_csv(dest_path, index=False)
        logger.info(f" Data saved to {dest_path} with shape {df.shape}")
    except Exception as e:
        logger.exception(" Error during data ingestion")
        raise e

if __name__ == "__main__":
    ingest_data(
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/refs/heads/master/data/Telco-Customer-Churn.csv",
        "data/raw/Telco-Customer-Churn.csv",
    )
